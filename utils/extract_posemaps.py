#!/usr/bin/env python3
import os
import sys
import json
import gzip
import base64
import time
import argparse
import signal
import platform
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

# your utilities
from utils.annotate import annotate_video, build_full_hand_heatmap
from dataset import load_rgb_frames_from_video2, video_to_tensor


# --------------------------- Helpers: io/serialization ---------------------------

def u16_pack01(x: np.ndarray) -> bytes:
    """Quantize float in [0,1] to uint16 bytes."""
    x = np.nan_to_num(x, nan=0.0)
    x = np.clip(x, 0.0, 1.0)
    arr = (x * 65535.0 + 0.5).astype(np.uint16)
    return arr.tobytes()

def u8_pack01(x: np.ndarray) -> bytes:
    """Quantize float in [0,1] to uint8 bytes."""
    x = np.nan_to_num(x, nan=0.0)
    x = np.clip(x, 0.0, 1.0)
    arr = (x * 255.0 + 0.5).astype(np.uint8)
    return arr.tobytes()

def b64enc(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")

def atomic_write_lines_gz(tmp_path: Path, final_path: Path, lines: List[str]) -> None:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(tmp_path, "wt", encoding="utf-8") as f:
        for ln in lines:
            if ln and ln[-1] != "\n":
                ln = ln + "\n"
            f.write(ln)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, final_path)

def iter_existing_ids(out_dir: Path) -> set:
    """Scan existing gz shards and return set of processed video_ids."""
    done = set()
    for p in sorted(out_dir.glob("pose-*.jsonl.gz")):
        try:
            with gzip.open(p, "rt", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    vid = rec.get("video_id")
                    if vid:
                        done.add(vid)
        except Exception as e:
            # ignore a corrupt shard; you can manually delete it if needed
            print(f"[warn] could not read shard {p}: {e}", file=sys.stderr)
    return done


# --------------------------- Timeout (POSIX) ---------------------------

class Timeout:
    """POSIX-only SIGALRM timeout context. No-op on Windows."""
    def __init__(self, seconds: int):
        self.seconds = seconds
        self.enabled = (platform.system() != "Windows")

    def __enter__(self):
        if self.enabled:
            signal.signal(signal.SIGALRM, self._handler)
            signal.alarm(self.seconds)
        return self

    def _handler(self, signum, frame):
        raise TimeoutError("per-video timeout")

    def __exit__(self, exc_type, exc, tb):
        if self.enabled:
            signal.alarm(0)
        return False

def _kp_from_ndarray(landmarks_t, T, H, W):
    """
    landmarks_t: np.ndarray [T, 2, 21, 3] with (x,y,z) in [0,1], NaNs when missing.
    Returns dict of float32 arrays in [0,1]: Lx,Ly,Lc,Rx,Ry,Rc  (all [T,21])
    """
    arr = np.asarray(landmarks_t, dtype=np.float32)  # [T,2,21,3]
    if not (arr.ndim == 4 and arr.shape[1] == 2 and arr.shape[2] == 21 and arr.shape[3] >= 2):
        raise ValueError(f"unexpected shape {arr.shape}, expected [T,2,21,3]")

    x = arr[..., 0]  # [T,2,21]
    y = arr[..., 1]
    # Build confidence: 1 if both x,y are finite, else 0
    c = (~(np.isnan(x) | np.isnan(y))).astype(np.float32)

    # Replace NaNs with 0 so quantization is stable
    x = np.nan_to_num(x, nan=0.0)
    y = np.nan_to_num(y, nan=0.0)

    x = np.clip(x, 0.0, 1.0)
    y = np.clip(y, 0.0, 1.0)

    Lx, Rx = x[:, 0, :], x[:, 1, :]
    Ly, Ry = y[:, 0, :], y[:, 1, :]
    Lc, Rc = c[:, 0, :], c[:, 1, :]

    return {"Lx": Lx, "Ly": Ly, "Lc": Lc,
            "Rx": Rx, "Ry": Ry, "Rc": Rc}

# --------------------------- Per-video worker ---------------------------

def process_video_worker(video_path, num_frames=16, store="keypoints",
                         target_hw=(224,224), per_video_timeout=25):
    video_id = os.path.basename(video_path); H, W = target_hw
    try:
        with Timeout(per_video_timeout):
            imgs = load_rgb_frames_from_video2(video_path, target_frames=num_frames, resize=target_hw)

            # pad/trim → T
            T = imgs.shape[0]
            if T == 0:
                # emit zeros but mark ok so we can load later
                zeros_u16 = b64enc(np.zeros((num_frames,21), np.uint16).tobytes())
                zeros_u8  = b64enc(np.zeros((num_frames,21), np.uint8 ).tobytes())
                return {"video_id": video_id, "status": "ok", "T": num_frames, "H": H, "W": W,
                        "hands": {"L":{"x":zeros_u16,"y":zeros_u16,"c":zeros_u8},
                                  "R":{"x":zeros_u16,"y":zeros_u16,"c":zeros_u8}}}

            if T < num_frames:
                pad = np.repeat(imgs[-1:,...], num_frames - T, axis=0)
                imgs = np.concatenate([imgs, pad], axis=0)
            elif T > num_frames:
                imgs = imgs[:num_frames]
            T = num_frames

            ten = video_to_tensor(imgs)       # (C,T,H,W)
            ret_img = ten.permute(1,0,2,3)    # (T,C,H,W)

            lm = annotate_video(ret_img, draw_landmarks=False)   # np [T,2,21,3]

            if store == "heatmaps":
                heatmaps = build_full_hand_heatmap(lm)           # torch [T,2,H,W] or np
                hm = heatmaps.detach().cpu().numpy() if hasattr(heatmaps, "detach") else np.asarray(heatmaps)
                return {"video_id": video_id, "status": "ok", "T": T, "H": H, "W": W,
                        "heatmaps": b64enc(hm.astype(np.float32).tobytes())}

            # store == "keypoints"
            kp = _kp_from_ndarray(lm, T, H, W)                   # dict of [T,21] in [0,1]
            # quantize & base64
            Lx = u16_pack01(kp["Lx"]); Ly = u16_pack01(kp["Ly"]); Lc = u8_pack01(kp["Lc"])
            Rx = u16_pack01(kp["Rx"]); Ry = u16_pack01(kp["Ry"]); Rc = u8_pack01(kp["Rc"])
            return {
                "video_id": video_id, "status": "ok", "T": T, "H": H, "W": W,
                "hands": {
                    "L": {"x": b64enc(Lx), "y": b64enc(Ly), "c": b64enc(Lc)},
                    "R": {"x": b64enc(Rx), "y": b64enc(Ry), "c": b64enc(Rc)}
                }
            }

    except TimeoutError:
        # emit zeros but ok
        zeros_u16 = b64enc(np.zeros((num_frames,21), np.uint16).tobytes())
        zeros_u8  = b64enc(np.zeros((num_frames,21), np.uint8 ).tobytes())
        return {"video_id": video_id, "status": "ok", "T": num_frames, "H": H, "W": W,
                "hands": {"L":{"x":zeros_u16,"y":zeros_u16,"c":zeros_u8},
                          "R":{"x":zeros_u16,"y":zeros_u16,"c":zeros_u8}}}
    except Exception as e:
        zeros_u16 = b64enc(np.zeros((num_frames,21), np.uint16).tobytes())
        zeros_u8  = b64enc(np.zeros((num_frames,21), np.uint8 ).tobytes())
        return {"video_id": video_id, "status": "ok", "T": num_frames, "H": H, "W": W,
                "hands": {"L":{"x":zeros_u16,"y":zeros_u16,"c":zeros_u8},
                          "R":{"x":zeros_u16,"y":zeros_u16,"c":zeros_u8}}}


# --------------------------- Driver ---------------------------

def make_video_list(datadir: str, csvs: List[str]) -> List[str]:
    basenames = []
    for csv_path in csvs:
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        # try 'filepath', else first column
        for _, row in df.iterrows():
            bn = os.path.basename(row["filepath"]) if "filepath" in df.columns else os.path.basename(str(row[0]))
            basenames.append(bn)
    uniq = sorted(set(basenames))
    paths = [os.path.join(datadir, bn) for bn in uniq if os.path.exists(os.path.join(datadir, bn))]
    return paths

def main():
    ap = argparse.ArgumentParser("Extract posemaps/keypoints to sharded JSONL.GZ (resumable, safe)")
    ap.add_argument("--datadir", type=str, required=True, help="Directory containing video files")
    ap.add_argument("--csv", type=str, nargs="+", required=True, help="CSV files with filepaths/basenames")
    ap.add_argument("--outdir", type=str, default="./pose_out", help="Output dir for shards")
    ap.add_argument("--num-frames", type=int, default=16)
    ap.add_argument("--store", type=str, choices=["keypoints","heatmaps"], default="keypoints",
                    help="Store compact keypoints (recommended) or full heatmaps")
    ap.add_argument("--shard-size", type=int, default=1000, help="Records per shard")
    ap.add_argument("--max-workers", type=int, default=None, help="Process pool size (default: CPU count)")
    ap.add_argument("--timeout", type=int, default=25, help="Per-video timeout (seconds; POSIX only)")
    args = ap.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    video_paths = make_video_list(args.datadir, args.csv)
    if not video_paths:
        print("No videos found.", file=sys.stderr)
        sys.exit(1)

    # Resume: build set of already processed ids
    done_ids = iter_existing_ids(out_dir)
    todo = [vp for vp in video_paths if os.path.basename(vp) not in done_ids]
    print(f"Total unique videos: {len(video_paths)} | already done: {len(done_ids)} | remaining: {len(todo)}")

    if not todo:
        print("All done. Nothing to process.")
        return

    max_workers = args.max_workers or os.cpu_count() or 4
    shard_size = max(1, args.shard_size)

    shard_idx = 0
    # Advance shard index to avoid overwriting existing shards
    while (out_dir / f"pose-{shard_idx:05d}.jsonl.gz").exists():
        shard_idx += 1

    buf: List[str] = []
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        fut_map = {ex.submit(process_video_worker, vp, args.num_frames, args.store, (224,224), args.timeout): vp for vp in todo}
        processed = 0

        for i, fut in enumerate(as_completed(fut_map), 1):
            rec = fut.result()
            buf.append(json.dumps(rec, separators=(",", ":")))

            # flush shard
            if len(buf) >= shard_size:
                tmp = out_dir / f"pose-{shard_idx:05d}.jsonl.gz.tmp"
                fin = out_dir / f"pose-{shard_idx:05d}.jsonl.gz"
                atomic_write_lines_gz(tmp, fin, buf)
                buf.clear()
                shard_idx += 1

            processed += 1
            if processed % 100 == 0:
                elapsed = time.time() - t0
                rate = processed / max(elapsed, 1e-6)
                print(f"[progress] processed {processed}/{len(todo)}  |  {rate:.2f} vids/s")

    # flush tail
    if buf:
        tmp = out_dir / f"pose-{shard_idx:05d}.jsonl.gz.tmp"
        fin = out_dir / f"pose-{shard_idx:05d}.jsonl.gz"
        atomic_write_lines_gz(tmp, fin, buf)
        buf.clear()

    elapsed = time.time() - t0
    print(f"Done. Processed {len(todo)} videos in {elapsed:.1f}s  (~{len(todo)/max(elapsed,1e-6):.2f} vids/s)")
    print(f"Shards written to: {out_dir.resolve()}")


if __name__ == "__main__":
    # Safer on some platforms for MP + MediaPipe
    try:
        import multiprocessing as mp
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()


    """rec = process_video_worker("/path/to/some.mp4", num_frames=16, store="keypoints")
    print(rec["status"], "hands" in rec) 
    print(rec["hands"]["L"].keys(), rec["hands"]["R"].keys())"""