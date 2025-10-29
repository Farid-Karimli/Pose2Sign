from torchvision import transforms
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn as nn
from skimage.metrics import structural_similarity as ssim
import os
from scipy import linalg

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: lpips not available. Install with: pip install lpips")

try:
    from torchvision.models import inception_v3, Inception_V3_Weights
    INCEPTION_AVAILABLE = True
except ImportError:
    INCEPTION_AVAILABLE = False
    print("Warning: torchvision inception_v3 not available for FID calculation")

# --- Helper functions ---
mse_loss_fn = nn.MSELoss()
to_tensor = transforms.ToTensor()

# Initialize LPIPS model (lazy initialization)
_lpips_model = None
def get_lpips_model(device='cuda'):
    """Lazy initialization of LPIPS model."""
    global _lpips_model
    if _lpips_model is None and LPIPS_AVAILABLE:
        _lpips_model = lpips.LPIPS(net='alex').to(device)
        _lpips_model.eval()
    return _lpips_model

# Initialize Inception model for FID (lazy initialization)
_inception_model = None
def get_inception_model(device='cuda'):
    """Lazy initialization of Inception V3 model for FID calculation."""
    global _inception_model
    if _inception_model is None and INCEPTION_AVAILABLE:
        _inception_model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
        _inception_model.fc = nn.Identity()  # Remove final FC layer
        _inception_model = _inception_model.to(device)
        _inception_model.eval()
    return _inception_model

def load_video_as_frames(video_path):
    """Load video and return a list of frames as NumPy arrays (H,W,C) in RGB order."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # OpenCV loads as BGR, convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()
    return frames


def resize_videos_to_common_size(vid1, vid2, target_size=None):
    """
    Resize both videos to a common size.
    If target_size is None, use the smaller dimensions of the two videos.
    
    Args:
        vid1, vid2: list of numpy arrays (H,W,C)
        target_size: tuple (width, height) or None
    
    Returns:
        resized_vid1, resized_vid2: lists of resized frames
    """
    if target_size is None:
        # Get dimensions of first frames
        h1, w1 = vid1[0].shape[:2]
        h2, w2 = vid2[0].shape[:2]
        
        # Use minimum dimensions to avoid upscaling
        target_w = min(w1, w2)
        target_h = min(h1, h2)
        target_size = (target_w, target_h)
    
    print(f"Resizing videos to common size: {target_size}")
    
    resized_vid1 = [cv2.resize(frame, target_size) for frame in vid1]
    resized_vid2 = [cv2.resize(frame, target_size) for frame in vid2]
    
    return resized_vid1, resized_vid2


def match_frame_counts(vid1, vid2):
    """
    Trim videos to have the same number of frames (use minimum length).
    
    Returns:
        vid1, vid2: trimmed to same length
    """
    min_frames = min(len(vid1), len(vid2))
    if len(vid1) != len(vid2):
        print(f"Frame count mismatch: vid1={len(vid1)}, vid2={len(vid2)}. Using first {min_frames} frames.")
    return vid1[:min_frames], vid2[:min_frames]


def calculate_ssim(vid1, vid2):
    """
    Calculate average SSIM between two lists of frames.
    
    Args:
        vid1, vid2: list of numpy arrays (H,W,C) in RGB, uint8 or float32
    
    Returns:
        average SSIM value
    """
    num_frames = len(vid1)
    ssim_list = []

    for i in range(num_frames):
        img1 = vid1[i].astype(np.float32)
        img2 = vid2[i].astype(np.float32)
        
        # Normalize to [0, 1] if needed
        if img1.max() > 1.0:
            img1 = img1 / 255.0
        if img2.max() > 1.0:
            img2 = img2 / 255.0

        # Ensure image is large enough for default window size
        min_side = min(img1.shape[0], img1.shape[1])
        win_size = 7 if min_side >= 7 else (min_side if min_side % 2 == 1 else min_side - 1)
        if win_size < 3:
            win_size = 3

        ssim_i = ssim(img1, img2, data_range=1.0, channel_axis=2, win_size=win_size)
        ssim_list.append(ssim_i)

    return np.mean(ssim_list)


def calculate_psnr(vid1, vid2):
    """
    Calculate peak signal-to-noise ratio (PSNR) between two videos.

    Args:
        vid1, vid2: list of numpy arrays (H,W,C) in RGB

    Returns:
        average PSNR value in dB
    """
    num_frames = len(vid1)
    psnr_list = []

    for i in range(num_frames):
        img1 = to_tensor(vid1[i])
        img2 = to_tensor(vid2[i])

        mse = mse_loss_fn(img1, img2).item()

        if mse == 0:
            psnr_i = 100.0
        else:
            # PSNR formula: 10 * log10(MAX^2 / MSE)
            # For normalized images [0,1], MAX=1, so MAX^2=1
            psnr_i = 10 * np.log10(1.0 / mse)

        psnr_list.append(psnr_i)

    return np.mean(psnr_list)


def calculate_lpips(vid1, vid2, device='cuda'):
    """
    Calculate LPIPS (Learned Perceptual Image Patch Similarity) between two videos.
    Lower is better (0 = identical images).

    Args:
        vid1, vid2: list of numpy arrays (H,W,C) in RGB, uint8 [0-255]
        device: device to run the model on ('cuda' or 'cpu')

    Returns:
        average LPIPS value across all frames
    """
    if not LPIPS_AVAILABLE:
        raise ImportError("LPIPS not available. Install with: pip install lpips")

    lpips_model = get_lpips_model(device)
    num_frames = len(vid1)
    lpips_list = []

    with torch.no_grad():
        for i in range(num_frames):
            # Convert to tensor and normalize to [-1, 1]
            img1 = to_tensor(vid1[i]).unsqueeze(0).to(device)
            img2 = to_tensor(vid2[i]).unsqueeze(0).to(device)

            # Normalize from [0,1] to [-1,1]
            img1 = img1 * 2 - 1
            img2 = img2 * 2 - 1

            # Calculate LPIPS
            lpips_value = lpips_model(img1, img2).item()
            lpips_list.append(lpips_value)

    return np.mean(lpips_list)


def calculate_mse(vid1, vid2, normalized=False):
    """
    Calculate mean squared error between two videos.

    Args:
        vid1, vid2: list of numpy arrays (H,W,C) in RGB, uint8 [0-255]
        normalized: if True, normalize to [0,1] before computing MSE
                   if False, compute MSE on raw pixel values [0-255]

    Returns:
        mse_list: list of MSE values per frame
        mean_mse: average MSE across all frames
    """
    num_frames = len(vid1)
    assert num_frames == len(vid2), "Videos must have the same number of frames"

    mse_list = []

    for i in range(num_frames):
        if normalized:
            # Normalize to [0,1]
            img1 = to_tensor(vid1[i])
            img2 = to_tensor(vid2[i])
            mse = torch.mean((img1 - img2) ** 2).item()
        else:
            # Raw pixel MSE (no normalization)
            img1 = vid1[i].astype(np.float32)
            img2 = vid2[i].astype(np.float32)
            mse = np.mean((img1 - img2) ** 2)

        mse_list.append(mse)

    mean_mse = np.mean(mse_list)
    return mse_list, mean_mse


def extract_inception_features(images, device='cuda', batch_size=32):
    """
    Extract Inception V3 features from a list of images.

    Args:
        images: list of numpy arrays (H,W,C) in RGB, uint8 [0-255]
        device: device to run the model on ('cuda' or 'cpu')
        batch_size: batch size for feature extraction

    Returns:
        numpy array of features (N, 2048)
    """
    if not INCEPTION_AVAILABLE:
        raise ImportError("Inception V3 not available for FID calculation")

    inception_model = get_inception_model(device)
    features_list = []

    # Preprocessing for Inception V3
    # Resize to 299x299 and normalize
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            batch_tensors = torch.stack([preprocess(img) for img in batch]).to(device)

            # Extract features
            features = inception_model(batch_tensors)
            features_list.append(features.cpu().numpy())

    return np.concatenate(features_list, axis=0)


def calculate_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Calculate Frechet Inception Distance (FID) between two distributions.

    Args:
        mu1, mu2: mean of the features
        sigma1, sigma2: covariance matrix of the features
        eps: epsilon for numerical stability

    Returns:
        FID score (lower is better)
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calculate_fid_from_videos(vid1, vid2, device='cuda', batch_size=32):
    """
    Calculate FID score between two videos by treating all frames as a distribution.

    Args:
        vid1, vid2: list of numpy arrays (H,W,C) in RGB, uint8 [0-255]
        device: device to run the model on ('cuda' or 'cpu')
        batch_size: batch size for feature extraction

    Returns:
        FID score (lower is better)
    """
    # Extract features
    features1 = extract_inception_features(vid1, device=device, batch_size=batch_size)
    features2 = extract_inception_features(vid2, device=device, batch_size=batch_size)

    # Calculate statistics
    mu1, sigma1 = np.mean(features1, axis=0), np.cov(features1, rowvar=False)
    mu2, sigma2 = np.mean(features2, axis=0), np.cov(features2, rowvar=False)

    # Calculate FID
    fid_score = calculate_fid(mu1, sigma1, mu2, sigma2)

    return fid_score


def calculate_vfid(vid1, vid2, device='cuda', batch_size=32):
    """
    Calculate Video FID (VFID) - frame-wise FID averaged over temporal dimension.
    This treats corresponding frames as distributions and averages FID across time.

    Args:
        vid1, vid2: list of numpy arrays (H,W,C) in RGB, uint8 [0-255]
        device: device to run the model on ('cuda' or 'cpu')
        batch_size: batch size for feature extraction

    Returns:
        VFID score (lower is better)
    """
    if not INCEPTION_AVAILABLE:
        raise ImportError("Inception V3 not available for VFID calculation")

    num_frames = len(vid1)
    assert num_frames == len(vid2), "Videos must have the same number of frames"

    # Extract features for all frames at once
    features1 = extract_inception_features(vid1, device=device, batch_size=batch_size)
    features2 = extract_inception_features(vid2, device=device, batch_size=batch_size)

    # For VFID, we calculate statistics over all frames together
    # (treating the video as a single distribution of frame features)
    mu1, sigma1 = np.mean(features1, axis=0), np.cov(features1, rowvar=False)
    mu2, sigma2 = np.mean(features2, axis=0), np.cov(features2, rowvar=False)

    vfid_score = calculate_fid(mu1, sigma1, mu2, sigma2)

    return vfid_score


def export_to_gif(frames, output_gif_path, fps=7):
    """Saves a list of PIL Images or numpy arrays as a GIF."""
    pil_frames = [Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame for frame in frames]
    pil_frames[0].save(
        output_gif_path, format='GIF', append_images=pil_frames[1:],
        save_all=True, duration=1000 // fps, loop=0
    )


def export_to_video(video_frames, output_video_path, fps):
    """Saves a list of numpy arrays as an MP4 video."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w, _ = video_frames[0].shape
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps=fps, frameSize=(w, h))
    for frame in video_frames:
        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(img)
    video_writer.release()


def tensor_to_pil(tensor):
    """Converts a [-1, 1] PyTorch tensor to a PIL Image."""
    tensor = (tensor + 1.0) / 2.0
    tensor = torch.clamp(tensor, 0, 1)
    numpy_array = tensor.permute(1, 2, 0).cpu().numpy()
    image_array = (numpy_array * 255).astype(np.uint8)
    return Image.fromarray(image_array)


def compare_videos(inference_path, gt_path, target_size=None, device='cuda',
                   compute_lpips=True, compute_fid=True, compute_vfid=True):
    """
    Load two videos and compute comparison metrics.

    Args:
        inference_path: path to inference video
        gt_path: path to ground truth video
        target_size: optional (width, height) tuple for resizing, or None for auto
        device: device to run neural network models on ('cuda' or 'cpu')
        compute_lpips: whether to compute LPIPS (requires lpips package)
        compute_fid: whether to compute FID (requires torchvision)
        compute_vfid: whether to compute VFID (requires torchvision)

    Returns:
        dict with metrics: avg_mse, avg_psnr, avg_ssim, avg_lpips, fid, vfid
    """
    # Load videos
    loaded_inference = load_video_as_frames(inference_path)
    print(f"Loaded inference video: {len(loaded_inference)} frames, size={loaded_inference[0].shape}")

    loaded_gt = load_video_as_frames(gt_path)
    print(f"Loaded ground truth video: {len(loaded_gt)} frames, size={loaded_gt[0].shape}")

    # Match frame counts
    loaded_inference, loaded_gt = match_frame_counts(loaded_inference, loaded_gt)

    # Resize to common size
    resized_inference, resized_gt = resize_videos_to_common_size(
        loaded_inference, loaded_gt, target_size=target_size
    )

    # Compute metrics
    mse_per_frame, avg_mse = calculate_mse(resized_inference, resized_gt)
    psnr = calculate_psnr(resized_inference, resized_gt)
    ssim_value = calculate_ssim(resized_inference, resized_gt)

    print(f"\nMetrics:")
    print(f"  Avg MSE:  {avg_mse:.6f}")
    print(f"  Avg PSNR: {psnr:.2f} dB")
    print(f"  Avg SSIM: {ssim_value:.4f}")

    results = {
        'avg_mse': avg_mse,
        'avg_psnr': psnr,
        'avg_ssim': ssim_value,
        'mse_per_frame': mse_per_frame
    }

    # Compute LPIPS if requested
    if compute_lpips and LPIPS_AVAILABLE:
        try:
            lpips_value = calculate_lpips(resized_inference, resized_gt, device=device)
            results['avg_lpips'] = lpips_value
            print(f"  Avg LPIPS: {lpips_value:.4f}")
        except Exception as e:
            print(f"  LPIPS calculation failed: {e}")
            results['avg_lpips'] = None
    else:
        results['avg_lpips'] = None

    # Compute FID if requested
    if compute_fid and INCEPTION_AVAILABLE:
        try:
            fid_value = calculate_fid_from_videos(resized_inference, resized_gt, device=device)
            results['fid'] = fid_value
            print(f"  FID: {fid_value:.4f}")
        except Exception as e:
            print(f"  FID calculation failed: {e}")
            results['fid'] = None
    else:
        results['fid'] = None

    # Compute VFID if requested
    if compute_vfid and INCEPTION_AVAILABLE:
        try:
            vfid_value = calculate_vfid(resized_inference, resized_gt, device=device)
            results['vfid'] = vfid_value
            print(f"  VFID: {vfid_value:.4f}")
        except Exception as e:
            print(f"  VFID calculation failed: {e}")
            results['vfid'] = None
    else:
        results['vfid'] = None

    return results


def compare_multiple_video_pairs(video_pairs, target_size=None, device='cuda',
                                 compute_lpips=True, compute_fid=True, compute_vfid=True):
    """
    Compare multiple pairs of videos and compute metrics for each pair.

    Args:
        video_pairs: list of tuples [(inference_path1, gt_path1), (inference_path2, gt_path2), ...]
        target_size: optional (width, height) tuple for resizing, or None for auto
        device: device to run neural network models on ('cuda' or 'cpu')
        compute_lpips: whether to compute LPIPS (requires lpips package)
        compute_fid: whether to compute FID (requires torchvision)
        compute_vfid: whether to compute VFID (requires torchvision)

    Returns:
        list of dicts containing metrics for each pair
    """
    all_results = []

    for idx, (inference_path, gt_path) in enumerate(video_pairs):
        print(f"\n{'='*60}")
        print(f"Processing pair {idx + 1}/{len(video_pairs)}")
        print(f"Inference: {inference_path}")
        print(f"Ground Truth: {gt_path}")
        print(f"{'='*60}")

        try:
            metrics = compare_videos(inference_path, gt_path, target_size=target_size,
                                    device=device, compute_lpips=compute_lpips,
                                    compute_fid=compute_fid, compute_vfid=compute_vfid)
            metrics['inference_path'] = inference_path
            metrics['gt_path'] = gt_path
            metrics['pair_index'] = idx
            all_results.append(metrics)
        except Exception as e:
            print(f"Error processing pair {idx + 1}: {str(e)}")
            all_results.append({
                'inference_path': inference_path,
                'gt_path': gt_path,
                'pair_index': idx,
                'error': str(e)
            })

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY OF ALL PAIRS")
    print(f"{'='*60}")

    valid_results = [r for r in all_results if 'error' not in r]

    if valid_results:
        avg_mse_all = np.mean([r['avg_mse'] for r in valid_results])
        avg_psnr_all = np.mean([r['avg_psnr'] for r in valid_results])
        avg_ssim_all = np.mean([r['avg_ssim'] for r in valid_results])

        print(f"Successfully processed {len(valid_results)}/{len(video_pairs)} pairs")
        print(f"\nAverage across all pairs:")
        print(f"  Avg MSE:  {avg_mse_all:.6f}")
        print(f"  Avg PSNR: {avg_psnr_all:.2f} dB")
        print(f"  Avg SSIM: {avg_ssim_all:.4f}")

        # Print LPIPS if available
        if compute_lpips and any(r.get('avg_lpips') is not None for r in valid_results):
            lpips_values = [r['avg_lpips'] for r in valid_results if r.get('avg_lpips') is not None]
            avg_lpips_all = np.mean(lpips_values)
            print(f"  Avg LPIPS: {avg_lpips_all:.4f}")

        # Print FID if available
        if compute_fid and any(r.get('fid') is not None for r in valid_results):
            fid_values = [r['fid'] for r in valid_results if r.get('fid') is not None]
            avg_fid_all = np.mean(fid_values)
            print(f"  Avg FID: {avg_fid_all:.4f}")

        # Print VFID if available
        if compute_vfid and any(r.get('vfid') is not None for r in valid_results):
            vfid_values = [r['vfid'] for r in valid_results if r.get('vfid') is not None]
            avg_vfid_all = np.mean(vfid_values)
            print(f"  Avg VFID: {avg_vfid_all:.4f}")

        print(f"\nPer-pair results:")
        for r in valid_results:
            result_str = f"  Pair {r['pair_index'] + 1}: MSE={r['avg_mse']:.6f}, PSNR={r['avg_psnr']:.2f} dB, SSIM={r['avg_ssim']:.4f}"
            if r.get('avg_lpips') is not None:
                result_str += f", LPIPS={r['avg_lpips']:.4f}"
            if r.get('fid') is not None:
                result_str += f", FID={r['fid']:.4f}"
            if r.get('vfid') is not None:
                result_str += f", VFID={r['vfid']:.4f}"
            print(result_str)

    if len(valid_results) < len(video_pairs):
        print(f"\nFailed to process {len(video_pairs) - len(valid_results)} pairs")

    return all_results


# Example usage
if __name__ == "__main__":
    # Single pair example
    inference_path = '/restricted/projectnb/cs599dg/Pose2Sign/Hyper-Motion/samples/zero_shot/TRANSFER_mp_prompt1.mp4'
    gt_path = '/restricted/projectnb/cs599dg/Pose2Sign/Hyper-Motion/samples/zero_shot/45548435897858-TRANSFER.mp4'
    
    # Option 1: Single pair
    # metrics = compare_videos(inference_path, gt_path)
    
    # Option 2: Multiple pairs
    VDM_RESULTS = "/restricted/projectnb/cs599dg/Pose2Sign/VDM_EVFI/scripts"

    video_pairs = [
        ("/restricted/projectnb/cs599dg/Pose2Sign/Hyper-Motion/samples/zero_shot/TRANSFER_mp_prompt1.mp4","/restricted/projectnb/cs599dg/Pose2Sign/Hyper-Motion/samples/zero_shot/45548435897858-TRANSFER.mp4"),
        ("/restricted/projectnb/cs599dg/Pose2Sign/Hyper-Motion/samples/zero_shot/LICENSE_mp_prompt2.mp4","/restricted/projectnb/cs599dg/Pose2Sign/Hyper-Motion/samples/zero_shot/81602328051681-LICENSE.mp4"),
        ("/restricted/projectnb/cs599dg/Pose2Sign/Hyper-Motion/samples/zero_shot/GATE_mp_prompt2.mp4", "/restricted/projectnb/cs599dg/Pose2Sign/Hyper-Motion/samples/zero_shot/868454966389171-GATE.mp4")
    ]


    # video_pairs = [
    #     (os.path.join(VDM_RESULTS, f"generated_from_item_{x}.gif_{x}_gt.mp4"), os.path.join(VDM_RESULTS, f"generated_from_item_{x}.gif_{x}_inference.mp4")) for x in range(0,30)
    # ]
    
    all_results = compare_multiple_video_pairs(video_pairs)
    
    # Option 3: Multiple pairs with specific target size
    # all_results = compare_multiple_video_pairs(video_pairs, target_size=(640, 480))