from skimage.util import view_as_windows
from utils import utils_image as util
import numpy as np
import matplotlib.pyplot as plt

# Global random state - set once at module level
rng = None


def set_noise_estimation_seed(seed=42):
    global rng
    rng = np.random.default_rng(seed)
    np.random.seed(seed)


# Initialize with default seed
set_noise_estimation_seed(42)


def estimate_noise_local_variance(image, patch_size=7, percentile=10):
    # Extract all patches at once
    patches = view_as_windows(image, (patch_size, patch_size))

    # Calculate variance for all patches
    local_vars = np.var(patches, axis=(2, 3)).flatten()

    # Select smoothest regions (lowest variance)
    threshold = np.percentile(local_vars, percentile)
    smooth_regions_var = local_vars[local_vars <= threshold]

    # Estimate noise as median of variances in smooth regions
    noise_variance = np.median(smooth_regions_var)
    return np.sqrt(noise_variance)


def estimate_noise_pca(image, patch_size=7, num_patches=1000, num_similar=50):
    # Extract all possible patches
    patches = view_as_windows(image, (patch_size, patch_size))
    patches = patches.reshape(-1, patch_size * patch_size)

    # Randomly choose reference patches
    num_patches = min(num_patches, patches.shape[0])
    ref_indices = rng.choice(patches.shape[0], size=num_patches, replace=False)

    noise_estimates = []

    for idx in ref_indices:
        ref_patch = patches[idx]

        # Find similar patches (L2 distance)
        dists = np.sum((patches - ref_patch) ** 2, axis=1)
        similar_idx = np.argpartition(dists, num_similar)[:num_similar]
        group = patches[similar_idx]

        if group.shape[0] > 1:
            # Remove DC component from each patch
            group = group - np.mean(group, axis=1, keepdims=True)
            # Center across patches
            group -= np.mean(group, axis=0, keepdims=True)

            # PCA via SVD
            U, s, Vt = np.linalg.svd(group, full_matrices=False)
            eigs = (s ** 2) / (group.shape[0] - 1)

            # Estimate variance from the smallest few eigenvalues
            num_noise_components = max(1, patch_size * patch_size // 4)
            noise_var = np.median(eigs[:num_noise_components])

            if noise_var > 0:
                noise_estimates.append(np.sqrt(noise_var))

    return float(np.median(noise_estimates)) if noise_estimates else 0.0


def estimate_noise_patch_similarity(image, patch_size=7, stride=4):
    h, w = image.shape

    # Calculate output dimensions for strided patches
    out_h = (h - patch_size) // stride + 1
    out_w = (w - patch_size) // stride + 1

    # Extract all patches at once
    patches = view_as_windows(image, (patch_size, patch_size), step=stride)
    patches = patches.reshape(out_h * out_w, patch_size * patch_size)

    # For each patch, find its most similar patch and compute difference
    noise_estimates = []
    num_samples = min(1000, len(patches))

    for i in range(num_samples):
        ref_patch = patches[i]
        # Compute distances to all other patches
        distances = np.sum((patches - ref_patch) ** 2, axis=1)
        distances[i] = np.inf  # Exclude self

        # Find most similar patch
        min_idx = np.argmin(distances)
        similar_patch = patches[min_idx]

        # The difference between similar patches is mostly noise
        diff = ref_patch - similar_patch
        # Estimate noise from difference (divide by sqrt(2) because
        # we're looking at difference of two noisy signals)
        noise_std = np.std(diff) / np.sqrt(2)
        noise_estimates.append(noise_std)

    # Return robust estimate
    return np.median(noise_estimates)


def estimate_noise_local_variance_non_uniform(image, block_size=32, overlap=16, percentile=10):
    h, w = image.shape
    stride = block_size - overlap
    noise_map = np.zeros(image.shape)
    weight_map = np.zeros(image.shape)

    # Create smooth weight (Gaussian window)
    weight = _gaussian_window(block_size)

    # Process each block
    for i in range(0, h - block_size + 1, stride):
        for j in range(0, w - block_size + 1, stride):
            # Extract block
            block = image[i:i + block_size, j:j + block_size]

            # Estimate noise in this block using estimate_noise_local_variance
            noise_std = estimate_noise_local_variance(image=block)

            # Add to noise map with weights
            noise_map[i:i + block_size, j:j + block_size] += noise_std * weight
            weight_map[i:i + block_size, j:j + block_size] += weight

    # Normalize by weights
    noise_map = np.divide(noise_map, weight_map,
                          out=np.zeros_like(noise_map),
                          where=weight_map > 0)

    # Handle edges that weren't covered
    if np.any(weight_map == 0):
        # Fill uncovered areas with nearest values
        mask = weight_map == 0
        noise_map = _fill_missing_values(noise_map, mask)

    return noise_map

def estimate_noise_pca_non_uniform(image, patch_size=4, num_patches=20, num_similar=10,
                                   grid_step=8, local_window=32):

    h, w = image.shape
    half_window = local_window // 2

    # Initialize noise map and weight map for accumulation
    noise_map = np.zeros((h, w))
    weight_map = np.zeros((h, w))

    # Create Gaussian weight for smooth blending (same as method1)
    weight = _gaussian_window(local_window)

    # Sample on grid with reduced margin to get closer to edges
    margin = max(half_window, patch_size * 2)
    y_coords = np.arange(margin, h - margin, grid_step)
    x_coords = np.arange(margin, w - margin, grid_step)

    # Process each grid location
    for i, y in enumerate(y_coords):
        for j, x in enumerate(x_coords):
            # Extract local window around this point
            y_start = max(0, y - half_window)
            y_end = min(h, y + half_window)
            x_start = max(0, x - half_window)
            x_end = min(w, x + half_window)

            local_region = image[y_start:y_end, x_start:x_end]

            # Estimate noise in this local window using estimate_noise_pca
            local_noise = estimate_noise_pca(
                image=local_region,
                patch_size=patch_size,
                num_patches=num_patches,
                num_similar=num_similar
            )

            if local_noise > 0:
                # Add to noise map with gaussian weight (same as estimate_noise_local_variance_non_uniform)
                noise_map[y_start:y_end, x_start:x_end] += local_noise * weight[:y_end - y_start, :x_end - x_start]
                weight_map[y_start:y_end, x_start:x_end] += weight[:y_end - y_start, :x_end - x_start]

    # Normalize by weights (same as estimate_noise_local_variance_non_uniform)
    noise_map = np.divide(noise_map, weight_map,
                          out=np.zeros_like(noise_map),
                          where=weight_map > 0)

    # Handle edges that weren't covered (same as estimate_noise_local_variance_non_uniform)
    if np.any(weight_map == 0):
        mask = weight_map == 0
        noise_map = _fill_missing_values(noise_map, mask)

    return noise_map

# Helper functions:
def _gaussian_window(size):
    """Create 2D Gaussian window for smooth blending"""
    x = np.arange(size) - size // 2
    y = np.arange(size) - size // 2
    X, Y = np.meshgrid(x, y)
    sigma = size / 4
    window = np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))
    return window / window.max()


def _fill_missing_values(array, mask):
    """Fill missing values using nearest neighbor interpolation"""
    from scipy.ndimage import distance_transform_edt

    ind = distance_transform_edt(mask, return_distances=False, return_indices=True)
    return array[tuple(ind)]


if __name__ == "__main__":

    image = util.imread_uint('testsets/FFDNet_gray/01.png', n_channels=1)
    image = util.uint2single(image)
    image = np.squeeze(image)

    # Plot 3D noise map
    H, W = image.shape
    Y, X = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

    uniform_noise = True
    non_uniform_noise = True

    if uniform_noise:
        noise_level_img = 127.5 / 255.
        print(f"noise level image: {noise_level_img}")
        image_uniform_noise = image + np.random.normal(0, noise_level_img, image.shape)

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, np.full(image.shape, noise_level_img), cmap='viridis')
        ax.set_title("Uniform Gaussian Noise Map (sigma)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("sigma value")
        plt.show()

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap='gray')
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(image_uniform_noise, cmap='gray')
        plt.title("Image with uniform gaussian Noise")
        plt.axis('off')
        plt.show()

        estimated_noise_local_var = estimate_noise_local_variance(image=image_uniform_noise)
        estimated_noise_pca = estimate_noise_pca(image=image_uniform_noise)
        estimated_noise_patch_similarity = estimate_noise_patch_similarity(image=image_uniform_noise)

        print(f"estimated_noise_local_var: {estimated_noise_local_var}")
        print(f"estimated_noise_pca: {estimated_noise_pca}")
        print(f"estimated_noise_patch_similarity: {estimated_noise_patch_similarity}")

    if non_uniform_noise:
        # Define non-uniform noise limits
        sigma_min = 0.01  # low noise in center
        sigma_max = 0.1  # high noise at edges

        # normalize distance from center
        cy, cx = H // 2, W // 2
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        dist_norm = dist / dist.max()

        sigma_map = sigma_min + (sigma_max - sigma_min) * dist_norm

        # Create noon uniform noise
        noise = np.random.normal(0, sigma_map)

        # Noisy image
        image_non_uniform_noise = image + noise

        print(f"noise level range: {sigma_min}, {sigma_max}")

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, sigma_map, cmap='viridis')
        ax.set_title("Non-Uniform Gaussian Noise Map (sigma)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("sigma value")
        plt.show()

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap='gray')
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(image_non_uniform_noise, cmap='gray')
        plt.title("Image with non-uniform gaussian noise")
        plt.axis('off')
        plt.show()

        estimated_noise_local_var_non_uniform = estimate_noise_local_variance_non_uniform(image=image_non_uniform_noise)
        estimated_noise_pca_non_uniform = estimate_noise_pca_non_uniform(image=image_non_uniform_noise)

        print(f"estimated_noise_local_var_non_uniform range: {estimated_noise_local_var_non_uniform.min():.4f} to "
              f"{estimated_noise_local_var_non_uniform.max():.4f}")
        print(f"estimated_noise_pca_non_uniform range: {estimated_noise_pca_non_uniform.min():.4f} to "
              f"{estimated_noise_pca_non_uniform.max():.4f}")