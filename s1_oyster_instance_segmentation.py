# s1_oyster_instance_segmentation.py
#
# Description:
# This script performs Stage 1 of the oyster analysis pipeline:
# 1. Loads a Whole Slide Image (WSI) of oyster tissue.
# 2. Preprocesses the image using adaptive thresholding and morphological operations
#    to create a binary mask highlighting tissue regions.
# 3. Detects contours from the binary mask and selects the two largest contours,
#    presumed to be the two main oyster sections.
# 4. Generates bounding boxes from these selected contours.
# 5. Uses the Segment Anything Model (SAM) with these bounding box prompts
#    to generate precise instance segmentation masks for each oyster.
# 6. Saves the resulting masks.

# Usage:
# 1. Ensure all dependencies are installed (OpenCV, NumPy, Matplotlib, PyTorch, segment-anything).
# 2. Download a SAM checkpoint file (e.g., sam_vit_h_4b8939.pth).
# 3. Update the configuration variables in the SCRIPT_CONFIG section below.
# 4. Run the script: python stage1_oyster_instance_segmentation.py

import numpy as np
import tifffile
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__() # Allow large images
import cv2
import matplotlib.pyplot as plt
import torch
from segment_anything import sam_model_registry, SamPredictor
import glob
import logging
import time

# --- SCRIPT CONFIGURATION ---
# INPUT_WSI_DIR = "data/downsampled_wsis/"  # Path to the directory containing downsampled WSI images
INPUT_WSI_DIR = "/Volumes/One Touch/MSX Project/TIFF/test"
VALID_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")  # Supported image formats
SAM_CHECKPOINT_PATH = (
    "pretrained_checkpoint/sam_vit_h_4b8939.pth"  # Path to SAM checkpoint
)
SAM_MODEL_TYPE = "vit_h"  # SAM model type (e.g., "vit_h", "vit_l", "vit_b")
OUTPUT_MASK_PARENT_DIR = "output_stage1_masks_logging"  # Parent directory to save masks for all outputs
STAGE1_PROCESSING_DOWNSAMPLE = 32.0     # Downsample factor for processing WSI images
DEBUG_VISUALIZATION_MAX_DIM = 1024       # Max dimension (width or height) for saved debug images. Adjust as needed.
MANUAL_WSI_OBJECTIVE_POWER = 20.0       # TODO: Same as s2

# Morphological operation parameters
GAUSSIAN_BLUR_KERNEL_SIZE = (3, 3)
ADAPTIVE_THRESH_BLOCK_SIZE = 11  # Must be odd
ADAPTIVE_THRESH_C = 2
MORPH_CLOSE_KERNEL_SIZE = (5, 5)
MORPH_CLOSE_ITERATIONS = 3
MORPH_OPEN_KERNEL_SIZE = (3, 3)
MORPH_OPEN_ITERATIONS = 2
MIN_CONTOUR_AREA_PERCENTAGE = 0.01
NUM_OYSTERS_TO_DETECT = 2  # Number of oysters to detect per image

# Control plotting for debugging/visualization (set to False to disable plots)
ENABLE_PLOTTING = False
# --- END SCRIPT CONFIGURATION ---

# --- Global Logger ---
logger = logging.getLogger(__name__)

def setup_logging(log_dir):
    """
    Configures logging to console and file.
    :param log_dir:
    :return: None
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_file_path = os.path.join(log_dir, f"stage1_processing_{timestamp}.log")

    logger.setLevel(logging.DEBUG)  # Set global minimum level

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)  # Show DEBUG and above on console
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File Handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)  # Log DEBUG and above to file
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    logger.info("Logging setup complete. Log file: %s", log_file_path)


# --- Helper Functions ---
def plot_or_save_debug_image(image_data, title, filename_suffix, current_wsi_output_dir, image_base_name, cmap=None):
    """
    Utility to either plot an image or save it to a 'debug_images' subdirectory.
    Handles RGB and grayscale images for saving. Includes logic to downsample
    images for saving if ENABLE_PLOTTING is False, to create smaller debug files.

    :param image_data: The image data to plot or save.
    :param title: Title for the plot (if plotting).
    :param filename_suffix: Suffix for the saved filename.
    :param current_wsi_output_dir: Directory to save the debug images.
    :param image_base_name: Base name of the original image for naming.
    :param cmap: Colormap to use for plotting (if applicable, e.g., 'gray' for grayscale).
    """
    if ENABLE_PLOTTING:
        plt.figure(figsize=(8, 8) if cmap else (10,10))
        if cmap:
            plt.imshow(image_data, cmap=cmap)
        else:
            plt.imshow(image_data)
        plt.title(title)
        plt.axis("off")
        plt.show()
    else:
        debug_img_dir = os.path.join(current_wsi_output_dir, "debug_images")
        os.makedirs(debug_img_dir, exist_ok=True)
        save_path = os.path.join(debug_img_dir, f"{image_base_name}_{filename_suffix}.png")

        # --- NEW: Resize image for debug saving ---
        current_height, current_width = image_data.shape[0], image_data.shape[1]
        if max(current_height, current_width) > DEBUG_VISUALIZATION_MAX_DIM:
            # Calculate new dimensions while maintaining aspect ratio
            if current_width > current_height:
                new_width = DEBUG_VISUALIZATION_MAX_DIM
                new_height = int(current_height * (new_width / current_width))
            else:
                new_height = DEBUG_VISUALIZATION_MAX_DIM
                new_width = int(current_width * (new_height / current_height))

            # Ensure new_width and new_height are at least 1 to avoid errors
            new_width = max(1, new_width)
            new_height = max(1, new_height)

            resized_image_data = cv2.resize(image_data, (new_width, new_height), interpolation=cv2.INTER_AREA)
            logger.debug(f"Resized debug image from {current_width}x{current_height} to {new_width}x{new_height} for saving.")
        else:
            resized_image_data = image_data # No resizing needed
        # --- END NEW ---

        try:
            if resized_image_data.ndim == 3 and resized_image_data.shape[2] == 3: # RGB
                cv2.imwrite(save_path, cv2.cvtColor(resized_image_data, cv2.COLOR_RGB2BGR))
            elif resized_image_data.ndim == 2: # Grayscale or binary
                cv2.imwrite(save_path, resized_image_data)
            else:
                logger.warning(f"Unsupported image format for saving debug image: {filename_suffix}, shape: {resized_image_data.shape}")
                return
            logger.debug(f"Debug image saved: {save_path}") # Changed from "WOULD HAVE BEEN"
        except Exception as e:
            logger.error(f"🔴 Error saving debug image {save_path}: {e}")


def extract_downsampled_overview_from_ome_tiff(
        ome_tiff_path,
        target_processing_downsample,  # Renamed for clarity
        manual_objective_power=MANUAL_WSI_OBJECTIVE_POWER
):
    logger.info(f"Extracting downsampled overview from: {ome_tiff_path}")
    logger.info(f"Target processing downsample factor: {target_processing_downsample}")
    try:
        with tifffile.TiffFile(ome_tiff_path) as tif:
            if not tif.is_ome:
                logger.warning("TIFF is not an OME-TIFF per tifffile.")

            if not tif.series or not tif.series[0].levels:
                logger.error(f"No image series or levels found in {ome_tiff_path}")
                return None

            main_series = tif.series[0]

            # Get actual downsamples from the TIFF structure if possible,
            # otherwise rely on common powers of 2.
            # For OME-TIFF, often the levels are just numbered 0, 1, 2...
            # and their downsample is implicitly 2^level_index.
            # We know from your previous run that QuPath/tifffile saw [1,2,4,8,16,32,64,128,256]
            # So, we can use these as the *effective* downsamples for these levels.

            available_downsamples = []
            qupath_like_downsamples = [2 ** i for i in range(len(main_series.levels))]  # Assumes powers of 2

            # Let's verify this against the actual dimensions for more robustness
            base_width, base_height = main_series.levels[0].shape[1], main_series.levels[0].shape[0]
            for i in range(len(main_series.levels)):
                lvl_width, lvl_height = main_series.levels[i].shape[1], main_series.levels[i].shape[0]
                # Calculate effective downsample based on width (or height)
                ds_w = base_width / lvl_width if lvl_width > 0 else float('inf')
                ds_h = base_height / lvl_height if lvl_height > 0 else float('inf')
                # Use the more conservative (larger) downsample if dimensions aren't perfectly scaled
                # or average them. For now, let's use the one derived from width.
                # This also assumes levels are ordered from highest to lowest resolution.
                if i == 0:
                    available_downsamples.append(1.0)
                else:
                    # Try to match with QuPath-like, or use calculated
                    if i < len(qupath_like_downsamples) and abs(ds_w - qupath_like_downsamples[i]) < 0.1 * \
                            qupath_like_downsamples[i]:  # within 10%
                        available_downsamples.append(float(qupath_like_downsamples[i]))
                    else:  # Fallback to calculated if not matching typical powers of 2
                        available_downsamples.append(round(ds_w, 2))  # Round to avoid minor float issues

            logger.info(f"Available effective downsamples in TIFF: {available_downsamples}")

            best_level_idx = -1
            # Find the smallest downsample factor that is >= target_processing_downsample
            # This ensures the resulting image is small enough.
            for i, ds in enumerate(available_downsamples):
                if ds >= target_processing_downsample:
                    best_level_idx = i
                    break

            if best_level_idx == -1:  # Target downsample is larger than any available
                if available_downsamples:
                    best_level_idx = len(available_downsamples) - 1  # Pick the lowest resolution available
                    logger.warning(
                        f"Target downsample {target_processing_downsample}x is too high. "
                        f"Using lowest available resolution (Level {best_level_idx}, ~{available_downsamples[best_level_idx]:.1f}x downsample)."
                    )
                else:
                    logger.error("No downsample levels found or processed correctly.")
                    return None

            actual_downsample_selected = available_downsamples[best_level_idx]
            logger.info(
                f"Selected Level {best_level_idx} with actual downsample ~{actual_downsample_selected:.1f}x for processing."
            )
            page_to_read = main_series.levels[best_level_idx]
            overview_image_raw = page_to_read.asarray()
            # ... (rest of your color conversion logic) ...
            logger.info(f"Raw overview image shape: {overview_image_raw.shape}")

            # Convert to RGB if necessary
            if overview_image_raw.ndim == 3 and overview_image_raw.shape[2] == 4:  # RGBA
                overview_rgb = cv2.cvtColor(overview_image_raw, cv2.COLOR_RGBA2RGB)
            elif overview_image_raw.ndim == 3 and overview_image_raw.shape[2] == 3:  # RGB
                overview_rgb = overview_image_raw
            elif overview_image_raw.ndim == 2:  # Grayscale
                overview_rgb = cv2.cvtColor(overview_image_raw, cv2.COLOR_GRAY2RGB)
            else:
                logger.error(f"Unsupported image format from TIFF page: {overview_image_raw.shape}")
                return None

            logger.info(f"Extracted RGB overview shape: {overview_rgb.shape}")
            return overview_rgb

    except Exception as e:
        logger.error(f"Error extracting downsampled overview from {ome_tiff_path}: {e}",
                     exc_info=True)  # Add exc_info for full traceback
        return None


def load_image(image_path): # Renamed for clarity, display handled by plot_or_save
    """
    Loads an image using OpenCV and converts to RGB.

    :param image_path: Path to the image file.
    :return: RGB image as a NumPy array, or None if loading fails.
    """
    logger.info(f"Loading image: {image_path}")
    bgr_image = cv2.imread(image_path)
    if bgr_image is None:
        logger.error(f"OpenCV could not load image: {image_path}")
        return None
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    logger.info(f"Successfully loaded image: {os.path.basename(image_path)}, shape: {rgb_image.shape}")
    return rgb_image


def preprocess_for_tissue_detection(rgb_image, current_wsi_output_dir, image_base_name):
    """
    Preprocesses the RGB image to create a binary mask for tissue detection.

    :param rgb_image: The input RGB image to preprocess.
    :param current_wsi_output_dir: Directory to save debug images.
    :param image_base_name: Base name of the original image for naming.
    :return: A binary mask where tissue is white (255) and background is black (0).
    """
    if rgb_image is None: return None
    logger.debug("Converting to grayscale and blurring...")
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, GAUSSIAN_BLUR_KERNEL_SIZE, 0)
    plot_or_save_debug_image(blurred_image, "Blurred Grayscale Image", "01_blurred_gray", current_wsi_output_dir, image_base_name, cmap="gray")

    logger.debug("Applying adaptive thresholding...")
    binary_image = cv2.adaptiveThreshold(
        blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_C
    )
    plot_or_save_debug_image(binary_image, "Adaptive Thresholding (Tissue White)", "02_adaptive_thresh", current_wsi_output_dir, image_base_name, cmap="gray")
    return binary_image

def apply_morphological_operations(binary_image, current_wsi_output_dir, image_base_name):
    """
    Applies morphological operations to the binary image to enhance tissue detection.

    :param binary_image: The binary image from adaptive thresholding.
    :param current_wsi_output_dir: Directory to save debug images.
    :param image_base_name: Base name of the original image for naming.
    :return: A processed binary image after morphological operations.
    """
    if binary_image is None: return None
    logger.info("Applying morphological operations...")

    kernel_close = np.ones(MORPH_CLOSE_KERNEL_SIZE, np.uint8)
    closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel_close, iterations=MORPH_CLOSE_ITERATIONS)
    plot_or_save_debug_image(closed_image, f"After MORPH_CLOSE ({MORPH_CLOSE_ITERATIONS} iter)", "03_morph_close", current_wsi_output_dir, image_base_name, cmap="gray")

    kernel_open = np.ones(MORPH_OPEN_KERNEL_SIZE, np.uint8)
    opened_image = cv2.morphologyEx(closed_image, cv2.MORPH_OPEN, kernel_open, iterations=MORPH_OPEN_ITERATIONS)
    plot_or_save_debug_image(opened_image, f"After MORPH_OPEN ({MORPH_OPEN_ITERATIONS} iter)", "04_morph_open", current_wsi_output_dir, image_base_name, cmap="gray")
    return opened_image


def get_oyster_prompts(binary_mask_opened, original_rgb_image, current_wsi_output_dir, image_base_name, num_oysters=NUM_OYSTERS_TO_DETECT):
    """
    Finds contours in the binary mask and generates bounding boxes for the largest contours.

    :param binary_mask_opened: The binary mask after morphological operations.
    :param original_rgb_image: The original RGB image for reference.
    :param current_wsi_output_dir: Directory to save debug images.
    :param image_base_name: Base name of the original image for naming.
    :param num_oysters: Number of oysters to detect (bounding boxes to generate).
    :return: A list of bounding boxes in the format [[x1, y1, x2, y2], ...] and the selected contours.
    """
    if binary_mask_opened is None or original_rgb_image is None: return [], []
    logger.debug("Finding contours...")
    contours, _ = cv2.findContours(binary_mask_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    logger.info(f"Found {len(contours)} initial contours.")
    if not contours:
        logger.warning("No contours found.")
        return [], []

    min_area = MIN_CONTOUR_AREA_PERCENTAGE * original_rgb_image.shape[0] * original_rgb_image.shape[1]
    logger.debug(f"Minimum contour area threshold: {min_area:.2f} pixels")
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    logger.info(f"Found {len(valid_contours)} contours above area threshold.")

    if not valid_contours:
        logger.warning("No valid contours found after area filtering.")
        return [], []
    if len(valid_contours) < num_oysters:
        logger.warning(f"Expected {num_oysters} oysters, found {len(valid_contours)}. Using all found.")
        selected_contours_list = sorted(valid_contours, key=cv2.contourArea, reverse=True)
    else:
        selected_contours_list = sorted(valid_contours, key=cv2.contourArea, reverse=True)[:num_oysters]
    logger.info(f"Selected {len(selected_contours_list)} contours as potential oysters.")

    bounding_boxes = []
    # Visualization of selected contours and bounding boxes
    contour_img_viz_all = original_rgb_image.copy()
    cv2.drawContours(contour_img_viz_all, contours, -1, (255,255,0), 1) # All contours
    cv2.drawContours(contour_img_viz_all, selected_contours_list, -1, (0,255,0), 3) # Selected
    plot_or_save_debug_image(contour_img_viz_all, "All (Yellow) & Selected (Green) Contours", "05_all_selected_contours", current_wsi_output_dir, image_base_name)

    contour_img_viz_boxes = original_rgb_image.copy()
    for i, cnt in enumerate(selected_contours_list):
        x, y, w, h = cv2.boundingRect(cnt)
        bounding_boxes.append([x, y, x + w, y + h])
        cv2.rectangle(contour_img_viz_boxes, (x, y), (x + w, y + h), (255, 0, 0), 5)
        cv2.putText(contour_img_viz_boxes, f"Oyster {i+1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0),2)
    plot_or_save_debug_image(contour_img_viz_boxes, f"Selected Contours with BBoxes", "06_bboxes", current_wsi_output_dir, image_base_name)

    logger.info(f"Generated {len(bounding_boxes)} Bounding Boxes.")
    return bounding_boxes, selected_contours_list


def create_and_save_contour_masks(
    selected_contours, original_rgb_image, current_output_dir, original_image_filename_base
):
    """
    Creates filled masks from contours and saves them.

    Args:
        selected_contours (list): List of contours to create masks from.
        original_rgb_image (np.ndarray): The original RGB image for reference.
        current_output_dir (str): Directory to save the generated masks.
        original_image_filename_base (str): Base filename of the original image for naming masks.
    """
    if not selected_contours:
        logger.info("No selected contours to create masks from.")
        return []  # Return empty list if no masks created

    contour_masks_dir = os.path.join(current_output_dir, "contour_based_masks")  # Subdirectory
    os.makedirs(contour_masks_dir, exist_ok=True)
    logger.info(
        f"\tSaving {len(selected_contours)} contour-based masks to {contour_masks_dir}..."
    )
    generated_masks = []

    # Combined plot for contour masks
    if ENABLE_PLOTTING and len(selected_contours) > 0:
        fig_contour, axes_contour = plt.subplots(
            1,
            len(selected_contours),
            figsize=(7 * len(selected_contours), 7),
            squeeze=False
        )

    for i, contour in enumerate(selected_contours):
        # Create a blank image (all zeros) with the same dimensions as the original
        mask_binary = np.zeros(
            original_rgb_image.shape[:2], dtype=np.uint8
        )  # Use only H, W for grayscale mask

        # Draw the contour filled in white (255) on the blank mask
        cv2.drawContours(mask_binary, [contour], -1, (255), thickness=cv2.FILLED)
        generated_masks.append(
            mask_binary.astype(bool)
        )  # Store as boolean mask for consistency

        mask_filename = os.path.join(
            contour_masks_dir, f"{original_image_filename_base}_oyster_{i + 1}_contour_mask.png"
        )
        try:
            cv2.imwrite(mask_filename, mask_binary)
            logger.debug(f"Saved: {mask_filename}")
        except Exception as e:
            logger.error(f"🛑 Error saving contour mask {mask_filename}: {e}")

        # Visualization similar to SAM masks
        if ENABLE_PLOTTING:
            ax = axes_contour[0, i]
            ax.imshow(original_rgb_image)  # Show the original RGB image

            # Create an RGBA overlay for the contour mask
            h, w = mask_binary.shape[0], mask_binary.shape[1]
            mask_overlay_rgba = np.zeros((h, w, 4), dtype=np.uint8)
            color = [0, 255, 0] if i % 2 == 0 else [0, 0, 255]
            mask_overlay_rgba[mask_binary.astype(bool), 0:3] = color
            mask_overlay_rgba[mask_binary.astype(bool), 3] = 150
            ax.imshow(mask_overlay_rgba)
            x_r, y_r, w_r, h_r = cv2.boundingRect(contour)
            rect = plt.Rectangle((x_r, y_r), w_r, h_r, fill=False, edgecolor="magenta", linewidth=1, linestyle="--")
            ax.add_patch(rect)
            ax.set_title(f"Oyster {i + 1} - Contour Mask")
            ax.axis('off')

    if ENABLE_PLOTTING and len(selected_contours) > 0:
        fig_contour.tight_layout()
        plt.show()

    if not ENABLE_PLOTTING:
        fig_debug_contour, axes_debug_contour = plt.subplots(
            1,
            len(generated_masks),
            figsize=(7 * len(generated_masks), 7),
            squeeze=False
        )

        # Apply resizing to original_rgb_image for debug plot background
        current_height, current_width = original_rgb_image.shape[0], original_rgb_image.shape[1]
        if max(current_height, current_width) > DEBUG_VISUALIZATION_MAX_DIM:
            if current_width > current_height:
                new_width = DEBUG_VISUALIZATION_MAX_DIM
                new_height = int(current_height * (new_width / current_width))
            else:
                new_height = DEBUG_VISUALIZATION_MAX_DIM
                new_width = int(current_width * (new_height / current_height))
            new_width = max(1, new_width)
            new_height = max(1, new_height)
            display_rgb_image = cv2.resize(original_rgb_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            # Need to also scale masks/bboxes if displaying on resized image
            scale_factor_x = new_width / current_width
            scale_factor_y = new_height / current_height
        else:
            display_rgb_image = original_rgb_image
            scale_factor_x, scale_factor_y = 1.0, 1.0

        for i, mask in enumerate(generated_masks):
            ax = axes_debug_contour[0, i]
            ax.imshow(display_rgb_image)  # Use the resized image here
            if mask.any():
                # Resize the mask too for overlay
                resized_mask = cv2.resize(mask.astype(np.uint8),
                                          (display_rgb_image.shape[1], display_rgb_image.shape[0]),
                                          interpolation=cv2.INTER_NEAREST).astype(bool)

                h_m, w_m = resized_mask.shape  # Use resized mask shape for overlay
                mask_overlay_rgba = np.zeros((h_m, w_m, 4), dtype=np.uint8)
                color = [0, 255, 0] if i % 2 == 0 else [0, 0, 255]
                mask_overlay_rgba[resized_mask, 0:3] = color
                mask_overlay_rgba[resized_mask, 3] = 150
                ax.imshow(mask_overlay_rgba)

                # Scale bounding box for display
                x_r, y_r, w_r, h_r = cv2.boundingRect(selected_contours[i])
                scaled_x = x_r * scale_factor_x
                scaled_y = y_r * scale_factor_y
                scaled_w = w_r * scale_factor_x
                scaled_h = h_r * scale_factor_y
                rect = plt.Rectangle((scaled_x, scaled_y), scaled_w, scaled_h, fill=False, edgecolor="magenta",
                                     linewidth=1, linestyle="--")
                ax.add_patch(rect)
            ax.set_title(f"Oyster {i + 1} - Contour Mask")
            ax.axis('off')
        fig_debug_contour.tight_layout()
        debug_img_dir = os.path.join(current_output_dir, "debug_images")
        os.makedirs(debug_img_dir, exist_ok=True)
        save_path = os.path.join(debug_img_dir, f"{original_image_filename_base}_07_contour_masks_overlay.png")
        fig_debug_contour.savefig(save_path)
        plt.close(fig_debug_contour)  # Important: Close to free memory
    logger.info("Contour-based mask saving complete.")
    return generated_masks


def initialize_sam_predictor(checkpoint_path, model_type):
    """Initializes and returns the SAM predictor.

    Args:
        checkpoint_path (str): Path to the SAM model checkpoint.
        model_type (str): Type of SAM model to use (e.g., "vit_h", "vit_l", "vit_b").
    """
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info(f"Using SAM device: {device}")
    try:
        logger.info("Loading SAM model...")
        sam_model = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam_model.to(device=device)
        predictor = SamPredictor(sam_model)
        logger.info("SAM model loaded successfully.")
        return predictor
    except FileNotFoundError:
        logger.error(
            f"🛑 SAM Checkpoint not found at {checkpoint_path}. Please download or check path."
        )
    except Exception as e:
        logger.error(f"🛑 Error loading SAM model: {e}")
    return None


def predict_and_visualize_masks(predictor, rgb_image, bounding_boxes, current_wsi_output_dir, image_base_name):
    """
    Uses SAM to predict masks for given bounding boxes and visualizes them.

    :param predictor: Initialized SAM predictor instance.
    :param rgb_image: The RGB image to process.
    :param bounding_boxes: List of bounding boxes in the format [[x1, y1, x2, y2], ...].
    :param current_wsi_output_dir: Directory to save debug images.
    :param image_base_name: Base name of the original image for naming.
    :return: A list of SAM-generated masks and their corresponding scores.
    """
    if not predictor or not bounding_boxes or rgb_image is None:
        logger.warning("Skipping SAM prediction due to missing predictor, boxes, or image.")
        return [], []

    img_for_sam = rgb_image.astype(np.uint8)  # Ensure uint8
    if rgb_image.max() <= 1.0 and rgb_image.min() >= 0.0:  # float 0-1 needs scaling
        img_for_sam = (rgb_image * 255).astype(np.uint8)

    try:
        predictor.set_image(img_for_sam)
        logger.debug(
            f"Image set in SAM predictor. Shape: {img_for_sam.shape}, dtype: {img_for_sam.dtype}"
        )
    except Exception as e:
        logger.error(f"🛑 Error setting image in SAM predictor: {e}")
        return [], []

    sam_masks_list = []
    sam_scores_list = []
    logger.info(f"Processing {len(bounding_boxes)} bounding boxes with SAM one by one...")

    # Combined plot for SAM masks
    if ENABLE_PLOTTING and len(bounding_boxes) > 0:
        fig_sam, axes_sam = plt.subplots(1, len(bounding_boxes), figsize=(7 * len(bounding_boxes), 7), squeeze=False)

    for i, box_coords in enumerate(bounding_boxes):
        input_box_np = np.array([box_coords])       # Process one box at a time, shape (1,4)
        logger.debug(f"Input box for SAM (Oyster Candidate {i + 1}): {input_box_np}")
        try:
            masks, scores, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box_np,
                multimask_output=False,
            )
            sam_masks_list.append(masks[0])  # masks is (1, H, W), so take masks[0]
            sam_scores_list.append(scores[0])  # scores is (1,), so take scores[0]
            logger.debug(
                f"SAM generated mask for Oyster Candidate {i + 1}. Score: {scores[0]:.3f}"
            )
        except Exception as e:
            logger.error(f"Error predicting SAM for Oyster {i + 1} with box {box_coords}: {e}")
            sam_masks_list.append(np.zeros((rgb_image.shape[0], rgb_image.shape[1]), dtype=bool))
            sam_scores_list.append(0.0)

        if ENABLE_PLOTTING:
            ax = axes_sam[0, i]
            ax.imshow(rgb_image)
            mask, score = sam_masks_list[-1], sam_scores_list[-1]  # Get last added
            if score > 0 or mask.any():  # Plot if not dummy error mask
                h_m, w_m = mask.shape
                mask_overlay_rgba = np.zeros((h_m, w_m, 4), dtype=np.uint8)
                color = [0, 255, 0] if i % 2 == 0 else [0, 0, 255]
                mask_overlay_rgba[mask, 0:3] = color
                mask_overlay_rgba[mask, 3] = 150
                ax.imshow(mask_overlay_rgba)
                box_for_plot = bounding_boxes[i]
                rect = plt.Rectangle((box_for_plot[0], box_for_plot[1]), box_for_plot[2] - box_for_plot[0],
                                     box_for_plot[3] - box_for_plot[1], fill=False, edgecolor='red', linewidth=2)
                ax.add_patch(rect)
            ax.set_title(f"Oyster {i + 1} - SAM (Score: {score:.3f})")
            ax.axis('off')

    if ENABLE_PLOTTING and len(bounding_boxes) > 0:
        fig_sam.tight_layout()
        plt.show()

    # Save debug plot of SAM masks if not plotting live
    if not ENABLE_PLOTTING and sam_masks_list and any(s > 0 for s in sam_scores_list):
        fig_debug_sam, axes_debug_sam = plt.subplots(
            1,
            len(sam_masks_list),
            figsize=(7 * len(sam_masks_list), 7),
            squeeze=False
        )
        # Apply resizing to rgb_image for debug plot background
        current_height, current_width = rgb_image.shape[0], rgb_image.shape[1]
        if max(current_height, current_width) > DEBUG_VISUALIZATION_MAX_DIM:
            if current_width > current_height:
                new_width = DEBUG_VISUALIZATION_MAX_DIM
                new_height = int(current_height * (new_width / current_width))
            else:
                new_height = DEBUG_VISUALIZATION_MAX_DIM
                new_width = int(current_width * (new_height / current_height))
            new_width = max(1, new_width)
            new_height = max(1, new_height)
            display_rgb_image = cv2.resize(rgb_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            scale_factor_x = new_width / current_width
            scale_factor_y = new_height / current_height
        else:
            display_rgb_image = rgb_image
            scale_factor_x, scale_factor_y = 1.0, 1.0

        for i, (mask, score) in enumerate(zip(sam_masks_list, sam_scores_list)):
            ax = axes_debug_sam[0, i]
            ax.imshow(display_rgb_image)  # Use the resized image here
            if score > 0 or mask.any():
                # Resize the mask too for overlay
                resized_mask = cv2.resize(mask.astype(np.uint8),
                                          (display_rgb_image.shape[1], display_rgb_image.shape[0]),
                                          interpolation=cv2.INTER_NEAREST).astype(bool)

                h_m, w_m = resized_mask.shape  # Use resized mask shape for overlay
                mask_overlay_rgba = np.zeros((h_m, w_m, 4), dtype=np.uint8)
                color = [0, 255, 0] if i % 2 == 0 else [0, 0, 255]
                mask_overlay_rgba[resized_mask, 0:3] = color
                mask_overlay_rgba[resized_mask, 3] = 150
                ax.imshow(mask_overlay_rgba)
                box_for_plot = bounding_boxes[i]
                # Scale bounding box for display
                scaled_x1 = box_for_plot[0] * scale_factor_x
                scaled_y1 = box_for_plot[1] * scale_factor_y
                scaled_x2 = box_for_plot[2] * scale_factor_x
                scaled_y2 = box_for_plot[3] * scale_factor_y
                rect = plt.Rectangle(
                    (scaled_x1, scaled_y1),
                    scaled_x2 - scaled_x1,
                    scaled_y2 - scaled_y1,
                    fill=False,
                    edgecolor='red',
                    linewidth=2
                )
                ax.add_patch(rect)
            ax.set_title(f"Oyster {i + 1} - SAM (Score: {score:.3f})")
            ax.axis('off')

        fig_debug_sam.tight_layout()
        debug_img_dir = os.path.join(current_wsi_output_dir, "debug_images")
        os.makedirs(debug_img_dir, exist_ok=True)
        save_path = os.path.join(debug_img_dir, f"{image_base_name}_08_sam_masks_overlay.png")
        fig_debug_sam.savefig(save_path)
        plt.close(fig_debug_sam)  # Close figure to free memory
        logger.debug(f"Saved SAM masks overlay debug image to {save_path}")

    return sam_masks_list, sam_scores_list


def save_sam_masks(masks, scores, current_output_dir, original_image_filename_base):
    """
    Saves the SAM-generated boolean masks as PNG files.

    :param masks: List of boolean masks generated by SAM.
    :param scores: List of scores corresponding to each mask.
    :param current_output_dir: Directory to save the SAM masks.
    :param original_image_filename_base: Base filename of the original image for naming masks.
    :return: None
    """
    if not masks:
        logger.info("No SAM masks to save.")
        return

    sam_masks_dir = os.path.join(current_output_dir, "sam_generated_masks")  # Subdirectory
    os.makedirs(sam_masks_dir, exist_ok=True)  # Create if it doesn't exist
    logger.info(f"\nSaving {len(masks)} SAM masks to {sam_masks_dir}...")

    for i, (mask_bool, score) in enumerate(zip(masks, scores)):
        if score == 0.0 and not mask_bool.any():
            logger.debug(
                f"Skipping save for SAM mask (Oyster Candidate {i + 1}) due to error or zero score."
            )
            continue
        mask_to_save = mask_bool.astype(np.uint8) * 255
        # Use a consistent naming, Oyster 1 from SAM corresponds to first contour, etc.
        mask_filename = os.path.join(
            sam_masks_dir, f"{original_image_filename_base}_oyster_{i + 1}_sam_mask_score_{score:.3f}.png"
        )
        try:
            cv2.imwrite(mask_filename, mask_to_save)
            logger.debug(f"saved: {mask_filename}")
        except Exception as e:
            logger.error(f"🛑 Error saving sam mask {mask_filename}: {e}")
    logger.info("sam mask saving complete.")

def process_single_wsi(image_file_path, sam_predictor_instance):
    """
    Processes a single WSI image file through the Stage 1 pipeline.

    :param image_file_path: Path to the WSI image file.
    :param sam_predictor_instance: Initialized SAM predictor instance.
    """
    logger.info(f"\t---Processing Image: {image_file_path}... ---")

    # Create unique output directory for this image
    image_base_name = os.path.splitext(os.path.basename(image_file_path))[0]
    current_wsi_output_dir = os.path.join(OUTPUT_MASK_PARENT_DIR, image_base_name)
    os.makedirs(current_wsi_output_dir, exist_ok=True)
    logger.info(f"Output directory for this image: {current_wsi_output_dir}")

    # 1. Load Image
    low_res_rgb_image = extract_downsampled_overview_from_ome_tiff(
        image_file_path,
        STAGE1_PROCESSING_DOWNSAMPLE,
    )

    if low_res_rgb_image is None:
        logger.warning(f"🛑 Failed to extract downsampled overview for {image_file_path}. Skipping this image.")
        return
    plot_or_save_debug_image(
        low_res_rgb_image,
        f"Downsampled Overview for Processing ({image_base_name})",
        "00_downsampled_overview",
        current_wsi_output_dir,
        image_base_name
    )

    # 2. Preprocess for tissue detection
    binary_tissue_image = preprocess_for_tissue_detection(low_res_rgb_image, current_wsi_output_dir, image_base_name)
    if binary_tissue_image is None:
        logger.warning(f"🛑 Preprocessing failed for {image_file_path}. Skipping this image.")
        return

    # 3. Apply Morphological Operations
    processed_binary_mask = apply_morphological_operations(binary_tissue_image, current_wsi_output_dir, image_base_name)
    if processed_binary_mask is None:
        logger.warning(f"🛑 Morphological operations failed for {image_file_path}. Skipping this image.")
        return

    # 4. Get Bounding Box Prompts
    oyster_bboxes, selected_oyster_contours = get_oyster_prompts(
        processed_binary_mask,
        low_res_rgb_image,
        current_wsi_output_dir,
        image_base_name,
    )
    if not oyster_bboxes:
        logger.warning(f"🛑 No bounding boxes generated for {image_file_path}. Skipping SAM prediction.")
        return

    # 4.5 Create and Save Masks directly from Contours
    contour_generated_masks = create_and_save_contour_masks(
        selected_oyster_contours,
        low_res_rgb_image,
        current_wsi_output_dir,
        image_base_name,
    )
    if not contour_generated_masks:
        logger.warning(f"🛑 No contour masks created for {image_file_path}.")
    else:
        logger.info(f"Contour masks created for {image_file_path} and saved in {current_wsi_output_dir}/contour_based_masks")

    # 5-7. Predict and Visualize Masks using SAM
    final_sam_masks, final_sam_scores = predict_and_visualize_masks(
        sam_predictor_instance, low_res_rgb_image, oyster_bboxes, current_wsi_output_dir, image_base_name
    )
    save_sam_masks(
        final_sam_masks,
        final_sam_scores,
        current_wsi_output_dir,
        image_base_name
    )

    logger.info("--- Processing complete for this image. ---\n")

# --_ Main Execution ---
if __name__ == "__main__":
    # Setup logging first
    setup_logging(OUTPUT_MASK_PARENT_DIR)  # Log will be in the parent output dir

    logger.info("\t\t\t--- Stage 1: Oyster Instance Segmentation ---")

    # Initialize SAM Predictor ONCE for the batch
    shared_sam_predictor = initialize_sam_predictor(
        SAM_CHECKPOINT_PATH, SAM_MODEL_TYPE
    )

    if not shared_sam_predictor:
        logger.error("🛑 SAM predictor initialization failed. Cannot process images.")
        exit(1)

    # Find all image files in the input directory
    image_file_paths = []
    for ext in VALID_IMAGE_EXTENSIONS:
        image_file_paths.extend(
            [
                os.path.join(INPUT_WSI_DIR, f)
                for f in os.listdir(INPUT_WSI_DIR)
                if f.lower().endswith(ext)
            ]
        )

    if not image_file_paths:
        logger.error(f"🛑 No valid images found in {INPUT_WSI_DIR} with extensions {VALID_IMAGE_EXTENSIONS}. Exiting.")
        exit(1)

    # Iterate through each image file and process it
    for image_file_path in image_file_paths:
        process_single_wsi(image_file_path, shared_sam_predictor)

    logger.info("--- Fin ---")
