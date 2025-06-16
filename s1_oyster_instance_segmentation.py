# s1_oyster_instance_segmentation.py
#
# Description:
# This script performs Stage 1 of the oyster analysis pipeline:
    # 1. Loads a downsampled Whole Slide Image (WSI) of oyster tissue.
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
import os
import cv2
import matplotlib.pyplot as plt
import torch
from segment_anything import sam_model_registry, SamPredictor

# --- SCRIPT CONFIGURATION ---
INPUT_WSI_PATH = "data/oyster10_ds.png" # Path to the downsampled WSI
SAM_CHECKPOINT_PATH = "pretrained_checkpoint/sam_vit_h_4b8939.pth" # Path to SAM checkpoint
SAM_MODEL_TYPE = "vit_h" # SAM model type (e.g., "vit_h", "vit_l", "vit_b")
OUTPUT_MASK_DIR = "oyster_instance_masks_sam" # Directory to save SAM-generated masks

# Morphological operation parameters
GAUSSIAN_BLUR_KERNEL_SIZE = (3, 3)
ADAPTIVE_THRESH_BLOCK_SIZE = 11 # Must be odd
ADAPTIVE_THRESH_C = 2
MORPH_CLOSE_KERNEL_SIZE = (5, 5)
MORPH_CLOSE_ITERATIONS = 3
MORPH_OPEN_KERNEL_SIZE = (3, 3)
MORPH_OPEN_ITERATIONS = 2
MIN_CONTOUR_AREA_PERCENTAGE = 0.01

# Control plotting for debugging/visualization (set to False to disable plots)
ENABLE_PLOTTING = True
# --- END SCRIPT CONFIGURATION ---

def load_and_display_image(image_path):
    """Loads an image using OpenCV, converts to RGB, and optionally displays it."""
    bgr_image = cv2.imread(image_path)
    if bgr_image is None:
        print(f"🛑 CRITICAL: OpenCV could not load image: {image_path}")
        return None
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    print(f"Successfully loaded image: {image_path}, shape: {rgb_image.shape}")
    if ENABLE_PLOTTING:
        plt.figure(figsize=(10, 10))
        plt.imshow(rgb_image)
        plt.title(f"Loaded Image: {os.path.basename(image_path)}")
        plt.axis('off')
        plt.show()
    return rgb_image

def preprocess_for_tissue_detection(rgb_image):
    """Converts image to grayscale, blurs, and applies adaptive thresholding."""
    if rgb_image is None: return None
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, GAUSSIAN_BLUR_KERNEL_SIZE, 0)

    # Adaptive thresholding (generates white tissue on black background due to THRESH_BINARY_INV)
    binary_image = cv2.adaptiveThreshold(
        blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_C
    )
    if ENABLE_PLOTTING:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(blurred_image, cmap='gray')
        plt.title("Blurred Grayscale Image")
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(binary_image, cmap='gray')
        plt.title("Adaptive Thresholding (Tissue White)")
        plt.axis('off')
        plt.show()
    return binary_image

def apply_morphological_operations(binary_image):
    """Applies morphological closing and opening to refine the binary mask."""
    if binary_image is None: return None
    print("Applying morphological operations...")

    kernel_close = np.ones(MORPH_CLOSE_KERNEL_SIZE, np.uint8)
    closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel_close, iterations=MORPH_CLOSE_ITERATIONS)

    kernel_open = np.ones(MORPH_OPEN_KERNEL_SIZE, np.uint8)
    opened_image = cv2.morphologyEx(closed_image, cv2.MORPH_OPEN, kernel_open, iterations=MORPH_OPEN_ITERATIONS)

    if ENABLE_PLOTTING:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(closed_image, cmap='gray')
        plt.title(f"After MORPH_CLOSE ({MORPH_CLOSE_ITERATIONS} iter)")
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(opened_image, cmap='gray')
        plt.title(f"After MORPH_OPEN ({MORPH_OPEN_ITERATIONS} iter)")
        plt.axis('off')
        plt.show()
    return opened_image

def get_oyster_prompts(binary_mask_opened, original_rgb_image, num_oysters=2):
    """
    Finds contours, filters them, and returns:
    1. Bounding boxes for the largest contours (for SAM).
    2. The selected contours themselves (for direct mask creation).
    """
    if binary_mask_opened is None or original_rgb_image is None: return []

    contours, _ = cv2.findContours(binary_mask_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Found {len(contours)} initial contours.")

    if not contours:
        print("🛑 No contours found.")
        return [], []

    min_area = MIN_CONTOUR_AREA_PERCENTAGE * original_rgb_image.shape[0] * original_rgb_image.shape[1]
    print(f"Minimum contour area threshold: {min_area:.2f} pixels")

    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    print(f"Found {len(valid_contours)} contours above area threshold.")

    if len(valid_contours) < num_oysters:
        print(f"⚠️ Expected {num_oysters} oysters, but found only {len(valid_contours)} valid contours. Adjust parameters or check image.")
        # Optionally, take all valid contours if fewer than num_oysters
        selected_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)
    elif not valid_contours:  # No valid contours at all
        print(f"🛑 No valid contours found after area filtering.")
        return [], []
    else:
        selected_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)[:num_oysters]

    print(f"Selected {len(selected_contours)} contours as potential oysters.")

    bounding_boxes = []
    if ENABLE_PLOTTING and selected_contours:
        contour_img_viz = original_rgb_image.copy()
        # Draw ALL initially found contours in a light color (e.g., yellow) for context
        cv2.drawContours(contour_img_viz, contours, -1, (255, 255, 0), 1)  # Thin yellow
        # Draw selected contours more prominently (e.g., green)
        cv2.drawContours(contour_img_viz, selected_contours, -1, (0, 255, 0), 3)  # Thicker green

        plt.figure(figsize=(10,10))
        plt.imshow(contour_img_viz)
        plt.title("All (Yellow) & Selected (Green) Contours")
        plt.axis('off')
        plt.show()

        contour_img_viz_selected = original_rgb_image.copy()
        for i, cnt in enumerate(selected_contours):
            x, y, w, h = cv2.boundingRect(cnt)
            bounding_boxes.append([x, y, x + w, y + h])
            cv2.rectangle(contour_img_viz_selected, (x, y), (x + w, y + h), (255, 0, 0), 5)
            cv2.putText(contour_img_viz_selected, f"Oyster Candidate {i + 1}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
        plt.figure(figsize=(12, 12))
        plt.imshow(contour_img_viz_selected)
        plt.title(f"Selected {len(selected_contours)} Oyster Contours with Bounding Boxes")
        plt.axis('off')
        plt.show()

    # Generate bounding boxes even if plotting is disabled
    if not bounding_boxes and selected_contours:  # If boxes weren't made during plotting
        for cnt in selected_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            bounding_boxes.append([x, y, x + w, y + h])

    print(f"Generated Bounding Boxes: {bounding_boxes}")
    return bounding_boxes, selected_contours

def create_and_save_contour_masks(selected_contours, original_rgb_image, output_dir, original_image_filename="oyster"):
    """Creates filled masks from contours and saves them."""
    if not selected_contours:
        print("No selected contours to create masks from.")
        return [] # Return empty list if no masks created

    contour_masks_dir = os.path.join(output_dir, "contour_based_masks") # Subdirectory
    os.makedirs(contour_masks_dir, exist_ok=True)
    print(f"\nSaving {len(selected_contours)} contour-based masks to {contour_masks_dir}...")
    base_name = os.path.splitext(os.path.basename(original_image_filename))[0]
    generated_masks = []

    # Determine figure size for plotting side-by-side if multiple contours
    num_masks_to_plot = len(selected_contours)
    if ENABLE_PLOTTING and num_masks_to_plot > 0:
        plt.figure(figsize=(7 * num_masks_to_plot, 7))  # Similar to SAM plot

    for i, contour in enumerate(selected_contours):
        # Create a blank image (all zeros) with the same dimensions as the original
        mask = np.zeros(original_rgb_image.shape[:2], dtype=np.uint8) # Use only H, W for grayscale mask

        # Draw the contour filled in white (255) on the blank mask
        cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)
        generated_masks.append(mask.astype(bool)) # Store as boolean mask for consistency

        mask_filename = os.path.join(contour_masks_dir, f"{base_name}_oyster_{i+1}_contour_mask.png")
        try:
            cv2.imwrite(mask_filename, mask)
            print(f"Saved: {mask_filename}")
        except Exception as e:
            print(f"🛑 Error saving contour mask {mask_filename}: {e}")

        # Visualization similar to SAM masks
        if ENABLE_PLOTTING:
            ax = plt.subplot(1, num_masks_to_plot, i + 1)
            ax.imshow(original_rgb_image)  # Show the original RGB image

            # Create an RGBA overlay for the contour mask
            h, w = mask.shape
            mask_overlay_rgba = np.zeros((h, w, 4), dtype=np.uint8)
            # Use the boolean mask for indexing
            contour_mask_bool = mask.astype(bool)

            # Choose a color for the mask (can use same alternating logic as SAM)
            color = [0, 255, 0] if i % 2 == 0 else [0, 0, 255]  # Green for first, Blue for second

            mask_overlay_rgba[contour_mask_bool, 0] = color[0]  # R
            mask_overlay_rgba[contour_mask_bool, 1] = color[1]  # G
            mask_overlay_rgba[contour_mask_bool, 2] = color[2]  # B
            mask_overlay_rgba[contour_mask_bool, 3] = 150  # Alpha (transparency)

            ax.imshow(mask_overlay_rgba)

            # Optionally, draw the bounding box of this contour for direct comparison to SAM's input
            x, y, w_rect, h_rect = cv2.boundingRect(contour)  # Get bounding rect of the contour itself
            rect = plt.Rectangle((x, y), w_rect, h_rect,
                                 fill=False, edgecolor='magenta', linewidth=1,
                                 linestyle='--')  # Different color/style
            ax.add_patch(rect)

            ax.set_title(f"Oyster {i + 1} - Contour-based Mask")
            ax.axis('off')

    if ENABLE_PLOTTING and num_masks_to_plot > 0:
        plt.tight_layout()
        plt.show()

    print("Contour-based mask creation, visualization, and saving complete.")
    return generated_masks

def initialize_sam_predictor(checkpoint_path, model_type):
    """Initializes and returns the SAM predictor."""
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using SAM device: {device}")
    try:
        print("Loading SAM model...")
        sam_model = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam_model.to(device=device)
        predictor = SamPredictor(sam_model)
        print("SAM model loaded successfully.")
        return predictor
    except FileNotFoundError:
        print(f"🛑 SAM Checkpoint not found at {checkpoint_path}. Please download or check path.")
    except Exception as e:
        print(f"🛑 Error loading SAM model: {e}")
    return None

def predict_and_visualize_masks(predictor, rgb_image, bounding_boxes):
    """Uses SAM to predict masks for given bounding boxes and visualizes them."""
    if not predictor or not bounding_boxes or rgb_image is None:
        print("Skipping SAM prediction due to missing predictor, boxes, or image.")
        return [], []

    # SAM expects the image in HWC uint8 format
    if rgb_image.dtype != np.uint8:
        print(f"Warning: Image dtype is {rgb_image.dtype}, converting to uint8 for SAM.")
        if rgb_image.max() <= 1.0 and rgb_image.min() >= 0.0: # float 0-1
            img_for_sam = (rgb_image * 255).astype(np.uint8)
        else: # General case
            img_for_sam = np.clip(rgb_image, 0, 255).astype(np.uint8)
    else:
        img_for_sam = rgb_image

    try:
        predictor.set_image(img_for_sam)
        print(f"Image set in SAM predictor. Shape: {img_for_sam.shape}, dtype: {img_for_sam.dtype}")
    except Exception as e:
        print(f"🛑 Error setting image in SAM predictor: {e}")
        return [], []

    sam_masks = []
    sam_scores = []
    print(f"Processing {len(bounding_boxes)} bounding boxes with SAM one by one...")

    for i, box_coords in enumerate(bounding_boxes):
        input_box_np = np.array([box_coords]) # Process one box at a time, shape (1,4)
        print(f"Input box for SAM (Oyster Candidate {i+1}): {input_box_np}")
        try:
            masks, scores, _ = predictor.predict(
                point_coords=None, point_labels=None,
                box=input_box_np, multimask_output=False
            )
            sam_masks.append(masks[0]) # masks is (1, H, W), so take masks[0]
            sam_scores.append(scores[0]) # scores is (1,), so take scores[0]
            print(f"SAM generated mask for Oyster Candidate {i+1}. Score: {scores[0]:.3f}")
        except Exception as e:
            print(f"🛑 Error predicting for Oyster Candidate {i+1} with box {box_coords}: {e}")
            sam_masks.append(np.zeros((rgb_image.shape[0], rgb_image.shape[1]), dtype=bool)) # Dummy
            sam_scores.append(0.0) # Dummy

    if ENABLE_PLOTTING and sam_masks and any(s > 0 for s in sam_scores):
        num_masks_to_plot = len(sam_masks)
        plt.figure(figsize=(7 * num_masks_to_plot, 7))
        for i, (mask, score) in enumerate(zip(sam_masks, sam_scores)):
            if score == 0.0 and not mask.any(): continue # Skip dummy error masks

            ax = plt.subplot(1, num_masks_to_plot, i + 1)
            ax.imshow(rgb_image)
            h, w = mask.shape
            mask_overlay_rgba = np.zeros((h, w, 4), dtype=np.uint8)
            color = [0, 255, 0] if i % 2 == 0 else [0, 0, 255] # Alternating colors
            mask_overlay_rgba[mask, 0:3] = color
            mask_overlay_rgba[mask, 3] = 150 # Alpha
            ax.imshow(mask_overlay_rgba)
            box_for_plot = bounding_boxes[i]
            rect = plt.Rectangle((box_for_plot[0], box_for_plot[1]), box_for_plot[2]-box_for_plot[0], box_for_plot[3]-box_for_plot[1],
                                 fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
            ax.set_title(f"Oyster {i+1} - SAM Mask (Score: {score:.3f})")
            ax.axis('off')
        plt.tight_layout()
        plt.show()
    return sam_masks, sam_scores

def save_sam_masks(masks, scores, output_dir, original_image_filename="oyster"):
    """Saves the SAM-generated boolean masks as PNG files."""
    if not masks:
        print("No SAM masks to save.")
        return

    sam_masks_dir = os.path.join(output_dir, "sam_generated_masks") # Subdirectory
    os.makedirs(sam_masks_dir, exist_ok=True) # Create if it doesn't exist
    print(f"\nSaving {len(masks)} SAM masks to {sam_masks_dir}...")
    base_name = os.path.splitext(os.path.basename(original_image_filename))[0]

    for i, (mask_bool, score) in enumerate(zip(masks, scores)):
        if score == 0.0 and not mask_bool.any():
            print(f"Skipping save for SAM mask (Oyster Candidate {i+1}) due to error or zero score.")
            continue
        mask_to_save = mask_bool.astype(np.uint8) * 255
        # Use a consistent naming, Oyster 1 from SAM corresponds to first contour, etc.
        mask_filename = os.path.join(sam_masks_dir, f"{base_name}_oyster_{i+1}_sam_mask_score_{score:.3f}.png")
        try:
            cv2.imwrite(mask_filename, mask_to_save)
            print(f"Saved: {mask_filename}")
        except Exception as e:
            print(f"🛑 Error saving SAM mask {mask_filename}: {e}")
    print("SAM mask saving complete.")

if __name__ == "__main__":
    print("--- Stage 1: Oyster Instance Segmentation ---")

    # 1. Load Image
    low_res_rgb_image = load_and_display_image(INPUT_WSI_PATH)

    if low_res_rgb_image is not None:
        # 2. Preprocess for tissue detection (Grayscale, Blur, Adaptive Threshold)
        binary_tissue_image = preprocess_for_tissue_detection(low_res_rgb_image)

        # 3. Apply Morphological Operations
        processed_binary_mask = apply_morphological_operations(binary_tissue_image)

        # 4. Get Bounding Box Prompts
        oyster_bboxes, selected_oyster_contours = get_oyster_prompts(processed_binary_mask, low_res_rgb_image)

        # 4.5 Create and Save Masks directly from Contours
        contour_generated_masks = []
        if selected_oyster_contours:
            contour_generated_masks = create_and_save_contour_masks(
                selected_oyster_contours,
                low_res_rgb_image,
                OUTPUT_MASK_DIR,
                INPUT_WSI_PATH
            )
        else:
            print("No selected contours to create direct masks from.")

        if oyster_bboxes:
            # 5. Initialize SAM
            sam_predictor = initialize_sam_predictor(SAM_CHECKPOINT_PATH, SAM_MODEL_TYPE)

            if sam_predictor:
                # 6. Predict and Visualize Masks using SAM
                final_oyster_masks, final_oyster_scores = predict_and_visualize_masks(
                    sam_predictor, low_res_rgb_image, oyster_bboxes
                )

                # 7. Save SAM Masks
                save_sam_masks(final_oyster_masks, final_oyster_scores, OUTPUT_MASK_DIR, INPUT_WSI_PATH)
            else:
                print("🛑 SAM predictor initialization failed. Cannot proceed with mask prediction.")
        else:
            print("🛑 No bounding boxes generated. Cannot proceed with SAM prediction.")
    else:
        print("🛑 Image loading failed. Exiting.")

    print("--- Fin ---")
