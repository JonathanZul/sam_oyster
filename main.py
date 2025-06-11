import openslide
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

wsi_path = "data/oyster_slide_downsampled_32x.png" #  <--- SET YOUR WSI FILE PATH HERE

low_res_image_bgr = cv2.imread(wsi_path)

# 1.1
if low_res_image_bgr is None:
    print(f"🛑 CRITICAL: OpenCV could not load the exported image: {wsi_path}")
    print("Please ensure the path is correct and the file exists.")
    low_res_image_rgb = None
else:
    low_res_image_rgb = cv2.cvtColor(low_res_image_bgr, cv2.COLOR_BGR2RGB)
    print(f"Successfully loaded exported image, shape: {low_res_image_rgb.shape}")

    plt.figure(figsize=(10,10))
    plt.imshow(low_res_image_rgb)
    plt.title("Low-Resolution WSI View (Exported from QuPath)")
    plt.axis('off')
    plt.show()

# 1.2
if low_res_image_rgb is not None:
    gray_image = cv2.cvtColor(low_res_image_rgb, cv2.COLOR_RGB2GRAY)

    # Optional: Apply Gaussian Blur to smooth and reduce noise
    # Adjust kernel size (e.g., (5,5) or (7,7)) as needed. It must be odd.
    blurred_image = cv2.GaussianBlur(gray_image, (3, 3), 0)

    plt.figure(figsize=(10,10))
    plt.imshow(blurred_image, cmap='gray')
    plt.title("Blurred Grayscale Image")
    plt.axis('off')
    plt.show()
else:
    print("Skipping pre-processing as image was not loaded.")

#1.3
if low_res_image_rgb is not None:
    # # Apply Otsu's thresholding (not the best for this type of image, but included for reference)
    # ret, thresh_image_otsu = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply adaptive thresholding
    thresh_image_adaptive = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    plt.figure(figsize=(10, 5))
    plt.imshow(thresh_image_adaptive, cmap='gray')
    plt.title("Adaptive Thresholding")
    plt.axis('off')
    plt.show()

    thresh_image = thresh_image_adaptive

    print("Starting morphological operations...")

    # first display the raw thresholded image (tissue white) to see the baseline
    plt.figure(figsize=(8, 8))
    plt.imshow(thresh_image, cmap='gray')
    plt.title("Initial Thresholded Image (Tissue White)")
    plt.show()

    # Fill holes within each oyster piece without merging them if possible.
    kernel_close = np.ones((5, 5), np.uint8)
    closed_image = cv2.morphologyEx(thresh_image, cv2.MORPH_CLOSE, kernel_close,
                                    iterations=3)

    plt.figure(figsize=(8, 8))
    plt.imshow(closed_image, cmap='gray')
    plt.title("After MORPH_CLOSE (Aim: Fill internal holes)")
    plt.show()

    # Remove small noise particles and try to break very thin connections
    kernel_open = np.ones((3, 3), np.uint8)
    opened_image = cv2.morphologyEx(closed_image, cv2.MORPH_OPEN, kernel_open, iterations=2)

    plt.figure(figsize=(10, 10))
    plt.imshow(opened_image, cmap='gray')
    plt.title("After Morphological Operations (Revised - Tissue White)")
    plt.axis('off')
    plt.show()

    # Find Contours
    contours, hierarchy = cv2.findContours(opened_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print(f"Found {len(contours)} initial contours.")

    # Draw contours for visualization (on a copy of the original image)
    contour_img_viz = low_res_image_rgb.copy()
    cv2.drawContours(contour_img_viz, contours, -1, (0, 255, 0), 3) # Draw all contours in green
    plt.figure(figsize=(10,10))
    plt.imshow(contour_img_viz)
    plt.title("All Detected Contours")
    plt.axis('off')
    plt.show()
else:
    print("Skipping contour generation as image was not loaded.")

# 1.5
oyster_bounding_boxes = []
if low_res_image_rgb is not None and contours:
    # Filter contours by area - we expect 2 large oyster sections
    min_contour_area_threshold = (0.01 * low_res_image_rgb.shape[0] * low_res_image_rgb.shape[1]) # 1% of image area
    print(f"Minimum contour area threshold: {min_contour_area_threshold}")

    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area_threshold]
    print(f"Found {len(valid_contours)} contours above area threshold.")

    # Sort valid contours by area in descending order and take the top 2
    valid_contours.sort(key=cv2.contourArea, reverse=True)
    selected_oyster_contours = valid_contours[:2] # Assuming there are at least 2

    if len(selected_oyster_contours) < 1 :
        print("🛑 Could not find enough large contours for oysters. Adjust threshold or check image.")
    else:
        print(f"Selected {len(selected_oyster_contours)} contours as potential oysters.")

        contour_img_viz_selected = low_res_image_rgb.copy() # For visualization
        for i, cnt in enumerate(selected_oyster_contours):
            x, y, w, h = cv2.boundingRect(cnt)
            # SAM expects boxes as [X_min, Y_min, X_max, Y_max]
            sam_box = [x, y, x + w, y + h]
            oyster_bounding_boxes.append(sam_box)

            # Draw bounding box and label for visualization
            cv2.rectangle(contour_img_viz_selected, (x, y), (x + w, y + h), (255, 0, 0), 5) # red box
            cv2.putText(contour_img_viz_selected, f"Oyster {i+1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,0), 3)

        plt.figure(figsize=(12,12))
        plt.imshow(contour_img_viz_selected)
        plt.title(f"Selected {len(selected_oyster_contours)} Oyster Contours with Bounding Boxes")
        plt.axis('off')
        plt.show()

        print(f"Generated Bounding Boxes for SAM: {oyster_bounding_boxes}")
else:
    print("Skipping prompt generation.")


# 1.6
from segment_anything import sam_model_registry, SamPredictor
import torch # Ensure torch is imported

# --- User-defined variables ---
sam_checkpoint_path = "pretrained_checkpoint/sam_vit_h_4b8939.pth"
model_type = "vit_h" # or "vit_l", "vit_b" depending on your checkpoint
# --- End User-defined variables ---

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Ensure low_res_image_rgb and oyster_bounding_boxes are available from previous steps
if 'low_res_image_rgb' not in locals() or low_res_image_rgb is None:
    print("🛑 low_res_image_rgb is not defined or is None. Please run previous steps.")
    predictor = None
elif 'oyster_bounding_boxes' not in locals() or not oyster_bounding_boxes:
    print("🛑 oyster_bounding_boxes is not defined or is empty. Please run previous steps.")
    predictor = None
else:
    try:
        print("Loading SAM model...")
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint_path)
        sam.to(device=device)
        predictor = SamPredictor(sam)
        print("SAM model loaded successfully.")

        print("Setting image in SAM predictor...")
        # SAM expects the image in HWC uint8 format
        if low_res_image_rgb.dtype != np.uint8:
            print(f"Warning: Image dtype is {low_res_image_rgb.dtype}, converting to uint8.")
            # Basic normalization if it's float (0-1 range)
            if low_res_image_rgb.max() <= 1.0 and low_res_image_rgb.min() >= 0.0:
                 img_for_sam = (low_res_image_rgb * 255).astype(np.uint8)
            else: # General case, clip and convert
                 img_for_sam = np.clip(low_res_image_rgb, 0, 255).astype(np.uint8)
        else:
            img_for_sam = low_res_image_rgb

        predictor.set_image(img_for_sam)
        print(f"Image set in SAM predictor. Image shape: {img_for_sam.shape}, dtype: {img_for_sam.dtype}")

    except FileNotFoundError:
        print(f"🛑 SAM Checkpoint not found at {sam_checkpoint_path}. Please download it or check the path.")
        predictor = None
    except Exception as e:
        print(f"🛑 Error loading SAM model or setting image: {e}")
        predictor = None

# 1.7
oyster_masks_sam = []
oyster_scores_sam = []

if predictor and oyster_bounding_boxes:
    print(f"Processing {len(oyster_bounding_boxes)} bounding boxes one by one...")
    for i, box_coords in enumerate(oyster_bounding_boxes):
        input_box_np = np.array([box_coords]) # SAM expects a batch, so make it (1, 4)
        print(f"Input box for SAM (Oyster {i+1}): {input_box_np}")

        try:
            masks, scores, logits = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box_np,         # Pass a single box, shaped (1, 4)
                multimask_output=False,
            )
            # masks shape will be (1, H, W), scores shape will be (1,)
            oyster_masks_sam.append(masks[0]) # Get the actual 2D mask
            oyster_scores_sam.append(scores[0])# Get the actual score
            print(f"SAM generated mask for Oyster {i+1}. Score: {scores[0]:.3f}")

        except Exception as e:
            print(f"🛑 Error predicting for Oyster {i+1} with box {box_coords}: {e}")
            # Add dummy entries or skip if you want to continue with other boxes
            oyster_masks_sam.append(np.zeros((low_res_image_rgb.shape[0], low_res_image_rgb.shape[1]), dtype=bool)) # Dummy mask
            oyster_scores_sam.append(0.0) # Dummy score


    if oyster_masks_sam and any(s > 0 for s in oyster_scores_sam): # Check if any valid mask was produced
        print(f"Successfully generated {sum(1 for s in oyster_scores_sam if s > 0)} SAM masks out of {len(oyster_bounding_boxes)} attempts.")
        print(f"Shapes of generated masks: {[m.shape for m in oyster_masks_sam]}")
        print(f"Confidence scores from SAM: {oyster_scores_sam}")

        # Visualization (adjust for individual box processing if needed)
        num_masks_to_plot = len(oyster_masks_sam)
        plt.figure(figsize=(7 * num_masks_to_plot, 7))
        for i, (mask, score) in enumerate(zip(oyster_masks_sam, oyster_scores_sam)):
            if score == 0.0 and not mask.any(): # Skip plotting dummy masks from errors
                print(f"Skipping plot for Oyster {i+1} due to previous error.")
                continue

            ax = plt.subplot(1, num_masks_to_plot, i + 1)
            ax.imshow(low_res_image_rgb)

            h, w = mask.shape
            mask_overlay_rgba = np.zeros((h, w, 4), dtype=np.uint8)
            color = [0, 255, 0] if i % 2 == 0 else [0, 0, 255]
            mask_overlay_rgba[mask, 0:3] = color
            mask_overlay_rgba[mask, 3] = 150
            ax.imshow(mask_overlay_rgba)

            # Draw the input bounding box for this specific mask
            box_for_plot = oyster_bounding_boxes[i] # Get the original box coordinates
            rect = plt.Rectangle((box_for_plot[0], box_for_plot[1]), box_for_plot[2]-box_for_plot[0], box_for_plot[3]-box_for_plot[1],
                                 fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)

            ax.set_title(f"Oyster {i+1} - SAM Mask (Score: {score:.3f})")
            ax.axis('off')
        plt.tight_layout()
        plt.show()
    else:
        print("No valid SAM masks were generated.")


elif not predictor:
    print("SAM predictor not initialized. Skipping prediction.")
elif not oyster_bounding_boxes:
    print("No bounding boxes available. Skipping SAM prediction.")

# 1.8
output_mask_dir = "oyster_instance_masks_sam" # Directory to save SAM masks
os.makedirs(output_mask_dir, exist_ok=True)

if oyster_masks_sam: # Check if the list is not empty
    print(f"\nSaving {len(oyster_masks_sam)} SAM masks...")
    for i, mask_bool in enumerate(oyster_masks_sam):
        # mask_bool is already a 2D boolean array (H, W) because multimask_output=False
        # Convert boolean mask (True/False) to uint8 (255/0) for saving as an image.
        mask_to_save = mask_bool.astype(np.uint8) * 255

        # You might want to include the score in the filename or save scores separately
        score = oyster_scores_sam[i]
        mask_filename = os.path.join(output_mask_dir, f"oyster_section_{i+1}_sam_mask_score_{score:.3f}.png")

        try:
            cv2.imwrite(mask_filename, mask_to_save)
            print(f"Saved mask for oyster section {i+1} to {mask_filename}")
        except Exception as e:
            print(f"Error saving mask {mask_filename}: {e}")

    print("Mask saving process complete.")
    # You can now use these masks for downstream tasks, e.g., cropping the original image.
    # Example:
    # masked_oyster_1 = cv2.bitwise_and(low_res_image_rgb, low_res_image_rgb, mask=oyster_masks_sam[0].astype(np.uint8))
    # plt.imshow(masked_oyster_1)
    # plt.title("Oyster 1 with Background Removed by SAM mask")
    # plt.show()

else:
    print("No SAM masks were generated or available to save.")
