import openslide
import numpy as np
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
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    plt.figure(figsize=(10,10))
    plt.imshow(blurred_image, cmap='gray')
    plt.title("Blurred Grayscale Image")
    plt.axis('off')
    plt.show()
else:
    print("Skipping pre-processing as image was not loaded.")

#1.3
if low_res_image_rgb is not None:
    # Apply Otsu's thresholding
    # This automatically finds an optimal threshold value.
    # You might need THRESH_BINARY_INV if your tissue is darker than background.
    # Experiment to get tissue as white (255) and background as black (0).
    ret, thresh_image_otsu = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)    # If Otsu makes tissue black and background white, invert it:
    # if np.mean(thresh_image[thresh_image > 0]) < 128: # Heuristic: if white parts are actually dark
    #    ret, thresh_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # --- START MODIFICATION ---
    # Check the average intensity of the regions Otsu marked as "foreground" (non-zero)
    # If the foreground (tissue) is black (mean close to 0 after Otsu's binary thresholding),
    # it means Otsu correctly separated it, but we want it white.
    # So, if the tissue is black, we invert the image.
    # If the tissue is already white (e.g. if background was darker for some reason), we don't invert.

    # Count pixels: Otsu creates a binary image (0 and 255).
    # If tissue is black (0) and background white (255) from Otsu, then most pixels might be background (255).
    # If tissue is white (255) and background black (0) from Otsu, then most pixels might be background (0).

    # A simpler check: H&E stained tissue is typically darker than the background.
    # So, cv2.THRESH_BINARY + cv2.THRESH_OTSU will likely make the tissue black.
    # We almost always want to invert this for contour finding of the tissue.
    print(f"Mean of Otsu thresholded image: {np.mean(thresh_image_otsu)}")
    # If mean is high, it means most of image is white (background is white, tissue is black).
    # If mean is low, it means most of image is black (background is black, tissue is white).

    # Let's assume tissue is darker and Otsu makes it black. So, we invert.
    thresh_image = cv2.bitwise_not(thresh_image_otsu)
    print("Applied bitwise_not to make tissue white for contour detection.")

    # You can display thresh_image_otsu and thresh_image here to verify
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(thresh_image_otsu, cmap='gray')
    plt.title("Otsu's Direct Output")
    plt.subplot(1, 2, 2)
    plt.imshow(thresh_image, cmap='gray')
    plt.title("After Ensuring Tissue is White")
    plt.show()
    # --- END MODIFICATION ---

    # --- START REVISED MORPHOLOGICAL OPERATIONS ---
    print("Starting revised morphological operations...")

    # Let's first display the raw thresholded image (tissue white) to see the baseline
    plt.figure(figsize=(8, 8))
    plt.imshow(thresh_image, cmap='gray')
    plt.title("Initial Thresholded Image (Tissue White)")
    plt.show()

    # Goal 1: Fill holes within each oyster piece without merging them if possible.
    # We'll use MORPH_CLOSE. Let's try with a slightly smaller kernel or fewer iterations first.
    kernel_close = np.ones((3, 3), np.uint8)  # Smaller kernel than (5,5) previously
    closed_image = cv2.morphologyEx(thresh_image, cv2.MORPH_CLOSE, kernel_close,
                                    iterations=2)  # Iterations can be 1 or 2

    plt.figure(figsize=(8, 8))
    plt.imshow(closed_image, cmap='gray')
    plt.title("After MORPH_CLOSE (Aim: Fill internal holes)")
    plt.show()

    # Goal 2: Remove small noise particles and try to break very thin connections
    # that might be incorrectly linking separate pieces, without destroying delicate parts of actual oysters.
    # We'll use MORPH_OPEN. Let's be gentle.
    kernel_open = np.ones((3, 3), np.uint8)
    opened_image = cv2.morphologyEx(closed_image, cv2.MORPH_OPEN, kernel_open, iterations=1)  # Gentle opening

    # Potentially, another very light closing if opening created small internal holes in main tissue
    # opened_image = cv2.morphologyEx(opened_image, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8), iterations=1)

    plt.figure(figsize=(10, 10))
    plt.imshow(opened_image, cmap='gray')
    plt.title("After Morphological Operations (Revised - Tissue White)")
    plt.axis('off')
    plt.show()
    # --- END REVISED MORPHOLOGICAL OPERATIONS ---

    # Find Contours (This part of the code remains the same)
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
            cv2.rectangle(contour_img_viz_selected, (x, y), (x + w, y + h), (255, 0, 0), 5) # Blue box
            cv2.putText(contour_img_viz_selected, f"Oyster {i+1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,0), 3)

        plt.figure(figsize=(12,12))
        plt.imshow(contour_img_viz_selected)
        plt.title(f"Selected {len(selected_oyster_contours)} Oyster Contours with Bounding Boxes")
        plt.axis('off')
        plt.show()

        print(f"Generated Bounding Boxes for SAM: {oyster_bounding_boxes}")
else:
    print("Skipping prompt generation.")

