# s3_process_annotated_region_patched.py
#
# Description:
# This script processes a single, exported OME-TIFF image region using
# annotations from a GeoJSON file. To ensure maximum accuracy for small objects,
# it operates in a patch-based manner for each annotation.

# !! This script assumes a region starting from a brand-new coordinate system (0,0).
#
# For each annotation (e.g., a polygon or point) in the GeoJSON file, it will:
# 1. Define and extract a small, high-resolution patch around the annotation.
# 2. Set the SAM Predictor's context using this patch.
# 3. Convert the annotation's global coordinates to local coordinates within the patch.
# 4. Use the local prompt to generate a precise segmentation mask.
# 5. Re-create a full-size mask and place the patch's mask in the correct global location.
# 6. Save the final binary mask and a detailed visualization of the patch prediction.
#
# Usage:
# 1. Update the configuration variables in the SCRIPT_CONFIG section.
# 2. Run the script: python s3_process_annotated_region_patched.py

import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import torch
import tifffile
import geojson
from shapely.geometry import shape
from segment_anything import sam_model_registry, SamPredictor
import logging
import time
import xml.etree.ElementTree as ET

# --- SCRIPT CONFIGURATION ---
# Paths
TIFF_PATH = '/Volumes/One Touch/MSX Project/Test Script/test_region_export/U5014-25 E_01.vsi - 20x_BF_01_Border_RegionExport.ome.tif'
GEOJSON_PATH = '/Volumes/One Touch/MSX Project/Test Script/test_region_export/annotations/U5014-25 E_01.vsi - 20x_BF_01_all_annotations.geojson'
SAM_CHECKPOINT_PATH = "pretrained_checkpoint/sam_vit_h_4b8939.pth"
SAM_MODEL_TYPE = "vit_h"
OUTPUT_DIR = "output_stage3_predictor_patched"

# Patching Configuration
# Amount of padding (in pixels) to add around a bounding box to create the patch.
# This gives SAM some context around the object.
PATCH_PADDING = 100

# Control plotting for debugging (will show plots live if True)
ENABLE_PLOTTING = False
MAX_PATCHES_TO_PROCESS = 10
# --- END SCRIPT CONFIGURATION ---

# --- Global Logger ---
logger = logging.getLogger(__name__)


def setup_logging(log_dir):
    """
    Configures logging to console and file.

    :param log_dir: Directory where log files will be saved.
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_file_path = os.path.join(log_dir, f"stage3_processing_{timestamp}.log")
    if logger.hasHandlers(): logger.handlers.clear()

    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info("Logging setup complete. Log file: %s", log_file_path)


def load_sam_predictor(checkpoint_path, model_type):
    """
    Initializes and returns the SAM predictor.

    :param checkpoint_path: Path to the SAM model checkpoint.
    :param model_type: Type of SAM model to use (e.g., "vit_h").
    """
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using SAM device: {device}")
    try:
        logger.info("Loading SAM model for predictor...")
        sam_model = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam_model.to(device=device)
        predictor = SamPredictor(sam_model)
        logger.info("SAM predictor loaded successfully.")
        return predictor
    except Exception as e:
        logger.error(f"🛑 Error loading SAM model for predictor: {e}", exc_info=True)
        return None


def load_image_and_offsets(image_path):
    """
    Loads an OME-TIFF image and extracts the X/Y pixel offsets from its metadata.

    :param image_path: Path to the OME-TIFF image file.
    :return: Tuple of (image in RGB uint8 format, x_offset_pixels, y_offset_pixels).
    """
    logger.info(f"Loading image and metadata: {image_path}")
    x_offset_pixels, y_offset_pixels = 0, 0
    try:
        with tifffile.TiffFile(image_path) as tif:
            image = tif.asarray()

            # Attempt to parse OME-XML metadata for offsets
            if tif.ome_metadata:
                root = ET.fromstring(tif.ome_metadata)
                # Namespace is often present in OME-XML, so we need to handle it
                ns = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
                plane = root.find('ome:Image[1]/ome:Pixels/ome:Plane[1]', ns)
                if plane is not None:
                    # Offsets are usually in microns, so we need pixel size to convert
                    pixels_node = root.find('ome:Image[1]/ome:Pixels', ns)
                    phys_size_x = float(pixels_node.get('PhysicalSizeX', 1.0))
                    phys_size_y = float(pixels_node.get('PhysicalSizeY', 1.0))

                    pos_x_microns = float(plane.get('PositionX', 0.0))
                    pos_y_microns = float(plane.get('PositionY', 0.0))

                    x_offset_pixels = int(round(pos_x_microns / phys_size_x))
                    y_offset_pixels = int(round(pos_y_microns / phys_size_y))
                    logger.info(f"Found OME offsets: X={x_offset_pixels}px, Y={y_offset_pixels}px")
                else:
                    logger.warning("Could not find Plane[1] in OME-XML metadata.")
            else:
                logger.warning("No OME-XML metadata found in TIFF file. Assuming offsets are (0,0).")

        # --- Image format normalization (same as before) ---
        if image.ndim == 3 and image.shape[2] == 4:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif image.ndim == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3 and image.shape[2] > 4:
            image_rgb = image[:, :, :3]
        else:
            image_rgb = image
        if image_rgb.dtype != np.uint8:
            image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)
        logger.info(f"Image loaded successfully, shape: {image_rgb.shape}")

        return image_rgb, x_offset_pixels, y_offset_pixels

    except Exception as e:
        logger.error(f"🛑 Failed to load image or parse offsets from {image_path}: {e}", exc_info=True)
        return None, 0, 0

def load_image(image_path):
    """
    Loads an image using tifffile and ensures it's in RGB uint8 format.

    :param image_path: Path to the OME-TIFF image file.
    :return: Image in RGB uint8 format, or None if loading fails.
    """
    logger.info(f"Loading image: {image_path}")
    try:
        image = tifffile.imread(image_path)
        if image.ndim == 3 and image.shape[2] == 4:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif image.ndim == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3 and image.shape[2] > 3:
            image_rgb = image[:,:,:3]
        else:
            image_rgb = image
        if image_rgb.dtype != np.uint8:
            image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)
        logger.info(f"Image loaded successfully, shape: {image_rgb.shape}")
        return image_rgb
    except Exception as e:
        logger.error(f"🛑 Failed to load image {image_path}: {e}", exc_info=True)
        return None


def load_prompts_and_infer_offsets(geojson_path):
    """
    Loads annotations from GeoJSON, converts them to prompts, AND infers the
    global coordinate offset by finding the min X/Y of all annotations.

    :param geojson_path: Path to the GeoJSON file containing annotations.
    :return: Tuple of (list of prompts, inferred_x_offset, inferred_y_offset).
    """
    logger.info(f"Loading annotations and inferring offset from: {geojson_path}")
    prompts = []
    min_x_overall, min_y_overall = float('inf'), float('inf')
    try:
        with open(geojson_path, 'r') as f:
            features = geojson.load(f)['features']

        for i, feature in enumerate(features):
            geom = shape(feature['geometry'])
            prompt = {'id': i, 'type': None, 'data': None}
            if geom.geom_type in ['Polygon', 'LineString']:
                prompt['type'] = 'box'
                bounds = np.array(geom.bounds)
                prompt['data'] = bounds
                min_x_overall = min(min_x_overall, bounds[0])
                min_y_overall = min(min_y_overall, bounds[1])
            elif geom.geom_type == 'Point':
                prompt['type'] = 'point'
                point_coords = np.array([geom.x, geom.y])
                prompt['data'] = point_coords
                min_x_overall = min(min_x_overall, point_coords[0])
                min_y_overall = min(min_y_overall, point_coords[1])
            else:
                continue
            prompts.append(prompt)

        inferred_x_offset = int(np.floor(min_x_overall))
        inferred_y_offset = int(np.floor(min_y_overall))

        logger.info(f"Successfully loaded {len(prompts)} global prompts.")
        logger.info(f"Inferred offset from annotations: X={inferred_x_offset}, Y={inferred_y_offset}")
        return prompts, inferred_x_offset, inferred_y_offset
    except Exception as e:
        logger.error(f"🛑 Failed to load or parse GeoJSON {geojson_path}: {e}", exc_info=True)
        return [], 0, 0


def save_patch_visualization(patch_image, small_mask, score, local_prompt, output_path):
    """
    Saves a lightweight, zoomed-in visualization of the patch, prompt, and mask.

    :param patch_image: The high-resolution image patch.
    :param small_mask: The binary mask generated by SAM for the patch.
    :param score: The annotation score from SAM.
    :param local_prompt: The local prompt data (box or point).
    :param output_path: Path to save the visualization image.
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(patch_image)

    mask_overlay = np.zeros_like(patch_image, dtype=np.uint8)
    mask_overlay[small_mask] = [30, 144, 255]  # Dodger blue color for mask
    plt.imshow(cv2.addWeighted(patch_image, 0.7, mask_overlay, 0.3, 0))

    if local_prompt['type'] == 'box':
        box = local_prompt['data']
        rect = plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, edgecolor='magenta',
                             linewidth=2)
        plt.gca().add_patch(rect)
    elif local_prompt['type'] == 'point':
        point = local_prompt['data']
        plt.plot(point[0], point[1], 'go', markersize=10)

    plt.title(f"Annotation Score: {score:.3f}", fontsize=12)
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    if ENABLE_PLOTTING:
        plt.show()
    plt.close()


# --- Main Execution ---
if __name__ == "__main__":
    run_timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_output_dir = os.path.join(OUTPUT_DIR, f"{os.path.splitext(os.path.basename(TIFF_PATH))[0]}_{run_timestamp}")
    os.makedirs(run_output_dir, exist_ok=True)

    setup_logging(run_output_dir)

    logger.info("--- Stage 3 (Patched): Segmentation from Annotations ---")

    predictor = load_sam_predictor(SAM_CHECKPOINT_PATH, SAM_MODEL_TYPE)
    if not predictor: exit(1)

    image_rgb = load_image(TIFF_PATH)
    if image_rgb is None: exit(1)
    img_h, img_w = image_rgb.shape[:2]

    prompts, x_offset, y_offset = load_prompts_and_infer_offsets(GEOJSON_PATH)
    if not prompts: exit(1)

    logger.info(f"Starting patch-based processing for {len(prompts)} prompts...")
    for i, prompt in enumerate(prompts):
        if i >= MAX_PATCHES_TO_PROCESS:
            logger.debug(f"  > Reached max patches to process ({MAX_PATCHES_TO_PROCESS}). Stopping.")
            break
        logger.info(f"--- Processing Annotation {i + 1}/{len(prompts)} (ID: {prompt['id']}) ---")

        # 1. Define patch coordinates based on the prompt's bounding box
        if prompt['type'] == 'box':
            corrected_bbox = prompt['data'] - np.array([x_offset, y_offset, x_offset, y_offset])
            patch_def_bbox = corrected_bbox
        elif prompt['type'] == 'point':
            corrected_point = prompt['data'] - np.array([x_offset, y_offset])
            px, py = corrected_point
            half_size = PATCH_PADDING
            patch_def_bbox = np.array([px - half_size, py - half_size, px + half_size, py + half_size])

            # Define patch coordinates with padding
        patch_x1 = int(max(0, patch_def_bbox[0] - PATCH_PADDING))
        patch_y1 = int(max(0, patch_def_bbox[1] - PATCH_PADDING))
        patch_x2 = int(min(img_w, patch_def_bbox[2] + PATCH_PADDING))
        patch_y2 = int(min(img_h, patch_def_bbox[3] + PATCH_PADDING))

        logger.debug(f"  Global Coords: {np.round(prompt['data'], 2)}")
        logger.debug(f"  Corrected BBox Def: {np.round(patch_def_bbox, 2)}")
        logger.debug(f"  Patch Coords: X={patch_x1}-{patch_x2}, Y={patch_y1}-{patch_y2}")

        # 2. Extract the high-resolution patch from the main image
        patch_image = image_rgb[patch_y1:patch_y2, patch_x1:patch_x2]
        if patch_image.size <= 0:
            logger.warning(f"  ! Skipping annotation {prompt['id']} because its patch is empty (likely out of bounds).")
            continue

        # 3. Set the image for the predictor (this is the key step)
        logger.debug(f"  Setting patch image (size {patch_image.shape[1]}x{patch_image.shape[0]}) in SAM.")
        predictor.set_image(patch_image)

        # 4. Transform corrected global coords to local patch coords
        local_prompt = {'type': prompt['type']}
        if prompt['type'] == 'box':
            local_box = corrected_bbox - np.array([patch_x1, patch_y1, patch_x1, patch_y1])
            local_prompt['data'] = local_box
            sam_box_prompt, sam_point_prompt, sam_label_prompt = local_box[np.newaxis, :], None, None
        elif prompt['type'] == 'point':
            local_point = corrected_point - np.array([patch_x1, patch_y1])
            local_prompt['data'] = local_point
            sam_point_prompt, sam_label_prompt, sam_box_prompt = local_point[np.newaxis, :], np.array([1]), None

        logger.debug(f"  Local Prompt Coords: {np.round(local_prompt['data'], 2)}")

        # 5. Predict on the patch
        try:
            masks_small, scores, _ = predictor.predict(
                point_coords=sam_point_prompt,
                point_labels=sam_label_prompt,
                box=sam_box_prompt,
                multimask_output=False,
            )
            small_mask = masks_small[0]
            best_score = scores[0]
            logger.info(f"  > Generated mask with score: {best_score:.4f}")

            # 6. Create a full-size mask and paste the small result
            final_mask = np.zeros((img_h, img_w), dtype=bool)
            final_mask[patch_y1:patch_y2, patch_x1:patch_x2] = small_mask

            # 7. Save outputs
            base_filename = f"annotation_{prompt['id']:03d}_{prompt['type']}_score_{best_score:.3f}"
            mask_output_path = os.path.join(run_output_dir, f"{base_filename}_mask.png")
            viz_output_path = os.path.join(run_output_dir, f"{base_filename}_viz.png")

            cv2.imwrite(mask_output_path, final_mask.astype(np.uint8) * 255)
            save_patch_visualization(patch_image, small_mask, best_score, local_prompt, viz_output_path)

        except Exception as e:
            logger.error(f"🛑 Error predicting mask for annotation {prompt['id']}: {e}", exc_info=True)

    logger.info("--- Fin ---")
