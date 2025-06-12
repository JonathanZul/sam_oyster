# s2_intra_oyster_segmentation.py
#
# Description:
# This script performs Stage 2 of the oyster analysis pipeline:
# Segments specific anatomical regions (e.g., gills, gonads) within
# an isolated oyster instance (mask obtained from Stage 1).
# It involves reading the original WSI, extracting high-magnification patches
# from within the oyster mask, and applying SAM with various prompting strategies.
#
# Usage:
# 1. Ensure Stage 1 has been run and oyster instance masks are available.
# 2. Update the configuration variables in the SCRIPT_CONFIG section.
# 3. Run the script: python stage2_intra_oyster_segmentation.py

import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import torch
import openslide # For reading the original WSI
import tifffile
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

# "/Users/jonathanzulluna/Desktop/Work Code/MSX Project/Data/Oyster Scans/Test 20X/U18068-24 A_01.vsi"

torch.set_default_dtype(torch.float32)

# --- SCRIPT CONFIGURATION ---
# Paths
ORIGINAL_WSI_PATH = "data/oyster_slide_uncompressed.ome.tif"
STAGE1_MASK_DIR = "oyster_instance_masks_sam/sam_generated_masks" # Directory containing masks from Stage 1
SAM_CHECKPOINT_PATH = "pretrained_checkpoint/sam_vit_h_4b8939.pth"
SAM_MODEL_TYPE = "vit_h"
OUTPUT_DIR_STAGE2 = "stage2_intra_oyster_segmentations"

# WSI & Patching Parameters
TARGET_MAGNIFICATION = 10.0 # Desired magnification for analysis (e.g., 5.0, 10.0, 20.0)
PATCH_SIZE = 512          # Pixel size of square patches to extract (e.g., 256, 512, 1024)
PATCH_OVERLAP = 64        # Overlap between patches (e.g., 0 for no overlap, or a portion of patch_size)
# If your WSI doesn't have objective power in metadata, you'll need to set this manually:
MANUAL_WSI_OBJECTIVE_POWER = 20.0 # Example: if scanned at 20x. Set to None if available in WSI.

# Select which oyster instance mask to process from Stage 1
# This will be used to focus patching.
# Make sure this file exists in STAGE1_MASK_DIR
TARGET_INSTANCE_MASK_FILENAME = "oyster_slide_downsampled_32x_oyster_1_sam_mask_score_0.955.png"

# Downsample factor used when creating the low_res_image for Stage 1 (needed for mask scaling)
STAGE1_DOWNSAMPLE_FACTOR = 32.0 # ⚠️ This MUST match the downsample factor used to create the image that Stage 1 masks were made from

# Control plotting
ENABLE_PLOTTING = True
# --- END SCRIPT CONFIGURATION ---

def load_sam_predictor(checkpoint_path, model_type):
    """Initializes and returns the SAM predictor."""
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using SAM device: {device}")
    try:
        print("Loading SAM model for predictor...")
        sam_model = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam_model.to(device=device)
        predictor = SamPredictor(sam_model)
        print("SAM predictor loaded successfully.")
        return predictor
    except FileNotFoundError:
        print(f"🛑 SAM Checkpoint not found at {checkpoint_path}.")
    except Exception as e:
        print(f"🛑 Error loading SAM model for predictor: {e}")
    return None

def load_wsi(wsi_path, manual_objective_power=None):
    """Loads the WSI using OpenSlide and prints its properties."""
    print(f"Loading WSI: {wsi_path}")
    try:
        wsi = openslide.OpenSlide(wsi_path)
        print("WSI loaded successfully.")
        print(f"  Level count: {wsi.level_count}")
        print(f"  Level dimensions: {wsi.level_dimensions}")
        print(f"  Level downsamples: {wsi.level_downsamples}")

        objective_power_prop = openslide.PROPERTY_NAME_OBJECTIVE_POWER
        if objective_power_prop in wsi.properties:
            base_magnification = float(wsi.properties[objective_power_prop])
            print(f"  Objective Power (from WSI properties): {base_magnification}x")
        elif manual_objective_power is not None:
            base_magnification = float(manual_objective_power)
            print(f"  Objective Power (manual override): {base_magnification}x")
        else:
            base_magnification = None
            print(f"  🛑 WARNING: Objective Power not found in WSI properties and not manually set.")
            print(f"     Cannot accurately determine magnification levels without it.")
        return wsi, base_magnification
    except openslide.OpenSlideError as e:
        print(f"🛑 Error opening WSI {wsi_path}: {e}")
    except Exception as e:
        print(f"🛑 An unexpected error occurred while opening WSI: {e}")
    return None, None

def load_ome_tiff_pyramid(wsi_path, manual_objective_power=None): # Removed target_magnification, not needed here
    print(f"Loading OME-TIFF with tifffile: {wsi_path}")
    try:
        # It's better to pass the TiffFile object or the series object out,
        # rather than reading the whole image with imread here if we want to patch.
        # Let's open it as a context manager to ensure it's handled properly
        # and extract the necessary info.
        # The actual reading of patches will happen later using this info.

        tif = tifffile.TiffFile(wsi_path) # Open the file

        # --- Keep the rest of your logic to parse levels ---
        if not tif.is_ome:
            print("Warning: TIFF file does not appear to be an OME-TIFF according to tifffile.")

        base_magnification = None
        if manual_objective_power:
            base_magnification = float(manual_objective_power)
            print(f"  Using manual objective power: {base_magnification}x")

        num_series = len(tif.series)
        print(f"  Tifffile found {num_series} series.")
        if num_series == 0:
            print("🛑 No series found in TIFF file.")
            tif.close() # Close file before returning
            return None, None, [], []

        main_series = tif.series[0]
        num_levels = len(main_series.levels)
        print(f"  Series 0 has {num_levels} levels (IFDs/pages).")
        # print(f"  Series 0 level dimensions (H,W,C from tifffile): {[level.shape for level in main_series.levels]}")

        level_dimensions_wh = [] # Store as (Width, Height)
        level_downsamples = []

        if num_levels > 0:
            expected_downsamples_from_qupath = [1, 2, 4, 8, 16, 32, 64, 128, 256]
            for i in range(min(num_levels, len(expected_downsamples_from_qupath))):
                lvl_shape = main_series.levels[i].shape # H, W, [C]
                level_dimensions_wh.append((lvl_shape[1], lvl_shape[0])) # W, H
                level_downsamples.append(float(expected_downsamples_from_qupath[i]))
        else:
            print(f"🛑 No levels found in series 0.")
            tif.close()
            return None, None, [], []


        print(f"  Processed level dimensions (W,H): {level_dimensions_wh}")
        print(f"  Processed level downsamples: {level_downsamples}")

        # For now, we return the TiffFile object itself so we can use it later to read regions.
        # The caller will be responsible for closing it.
        # Or, we can pass the main_series object. Let's pass tif object for now.
        return tif, base_magnification, level_dimensions_wh, level_downsamples

    except Exception as e:
        print(f"🛑 Error opening/processing OME-TIFF with tifffile: {e}")
        if 'tif' in locals() and tif: # Ensure tif is defined and not None
            try:
                tif.close()
            except: pass
    return None, None, [], []

def load_stage1_instance_mask(mask_dir, mask_filename, expected_stage1_downsample_factor):
    """Loads the instance mask from Stage 1 and its corresponding downsample factor."""
    instance_mask_path = os.path.join(mask_dir, mask_filename)
    print(f"Loading Stage 1 instance mask: {instance_mask_path}")
    mask_low_res = cv2.imread(instance_mask_path, cv2.IMREAD_GRAYSCALE)

    if mask_low_res is None:
        print(f"🛑 Instance mask {instance_mask_path} not found or could not be loaded.")
        return None
    else:
        # Ensure it's binary 0 or 255
        _, mask_low_res_binary = cv2.threshold(mask_low_res, 127, 255, cv2.THRESH_BINARY)
        print(f"Instance mask loaded, shape: {mask_low_res_binary.shape}")
        if ENABLE_PLOTTING:
            plt.figure(figsize=(6,6))
            plt.imshow(mask_low_res_binary, cmap='gray')
            plt.title(f"Loaded Stage 1 Mask: {mask_filename}")
            plt.show()
        return mask_low_res_binary

def get_level_for_target_mag(base_mag, target_mag, known_downsamples):
    if not base_mag or not target_mag:
        return 0, 1.0 # Default to level 0, downsample 1.0
    if target_mag > base_mag:
        print(f"Warning: Target magnification {target_mag}x is higher than base {base_mag}x. Using base.")
        return 0, 1.0

    ideal_downsample = base_mag / target_mag
    best_level_idx = 0
    smallest_diff = float('inf')

    # Find the known_downsample that is >= ideal_downsample and closest
    # Or, if we want the one that results in magnification >= target_mag,
    # we find the largest downsample that is <= ideal_downsample.
    # Let's aim for magnification >= target_mag, so pick smallest downsample >= ideal_downsample.
    # No, it's the other way: to get effective_mag >= target_mag,
    # we need actual_downsample <= ideal_downsample.
    # So, we want the largest `ds` in `known_downsamples` such that `ds <= ideal_downsample`.

    closest_downsample = 1.0
    for i, ds in enumerate(known_downsamples):
        if ds <= ideal_downsample:
            if (ideal_downsample - ds) < smallest_diff:
                smallest_diff = ideal_downsample - ds
                best_level_idx = i
                closest_downsample = ds
        else: # ds > ideal_downsample, if we haven't found one yet, previous one is best
            if smallest_diff == float('inf'): # If all known_downsamples are > ideal
                best_level_idx = i # Pick the smallest downsample available
                closest_downsample = ds
            break # Stop, as downsamples are increasing

    # If no suitable downsample was found (e.g. ideal_downsample is very small)
    if smallest_diff == float('inf') and len(known_downsamples) > 0:
        best_level_idx = 0
        closest_downsample = known_downsamples[0]

    return best_level_idx, closest_downsample

def load_sam_automask_generator(
        checkpoint_path,
        model_type,
        points_per_side=32,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        min_mask_region_area=100,
        points_per_batch=64,
        box_nms_thresh=0.7,
        crop_n_layers=0,
        crop_nms_thresh=0.7,
        output_mode="binary_mask"
):
    """Initializes and returns the SAM AutomaticMaskGenerator."""
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using SAM device for AutomaticMaskGenerator: {device}")
    try:
        print("Loading SAM model for AutomaticMaskGenerator...")
        sam_model = sam_model_registry[model_type](checkpoint=checkpoint_path)
        if device == "mps":
            sam_model.to(torch.float32)  # Ensure model parameters are float32 before moving to MPS
            # print("  ⚠️ Note: MPS backend requires float32 precision for SAM models.")
            mask_generator = SamAutomaticMaskGenerator(
                model=sam_model,
                points_per_side=points_per_side,
                pred_iou_thresh=pred_iou_thresh,
                stability_score_thresh=stability_score_thresh,
                min_mask_region_area=min_mask_region_area,
                points_per_batch=points_per_batch,
                box_nms_thresh=box_nms_thresh,
                crop_n_layers=crop_n_layers,
                crop_nms_thresh=crop_nms_thresh,
                output_mode=output_mode,
            )
        sam_model.to(device=device)
        print("SAM AutomaticMaskGenerator loaded successfully.")
        return mask_generator
    except FileNotFoundError:
        print(f"🛑 SAM Checkpoint not found at {checkpoint_path}.")
    except Exception as e:
        print(f"🛑 Error loading SAM model for AutomaticMaskGenerator: {e}")
    return None

def show_anns(anns, ax):
    """Helper function to display masks from SamAutomaticMaskGenerator."""
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    # ax.set_autoscale_on(False) # Deprecated
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation'] # This is a boolean mask
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0] # Random color for each mask
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35))) # Show mask with some transparency

# --- Main Execution ---
if __name__ == "__main__":
    print("--- Stage 2: Intra-Oyster Region Segmentation ---")
    os.makedirs(OUTPUT_DIR_STAGE2, exist_ok=True)

    # 1. Load SAM (Predictor for later, MaskGenerator for now)
    # sam_predictor = load_sam_predictor(SAM_CHECKPOINT_PATH, SAM_MODEL_TYPE)
    # if not sam_predictor:
    #     print("🛑 SAM Predictor could not be initialized. Exiting.")
    #     exit()

    sam_mask_generator = load_sam_automask_generator(SAM_CHECKPOINT_PATH, SAM_MODEL_TYPE)
    if not sam_mask_generator:
        print("🛑 SAM AutomaticMaskGenerator could not be initialized. Exiting.")
        exit()

    # 2. Load the original WSI
    wsi_file_obj, base_wsi_magnification, wsi_level_dimensions, wsi_level_downsamples = load_ome_tiff_pyramid(
        ORIGINAL_WSI_PATH,
        MANUAL_WSI_OBJECTIVE_POWER
    )

    if not wsi_file_obj:
        print("🛑 WSI could not be loaded. Exiting.")
        exit()

    # 3. Load the target oyster instance mask from Stage 1
    oyster_instance_mask_stage1_res = load_stage1_instance_mask(
        STAGE1_MASK_DIR,
        TARGET_INSTANCE_MASK_FILENAME,
        STAGE1_DOWNSAMPLE_FACTOR
    )
    if oyster_instance_mask_stage1_res is None:
        wsi_file_obj.close()  # Close the WSI file if we exit
        exit()

    print("\n--- Phase 2.1: Setup and Initial Loading Complete ---")
    print(f"  Original WSI: {os.path.basename(ORIGINAL_WSI_PATH)}")
    print(f"  Base WSI Magnification: {base_wsi_magnification}x")
    print(f"  Target Analysis Magnification: {TARGET_MAGNIFICATION}x")
    print(f"  Patch Size: {PATCH_SIZE}x{PATCH_SIZE} pixels")
    print(f"  Instance Mask (Stage 1): {TARGET_INSTANCE_MASK_FILENAME} (shape: {oyster_instance_mask_stage1_res.shape})")
    print(f"  Stage 1 Downsample Factor (for mask scaling): {STAGE1_DOWNSAMPLE_FACTOR}")

    best_level_for_patches = -1
    actual_downsample_at_level = -1.0
    effective_magnification = -1.0
    dimensions_of_patching_level = (0,0)

    if base_wsi_magnification and TARGET_MAGNIFICATION and wsi_level_downsamples:
        try:
            # Use the helper function with the loaded downsamples
            best_level_for_patches, actual_downsample_at_level = get_level_for_target_mag(
                base_wsi_magnification,
                TARGET_MAGNIFICATION,
                wsi_level_downsamples  # Pass the list of downsamples we got from tifffile processing
            )
            effective_magnification = base_wsi_magnification / actual_downsample_at_level
            dimensions_of_patching_level = wsi_level_dimensions[best_level_for_patches]

            print(f"\nPatching Info:")
            print(
                f"\tTarget downsample for {TARGET_MAGNIFICATION}x: {base_wsi_magnification / TARGET_MAGNIFICATION:.2f}")
            print(
                f"\tBest WSI level index for patching: {best_level_for_patches} (actual downsample: {actual_downsample_at_level:.2f})")
            print(f"\tEffective magnification at this level: {effective_magnification:.2f}x")
            print(f"\tDimensions of this level (W,H): {dimensions_of_patching_level}")
        except Exception as e:
            print(f"Error calculating best level: {e}. Exiting.")
            wsi_file_obj.close()
            exit()

            # print(f"Error calculating best level: {e}. Defaulting to level 0.")
            # best_level_for_patches = 0
            # actual_downsample_at_level = wsi_level_downsamples[0] if wsi_level_downsamples else 1.0
            # effective_magnification = base_wsi_magnification / actual_downsample_at_level
            # dimensions_of_patching_level = wsi_level_dimensions[0] if wsi_level_dimensions else (0, 0)
    else:
        print("Could not determine best patching level due to missing magnification or downsample info.")
        wsi_file_obj.close()
        exit()

    # test_level_idx = 0  # Try 0, then maybe 2, etc.
    # print(f"DEBUG: Attempting to read level {test_level_idx} directly...")
    # try:
    #     debug_level_data = wsi_file_obj.series[0].levels[test_level_idx].asarray()
    #     print(f"DEBUG: Successfully read level {test_level_idx}, shape: {debug_level_data.shape}")
    #     # You could try to plot this debug_level_data if it's not too huge
    # except Exception as e_debug:
    #     print(f"DEBUG: Error reading level {test_level_idx}: {e_debug}")

    # --- MASK SCALING (to the chosen best_level_for_patches resolution) ---
    scaled_instance_mask_for_patching_binary = None
    if actual_downsample_at_level > 0:
        print(f"\nScaling Stage 1 mask to the resolution of WSI Level {best_level_for_patches}...")
        # dimensions_of_patching_level is (Width, Height)
        scaled_instance_mask_for_patching = cv2.resize(
            oyster_instance_mask_stage1_res,
            (dimensions_of_patching_level[0], dimensions_of_patching_level[1]),
            # cv2.resize dsize is (width, height)
            interpolation=cv2.INTER_NEAREST
        )
        _, scaled_instance_mask_for_patching_binary = cv2.threshold(
            scaled_instance_mask_for_patching, 127, 255, cv2.THRESH_BINARY
        )
        print(
            f"Scaled Stage 1 mask to Level {best_level_for_patches} resolution, new shape: {scaled_instance_mask_for_patching_binary.shape}"
        )

        # Uncomment for debugging/visualization
        # if ENABLE_PLOTTING:
        #     try:
        #         # Load the actual image plane we will be patching from for visualization
        #         image_plane_for_patching = wsi_file_obj.series[0].levels[best_level_for_patches].asarray()
        #         if image_plane_for_patching.ndim == 3 and image_plane_for_patching.shape[2] == 4:  # RGBA
        #             image_plane_for_patching = cv2.cvtColor(image_plane_for_patching, cv2.COLOR_RGBA2RGB)
        #         elif image_plane_for_patching.ndim == 2:  # Grayscale
        #             image_plane_for_patching = cv2.cvtColor(image_plane_for_patching, cv2.COLOR_GRAY2RGB)
        #         # Add other conversions if needed (e.g. if image_plane_for_patching.shape[2] == 1)
        #
        #         plt.figure(figsize=(12, 12))
        #         overlay_viz = image_plane_for_patching.copy()
        #         # Ensure mask_viz_rgb is created correctly for overlay
        #         mask_for_viz_rgb = np.zeros_like(image_plane_for_patching)
        #         mask_for_viz_rgb[scaled_instance_mask_for_patching_binary == 255] = [255, 0,
        #                                                                              0]  # Red where mask is white
        #
        #         alpha = 0.3
        #         cv2.addWeighted(mask_for_viz_rgb, alpha, overlay_viz, 1 - alpha, 0, overlay_viz)
        #         plt.imshow(overlay_viz)
        #         plt.title(
        #             f"Scaled Stage 1 Mask Overlaid on Patching Level {best_level_for_patches} (~{effective_magnification:.2f}x)")
        #         plt.show()
        #     except Exception as e_plot:
        #         print(f"Error generating overlay plot for scaled mask: {e_plot}")
    else:
        print("Could not perform mask scaling due to invalid actual_downsample_at_level.")
        wsi_file_obj.close()
        exit()
    # --- END MASK SCALING ---

    # --- Load the entire image plane for patching ONCE ---
    image_plane_for_patching = None
    if best_level_for_patches != -1:  # Check if best_level was determined
        print(f"\nLoading entire image plane for WSI Level {best_level_for_patches} for patching...")
        try:
            image_plane_for_patching = wsi_file_obj.series[0].levels[best_level_for_patches].asarray()
            if image_plane_for_patching.ndim == 3 and image_plane_for_patching.shape[2] == 4:  # RGBA
                image_plane_for_patching = cv2.cvtColor(image_plane_for_patching, cv2.COLOR_RGBA2RGB)
            elif image_plane_for_patching.ndim == 2:  # Grayscale
                image_plane_for_patching = cv2.cvtColor(image_plane_for_patching, cv2.COLOR_GRAY2RGB)
            # Add other conversions if needed (e.g. if image_plane_for_patching.shape[2] == 1 for some TIFFs)
            print(f"Successfully loaded image plane for patching, shape: {image_plane_for_patching.shape}")

            if ENABLE_PLOTTING:  # Show overlay plot
                plt.figure(figsize=(12, 12))
                overlay_viz = image_plane_for_patching.copy()
                mask_for_viz_rgb = np.zeros_like(image_plane_for_patching)
                mask_for_viz_rgb[scaled_instance_mask_for_patching_binary == 255] = [255, 0, 0]
                alpha = 0.3
                cv2.addWeighted(mask_for_viz_rgb, alpha, overlay_viz, 1 - alpha, 0, overlay_viz)
                plt.imshow(overlay_viz)
                plt.title(
                    f"Scaled Stage 1 Mask Overlaid on Patching Level {best_level_for_patches} (~{effective_magnification:.2f}x)")
                plt.show()
        except Exception as e_load_plane:
            print(f"🛑 Error loading entire image plane for level {best_level_for_patches}: {e_load_plane}")
            image_plane_for_patching = None  # Ensure it's None if loading failed
    else:
        print(f"🛑 Invalid best_level_for_patches: {best_level_for_patches}")

    if image_plane_for_patching is None:
        print("🛑 Cannot proceed with patching as the image plane could not be loaded. Exiting.")
        wsi_file_obj.close()
        exit()
    # --- End loading image plane for patching ---

    print("\n--- Phase 2.2: Generating Patch Coordinates and Extracting Patches ---")

    valid_patches_info = []  # Store dicts with coords and patch data

    level_height = dimensions_of_patching_level[1]  # From earlier calculation
    level_width = dimensions_of_patching_level[0]  # From earlier calculation

    step_size = PATCH_SIZE - PATCH_OVERLAP
    if step_size <= 0:
        print(
            f"🛑 Error: PATCH_OVERLAP ({PATCH_OVERLAP}) >= PATCH_SIZE ({PATCH_SIZE}). Step size would be non-positive.")
        wsi_file_obj.close()
        exit()

    # Number of patches for verbose progress (optional)
    num_potential_patches_y = (level_height - PATCH_SIZE) // step_size + 1 if level_height >= PATCH_SIZE else 0
    num_potential_patches_x = (level_width - PATCH_SIZE) // step_size + 1 if level_width >= PATCH_SIZE else 0
    total_potential_patches = num_potential_patches_y * num_potential_patches_x
    print(f"Iterating over WSI Level {best_level_for_patches} (Dimensions: {level_width}x{level_height})")
    print(f"Patch size: {PATCH_SIZE}x{PATCH_SIZE}, Step size: {step_size}")
    print(f"Potential patches to check (approx): {total_potential_patches}")

    patch_count = 0
    valid_patch_count = 0
    for y_coord in range(0, level_height - PATCH_SIZE + 1, step_size):
        for x_coord in range(0, level_width - PATCH_SIZE + 1, step_size):
            patch_count += 1
            if patch_count % 1000 == 0:  # Adjusted progress print frequency
                print(f"  Checked {patch_count}/{total_potential_patches} potential patch locations...")

            mask_patch_region = scaled_instance_mask_for_patching_binary[
                                y_coord: y_coord + PATCH_SIZE,
                                x_coord: x_coord + PATCH_SIZE
                                ]

            tissue_threshold_pixels = (PATCH_SIZE * PATCH_SIZE) * 0.1
            if cv2.countNonZero(mask_patch_region) > tissue_threshold_pixels:
                valid_patch_count += 1

                # --- CORRECTED PATCH EXTRACTION ---
                # Slice the pre-loaded image_plane_for_patching
                image_patch_data = image_plane_for_patching[
                                   y_coord: y_coord + PATCH_SIZE,
                                   x_coord: x_coord + PATCH_SIZE,
                                   :  # Select all color channels if present
                                   ]
                # --- END CORRECTED PATCH EXTRACTION ---

                # Store coordinates and the patch data itself
                valid_patches_info.append({
                    "x": x_coord, "y": y_coord,
                    "level": best_level_for_patches,
                    "patch_filename_prefix": f"patch_lvl{best_level_for_patches}_x{x_coord}_y{y_coord}",
                    "data": image_patch_data  # Store the patch data
                })

                if ENABLE_PLOTTING and valid_patch_count <= 5:
                    plt.figure(figsize=(5, 5))
                    plt.imshow(image_patch_data)  # image_patch_data is already RGB
                    plt.title(f"Valid Patch {valid_patch_count} (L{best_level_for_patches} @ {x_coord},{y_coord})")
                    plt.axis('off')
                    plt.show()

    print(f"\nPatch generation complete.")
    print(f"  Total potential patch locations checked: {patch_count}")
    print(f"  Number of valid patches found: {len(valid_patches_info)}")
    if valid_patches_info:
        print(f"  (Displayed first {min(5, len(valid_patches_info))} valid patches for visualization)")

        # --- Phase 2.3: Process Each Patch with SAM AutomaticMaskGenerator ---
        print("\n--- Phase 2.3: Processing Patches with SAM AutomaticMaskGenerator ---")
        # Create a subdirectory for this specific oyster instance's patch results
        instance_output_dir = os.path.join(OUTPUT_DIR_STAGE2, os.path.splitext(TARGET_INSTANCE_MASK_FILENAME)[0])
        os.makedirs(instance_output_dir, exist_ok=True)

        # Limit processing for now to a few patches for quick testing
        num_patches_to_process_with_sam = min(10, len(valid_patches_info))  # Process first 5 or fewer
        print(f"Will process first {num_patches_to_process_with_sam} valid patches with SAM AutomaticMaskGenerator...")

        for i, patch_info in enumerate(valid_patches_info[:num_patches_to_process_with_sam]):
            patch_image = patch_info['data']
            patch_filename_prefix = patch_info['patch_filename_prefix']
            print(f"\nProcessing patch {i + 1}/{num_patches_to_process_with_sam}: {patch_filename_prefix}")

            # SAM expects uint8 HWC images
            if patch_image.dtype != np.uint8:
                print(f"  Patch dtype is {patch_image.dtype}, converting to uint8.")
                if patch_image.max() <= 1.0 and patch_image.min() >= 0.0:  # float 0-1
                    patch_image_uint8 = (patch_image * 255).astype(np.uint8)
                else:  # General case
                    patch_image_uint8 = np.clip(patch_image, 0, 255).astype(np.uint8)
            else:
                patch_image_uint8 = patch_image

            if patch_image_uint8.ndim == 3 and patch_image_uint8.shape[2] == 1:  # Grayscale with extra dim
                patch_image_uint8 = cv2.cvtColor(patch_image_uint8, cv2.COLOR_GRAY2RGB)
            elif patch_image_uint8.ndim == 2:  # Grayscale
                patch_image_uint8 = cv2.cvtColor(patch_image_uint8, cv2.COLOR_GRAY2RGB)

            if sam_mask_generator:
                print(f"  Running SAM AutomaticMaskGenerator on patch...")
                generated_masks_anns = sam_mask_generator.generate(patch_image_uint8)
                print(f"  SAM generated {len(generated_masks_anns)} masks for this patch.")

                if generated_masks_anns:
                    patch_output_sub_dir = os.path.join(instance_output_dir, patch_filename_prefix)
                    os.makedirs(patch_output_sub_dir, exist_ok=True)

                    # Option 1: Save individual masks as PNGs
                    for j, ann in enumerate(generated_masks_anns):
                        mask_data = ann['segmentation']  # boolean mask
                        # Include area and predicted_iou in filename for easier sorting/filtering later
                        mask_save_path = os.path.join(
                            patch_output_sub_dir,
                            f"mask_{j:03d}_area_{ann['area']}_iou_{ann['predicted_iou']:.2f}.png"
                        )
                        cv2.imwrite(mask_save_path, mask_data.astype(np.uint8) * 255)
                    print(f"  Saved {len(generated_masks_anns)} individual masks to {patch_output_sub_dir}")

                if ENABLE_PLOTTING and generated_masks_anns:
                    fig, ax = plt.subplots(figsize=(8, 8))
                    ax.imshow(patch_image_uint8)
                    show_anns(generated_masks_anns, ax)  # Use the helper function
                    ax.set_title(f"SAM Auto-Masks for {patch_filename_prefix}")
                    ax.axis('off')
                    # plt.show()
                    plt.savefig(os.path.join(patch_output_sub_dir, patch_filename_prefix + "_plot.png"))
                    print("Saved SAM mask overlay plot for this patch.")

        print("\n--- Stage 2 SAM Processing (Limited Run) Complete ---")

        if wsi_file_obj:
            wsi_file_obj.close()
            print("Closed WSI file object.")
