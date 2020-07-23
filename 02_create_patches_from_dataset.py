import cv2
import os
import argparse
import numpy as np


# Argparse setup
parser = argparse.ArgumentParser(description="Create training set of patches from n images")
parser.add_argument("--dir", type=str, default="./data/", help="Path to data directory")
parser.add_argument("--dim", type=int, default=512, help="Patch size. default 512")
parser.add_argument("--overlap", dest="overlap", action="store_true", help="Boolean to signal overlaping tiling")
parser.set_defaults(overlap=False)
args = parser.parse_args()

# Assign to global names
base_dir = args.dir
dim = args.dim
overlap = args.overlap

# Create output folder
if overlap:
     patch_dir = os.path.join(base_dir, "Patches_Overlaped_" + str(dim))
else:
    patch_dir = os.path.join(base_dir, "Patches_" + str(dim))
cmd = "mkdir -p " + patch_dir
os.system(cmd)


# Setup I/O directories
image_in = os.path.join(base_dir, "Images")
mask_in = os.path.join(base_dir, "Masks")

# Get files in dataset
files = os.listdir(image_in)

step = 1
if overlap:
    # Perform overlap tiling
    for file in files:
        print("Processing file", step, "of", len(files))
        step += 1

        fname = file.split(".")[0]

        # Create folder name and subdirectories
        folder = os.path.join(patch_dir, fname)
        cmd = "mkdir -p " + folder
        os.system(cmd)
        for sub_folder in ["X", "y"]:
            cmd = "mkdir -p " + os.path.join(folder, sub_folder)
            os.system(cmd)

        # Load image and mask
        image = cv2.imread(os.path.join(image_in, fname + ".tif"))
        mask = cv2.imread(os.path.join(mask_in, fname + ".png"))

        h, w = image.shape[0], image.shape[1]
        # Compute number of vertical and horizontal steps
        w_steps = w // dim
        w_overlap = (dim - (w % dim)) // w_steps
        h_steps = h // dim
        h_overlap = (dim - (h % dim)) // h_steps
        # starting positions
        w_x, w_y = 0, dim
        h_x, h_y = 0, dim

        count = 1
        # Loop through all tiles
        for i in range(h_steps+1):
            for j in range(w_steps+1):
                
                # Grab tiles
                image_patch = image[h_x:h_y, w_x:w_y, :]
                mask_patch = mask[h_x:h_y, w_x:w_y, :]
                
                # Check dim will fit in image
                if image_patch.shape[0] < dim or image_patch.shape[1] < dim:
                    # Increment
                    w_x += dim
                    w_y += dim
                    continue

                # Filter out background patches
                if np.sum(mask_patch) == 0:
                    # Increment
                    w_x += dim
                    w_y += dim
                    continue

                # Save patches
                image_name = "X/" + fname + "_" + "{:04d}.png".format(count)
                image_path = os.path.join(folder, image_name)

                mask_name = "y/" + fname + "_" + "{:04d}.png".format(count)
                mask_path = os.path.join(folder, mask_name)

                print("Saving...", image_path, mask_path)

                cv2.imwrite(image_path, image_patch)
                cv2.imwrite(mask_path, mask_patch)

                count += 1

                # Update column positions
                w_x += dim - w_overlap
                w_y += dim - w_overlap
            
            # Update row positions
            h_x += dim - h_overlap
            h_y += dim - h_overlap
            w_x, w_y = 0, dim



else:
    # Sliding window tiling
    for file in files:
        print("Processing file", step, "of", len(files))
        step += 1
        
        # Get file name
        fname = file.split(".")[0]

        # Create folder name and subdirectories
        folder = os.path.join(patch_dir, fname)
        cmd = "mkdir -p " + folder
        os.system(cmd)
        for sub_folder in ["X", "y"]:
            cmd = "mkdir -p " + os.path.join(folder, sub_folder)
            os.system(cmd)

        # Load image and mask
        image = cv2.imread(os.path.join(image_in, fname + ".tif"))
        mask = cv2.imread(os.path.join(mask_in, fname + ".png"))

        # Calculate number of steps to tile image
        row_steps = (image.shape[0] // dim) + 1
        col_steps = (image.shape[1] // dim) + 1

        count = 1
        row_pos = 0
        col_pos = 0
        for r in range(row_steps):
            for c in range(col_steps):

                image_patch = image[row_pos:row_pos+dim, col_pos:col_pos+dim, :]
                mask_patch = mask[row_pos:row_pos+dim, col_pos:col_pos+dim, :]
                # Check dim will fit in image
                if image_patch.shape[0] < dim or image_patch.shape[1] < dim:
                    # Increment
                    col_pos += dim
                    continue

                # Filter out background patches
                if np.sum(mask_patch) == 0:
                    # Increment
                    col_pos += dim
                    continue

                # Save patches
                image_name = "X/" + fname + "_" + "{:04d}.png".format(count)
                image_path = os.path.join(folder, image_name)

                mask_name = "y/" + fname + "_" + "{:04d}.png".format(count)
                mask_path = os.path.join(folder, mask_name)

                print("Saving...", image_path, mask_path)

                cv2.imwrite(image_path, image_patch)
                cv2.imwrite(mask_path, mask_patch)

                count += 1

                col_pos += dim

            row_pos += dim
            col_pos = 0


print("done.")




