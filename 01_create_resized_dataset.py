import cv2
import os


# Input-directory setup
base_dir = "/home/simon/Documents/PhD/Data/Histo_Segmentation/"
image_in = os.path.join(base_dir, "Images")
mask_in = os.path.join(base_dir, "Masks")


# Reduction factor
factor = 10
ratio = 1 / factor


# Output-directory setup
out_dir = os.path.join("/home/simon/Documents/PhD/Data/Histo_Segmentation/Datasets_n290/", str(factor)+"x")
image_out = os.path.join(out_dir, "Images")
mask_out = os.path.join(out_dir, "Masks")


# Create necessary folders
for folder in [out_dir, image_out, mask_out]:
    cmd = "mkdir -p " + folder
    os.system(cmd)


# Get file list
files = os.listdir(image_in)

# Process images...
step = 1
for file in files:
    print("Processing image", step, "of", len(files))
    # Get name
    fname = file.split(".")[0]

    # Import
    image = cv2.imread(os.path.join(image_in, fname + ".tif"))
    mask = cv2.imread(os.path.join(mask_in, fname + ".png"))

    # Resize
    image = cv2.resize(image, (0,0), fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
    mask = cv2.resize(mask, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)

    # Save
    cv2.imwrite(os.path.join(image_out, fname + ".tif"), image)
    cv2.imwrite(os.path.join(mask_out, fname + ".png"), mask)

    step += 1
