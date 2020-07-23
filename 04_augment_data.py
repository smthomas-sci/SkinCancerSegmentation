
"""
Augment Training Data via simple Duplication
"""

import os
import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt

color_dict = {
    "EPI":  [73, 0, 106],
    "GLD":  [108, 0, 115],
    "INF":  [145, 1, 122],
    "RET":  [181, 9, 130],
    "FOL":  [216, 47, 148],
    "PAP":  [236, 85, 157],
    "HYP":  [254, 246, 242],
    "KER":  [248, 123, 168],
    "BKG":  [0, 0, 0],
    "BCC":  [127, 255, 255],
    "SCC":  [127, 255, 142],
    "IEC":  [255, 127, 127]
}



X_dir = "/home/simon/Documents/PhD/Data/Histo_Segmentation/Datasets_n290/2x/TrainingData/2x_n_290/X_train"
y_dir = "/home/simon/Documents/PhD/Data/Histo_Segmentation/Datasets_n290/2x/TrainingData/2x_n_290/y_train"


files = os.listdir(X_dir)




for i, file in enumerate(files):

    # Import mask
    mask = io.imread(os.path.join(y_dir, file))

    for key in ["BCC", "SCC", "IEC"]:
        present = np.any(np.all(mask == tuple(color_dict[key]), axis=-1))
        if present:

            # Load image
            image = io.imread(os.path.join(X_dir, file))

            # Augment
            fs = ["LR", "UD"]
            for f, func in enumerate([np.fliplr, np.flipud]):
                mask_aug = func(mask)
                image_aug = func(image)

                # Rotate
                deg = ["0", "90", "180", "270"]
                for k in range(4):
                    mask_out = np.rot90(mask_aug, k)
                    image_out = np.rot90(image_aug, k)

                    # Save
                    fname = file.split(".")[0] + "_" + fs[f] + "_" + deg[k] + ".png"
                    io.imsave(os.path.join(y_dir, fname), mask_out)
                    io.imsave(os.path.join(X_dir, fname), image_out)

            # Move to next
            break
    print("Step", i, "of", len(files))

