
import numpy as np
import skimage.io as io
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


input_dir = "/home/simon/Desktop/10x_Experiments_Over_Aug/ALL_PROBMAPS/"
output_dir = "/home/simon/Desktop/10x_Experiments_Over_Aug/ALL_CONFIDENCE/"

# Remove healthy
files = [file for file in os.listdir(input_dir) if "HEAL" not in file]

for i, file in enumerate(files):

    print(i/len(files))

    fname = os.path.join(input_dir, file)

    # Load
    probs = np.load(fname)

    # Get max pos
    conf = 1 - np.max(probs, axis=-1)

    # Plot results
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)

    im = ax.imshow(conf, cmap="magma", vmin=0, vmax=1)

    # Modify color bar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0)
    colbar = fig.colorbar(im, cax=cax)
    colbar.ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1])
    colbar.outline.set_edgecolor((0,0,0,0))
    colbar.ax.tick_params(length=0)

    ax.axis('off')
    #plt.show()

    save_name = os.path.join(output_dir, file.split('.')[0] + ".png")

    plt.savefig(save_name, dpi=300)
    plt.close()

    del probs
