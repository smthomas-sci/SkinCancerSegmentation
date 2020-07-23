"""

A quick script to evaluate a trained model.

Author: Simon Thomas
Email: simon.thomas@uq.edu.au

Start Date: 11/03/19
Last Update: 11/03/19

"""

import argparse

from keras.optimizers import Adam

from seg_utils import *
from seg_models import ResNet_UNet, ResNet_UNet_Dropout

from sklearn.metrics import roc_curve, auc

from numpy.random import seed as set_np_seed
from tensorflow import set_random_seed as set_tf_seed

# Set seed
seed = 1
set_np_seed(seed)
set_tf_seed(seed)


# Argparse setup
parser = argparse.ArgumentParser(description="Execute custom patch training regime")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training. Max of 12 on wiener")
parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for training")
parser.add_argument("--weights", type=str, default=None, help="Path to pre-trained weights to load")
parser.add_argument("--dim", type=int, default=512, help="Patch size - Note: >512 may cause memory issues")
parser.add_argument("--num_classes", type=int, default=12, help="Number of classes to classify")
parser.add_argument("--data", type=str, default="./data/", help="Path to data directory")
parser.add_argument("--set", type=str, default="val", help="Path to data directory")
args = parser.parse_args()

# Assign to global names
batch_size = args.batch_size
learning_rate = args.learning_rate
weights = args.weights
dim = args.dim
num_classes = args.num_classes
data_dir = args.data
data_set = args.set

print("[INFO] - EVALUATION RUN")
print("[INF0] - random seed -", seed)
print("[INFO] hyper-parameter details ...")
print("Batch Size:", batch_size)
print("Learning Rate:", learning_rate)
print("Weights:", weights)
print("Patch Dim:", dim)
print("Num Classes:", num_classes)
print("Data:", data_dir)
print("Set:", data_set)

# Path & Directory Setup
X_eval_dir = os.path.join(data_dir, "X_" + data_set)
y_eval_dir = os.path.join(data_dir, "y_" + data_set)


# Create color palette
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

# Create color and palette for generators
classes = list(color_dict.keys())
colors = [color_dict[key] for key in classes]
palette = Palette(colors)

# Create generator
eval_gen = SegmentationGen(
                batch_size, X_eval_dir,
                y_eval_dir, palette,
                x_dim=dim, y_dim=dim,
                )

# Import model for single GPU
model = ResNet_UNet(dim=dim, num_classes=num_classes)

# Load pre-trained weights
if weights:
    model.load_weights(weights)

# Lock weights for evaluation
model.trainable = False

# Compile model for training
model.compile(
            optimizer=Adam(lr=learning_rate),
            loss="categorical_crossentropy",
            sample_weight_mode="temporal",
            metrics=["accuracy"],
            weighted_metrics=["accuracy"]
            )

# Evaluate
print("[INFO] evaluating...")
results = model.evaluate_generator(generator=eval_gen,
                                   steps=eval_gen.n // eval_gen.batch_size,
                                   verbose=1)
print("Loss:", results[0], "Acc:", results[1], "Weighted Acc:", results[-1])

# Confusion Matrix
epoch_cm = np.zeros((len(classes), len(classes)))

# ROC Analysis
ROC = {}

# Loop through validation set
for n in range(eval_gen.n // eval_gen.batch_size):

    print("Step", n+1, "of", eval_gen.n // eval_gen.batch_size)

    # Grab next batch
    X, y_true, _ = next(eval_gen)

    # Make prediction with model
    y_pred = model.predict(X)

    # Calculate ROC values
    for idx, tissue_type in enumerate(classes):
        # Get scores
        true = np.ravel(y_true[:, :, idx])
        pred = np.ravel(y_pred[:, :, idx])

        # Save to disk
        folder = "./ROC/" + tissue_type
        os.system("mkdir -p " + folder)
        fname = folder + "/" + str(n) + "_true"
        np.save(fname, true)
        fname = folder + "/" + str(n) + "_pred"
        np.save(fname, pred)

    # Find highest classes prediction
    y_true = np.argmax(y_true, axis=-1)
    y_pred = np.argmax(y_pred, axis=-1)

    # Flatten batch into single array
    y_true = np.ndarray.flatten(y_true)
    y_pred = np.ndarray.flatten(y_pred)

    # Create batch CM
    batch_cm = ConfusionMatrix(y_true, y_pred)

    # Get all classes in batch
    all_classes = list(batch_cm.classes)

    batch_cm = batch_cm.to_array()

    # Update epoch CM
    for i in all_classes:
        for j in all_classes:
            epoch_cm[i, j] += batch_cm[all_classes.index(i), all_classes.index(j)]


# Create Colorful CM
# Compute row sums for Recall
row_sums = epoch_cm.sum(axis=1)
matrix = np.round(epoch_cm / row_sums[:, np.newaxis], 3)

# Set up colors
# Set up colors
color = [255, 118, 25]
orange = [c / 255. for c in color]
pink = [ c / 255. for c in [235, 66, 244]]
purple = [c / 255. for c in [209, 66, 244]]

white_orange = LinearSegmentedColormap.from_list("", ["white", orange])
white_pink = LinearSegmentedColormap.from_list("", ["white", pink])
white_purple = LinearSegmentedColormap.from_list("", ["white", purple])

# Plot
fig = plt.figure(figsize=(12, 14))
ax = plt.gca()
im = ax.matshow(matrix, interpolation='nearest', cmap=white_purple)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.2)
plt.colorbar(im, cax=cax)


ax.set_xticklabels([''] + classes, fontsize=8)
ax.set_yticklabels([''] + classes, fontsize=8)

# Get ticks to show properly
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(1))

#ax.set_title("Recall - " + X_eval_dir)
ax.set_ylabel("Ground Truth", fontsize=15)
ax.set_xlabel("Predicted", fontsize=15)

for i in range(len(classes)):
    for j in range(len(classes)):
        ax.text(j - 0.1, i, str(matrix[i, j]), fontsize=8)


out_dir = "/clusterdata/s4200058/"
out_dir = "/home/simon/Desktop/"
plt.savefig( out_dir + data_set + "_CM.png", format="png")
plt.close(fig)

# Save CM
fname = out_dir + "CM"
print("Saving CM to", fname)
np.save(fname, epoch_cm)


# ------------ #
# ROC ANALYSIS
# ------------ #

for tissue_type in classes:
    path = os.path.join("./ROC/", tissue_type)

    # Initialise arrays
    true = np.array([])
    pred = np.array([])

    # Load in files
    for n in range(eval_gen.n // eval_gen.batch_size):
        # Load true files
        file = os.path.join(path, str(n) + "_true.npy")
        true = np.append(true, np.load(file, mmap_mode="r"))

        # Load pred files
        file = os.path.join(path, str(n) + "_pred.npy")
        pred = np.append(pred, np.load(file, mmap_mode="r"))

    # Calculate ROC
    fpr, tpr, thresholds = roc_curve(true, pred)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(tissue_type + " - AUC: " + str(roc_auc))
    plt.savefig(out_dir + tissue_type + "_ROC.png", format="png")
    plt.close()

    del true, pred

print("Finished.")

