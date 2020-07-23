"""

Training script to train ResNet-UNet architecture
for segmentation on n-classes.

Author: Simon Thomas
Email: simon.thomas@uq.edu.au

Start Date: 09/01/19
Last Update: 11/03/19

    Python v3.6:
        For less than 12 classes the python interpreter must be >=3.6. This is purely
        to ensure that dictionary.keys() returns the items in the original order
        as per update described https://docs.python.org/3/whatsnew/3.6.html#whatsnew36-compactdict

    Data:
        The data directory should include the following subdirectories and each should
        contain the appropriate images: X_train, y_train, X_val, y_val, X_test, y_test

    Usage:
        python patch_training.py --batch_size 1 --epochs 10 --learning_rate 0.001 \
        --dim 512 --num_classes 12 --gpus 1  --log_dir ./logs/ --data data/ \
        --fine_tune --weights ./weights/BS_1_PS_512_C_12_FT_True_E_10_LR_0.001.h5


# Notes about learning phase
https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html


"""

import argparse

from keras.optimizers import Adam
from keras.utils.training_utils import multi_gpu_model

from seg_utils import *
from seg_models import ResNet_UNet, ResNet_UNet_ExtraConv, ResNet_UNet_More_Params
from seg_models import ResNet_UNet_BN, ResNet_UNet_Dropout, ResNet_UNet_Reg, UNet

from numpy.random import seed as set_np_seed
from tensorflow import set_random_seed as set_tf_seed

# Set seed
seed = 1
set_np_seed(seed)
set_tf_seed(seed)


# Argparse setup
parser = argparse.ArgumentParser(description="Execute custom patch training regime")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training. Max of 12 on wiener")
parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training")
parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for training")
parser.add_argument("--weights", type=str, default=None, help="Path to pre-trained weights to load")
parser.add_argument("--dim", type=int, default=512, help="Patch size - Note: >512 may cause memory issues")
parser.add_argument("--num_classes", type=int, default=12, help="Number of classes to classify")
parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs available on machine")
parser.add_argument("--fine_tune", dest='fine_tune', action='store_true', help="Whether to fine-tune model")
parser.add_argument("--log_dir", type=str, default="logs", help="Path to tensorboard log directory")
parser.add_argument("--data_dir", type=str, default="./data/", help="Path to data directory")
parser.add_argument("--output_dir", type=str, default="./", help="Path to output directory")
parser.add_argument("--classes", type=str, nargs="+", default=None, help="Not yet implemented")
parser.add_argument("--weight_mod", nargs="*", default="F", type=str, help="Class loss weight modifications")
parser.set_defaults(fine_tune=False)
args = parser.parse_args()

# Assign to global names
batch_size = args.batch_size
epochs = args.epochs
learning_rate = args.learning_rate
weights = args.weights
dim = args.dim
num_classes = args.num_classes
gpus = args.gpus
fine_tune = args.fine_tune
log_dir = args.log_dir
data_dir = args.data_dir
output_dir = args.output_dir
classes = args.classes
weight_mod = args.weight_mod
model_type = UNet


# Create unique run name
run_name = str(data_dir.split("/")[-2]) + "_BS_" + str(batch_size) + \
           "_PS_" + str(dim) + "_C_" + str(num_classes) + \
           "_FT_" + str(fine_tune) + "_E_" + str(epochs) + \
           "_LR_" + str(learning_rate) + "_WM_" + "_".join(weight_mod) + \
           "_model_" + str(model_type).split(" ")[1] + "_less_params_all_12"

dropout = None
if dropout:
    run_name += "_DO_" + str(dropout)

print("[INF0] - random seed -", seed)
print("[INFO] hyper-parameter details ...")
print("Run Name:", run_name)
print("Batch Size:", batch_size)
print("Epochs:", epochs)
print("Learning Rate:", learning_rate)
print("Weights:", weights)
print("Patch Dim:", dim)
print("Num Classes:", num_classes)
print("GPUS:", gpus)
print("Fine-Tune:", fine_tune)
print("Data:", data_dir)
print("Weight Mod:", "_".join(weight_mod))

# Path & Directory Setup
os.system("mkdir -p " + log_dir)
os.system("mkdir -p " + log_dir + "/" + run_name)
os.system("mkdir -p weights")
os.system("mkdir -p WSI_test")
os.system("mkdir -p WSI_test/images")
os.system("mkdir -p WSI_test/segmentations")

# Training
X_train_dir = os.path.join(data_dir, "X_train")
y_train_dir = os.path.join(data_dir, "y_train")

# Validation
X_val_dir = os.path.join(data_dir, "X_val")
y_val_dir = os.path.join(data_dir, "y_val")


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

# Set up colors to match classes
if not classes:
    classes = list(color_dict.keys())
    colors = [color_dict[key] for key in color_dict.keys()]

# Create palette for generators
palette = Palette(colors)

# Create weight_mod dictionary
if weight_mod != "F":
    keys = [classes.index(tissue_class) for tissue_class in args.weight_mod[0::2]]
    values = [float(val) for val in args.weight_mod[1::2]]
    weight_mod = dict(zip(keys, values))
    print("Modifying weight balances:")
    for key, value in zip(args.weight_mod[0::2], values):
        print(key, value)
else:
    # Change to boolean value
    weight_mod = None

# Create generators
train_gen = SegmentationGen(
                batch_size, X_train_dir,
                y_train_dir, palette,
                x_dim=dim, y_dim=dim,
                weight_mod=weight_mod
                )

val_gen = SegmentationGen(
                batch_size, X_val_dir,
                y_val_dir, palette,
                x_dim=dim, y_dim=dim,
                )

if gpus > 1:
    print("[INFO] training with {} GPUs...".format(gpus))

    # Store a copy of the model on *every* GPU, and then combine
    # then combine the results from the gradient updates from the CPU
    with tf.device("/cpu:0"):
        # Import model
        orig_model = model_type(dim=dim, num_classes=num_classes)

        # Load pre-trained weights
        if weights:
            orig_model.load_weights(weights)

        # Lock / unlock weights for training
        set_weights_for_training(orig_model, fine_tune)

    # Create multi-GPU version
    model = multi_gpu_model(orig_model, gpus=gpus)

else:
    print("[INFO] training with 1 GPU...")

    # Import model for single GPU
    model = model_type(dim=dim, num_classes=num_classes)

    # Load pre-trained weights
    if weights:
        model.load_weights(weights)

    # Lock / unlock weights for training
    set_weights_for_training(model, fine_tune)


# Compile model for training
model.compile(
            optimizer=Adam(lr=learning_rate),
            loss="categorical_crossentropy",
            sample_weight_mode="temporal",
            metrics=["accuracy"],
            weighted_metrics=["accuracy"]
            )

# Create Callbacks
if gpus > 1:
    model_to_save = orig_model
else:
    model_to_save = model

callback_list = [
                    Validation(generator=val_gen,
                               steps=100,  # val_gen.n // val_gen.batch_size,
                               classes=classes, run_name=run_name,
                               color_list=colors,
                               write_graph=False,
                               WSI=False,
                               model_to_save=model_to_save,
                               weight_path="./weights/" + run_name + "_checkpoint_{epoch:03d}.h5",
                               log_dir=log_dir + "/" + run_name),
                ]

# Train
history = model.fit_generator(
                        epochs=epochs,
                        generator=train_gen,
                        steps_per_epoch=train_gen.n // train_gen.batch_size,
                        validation_data=val_gen,
                        validation_steps=val_gen.n // val_gen.batch_size,
                        callbacks=callback_list,
                        )

# Save weights
weight_path = "./weights/" + run_name + ".h5"
print("Saving weights as:", weight_path)
if gpus > 1:
    orig_model.save_weights(weight_path)
else:
    model.save_weights(weight_path)

print("Finished.")



