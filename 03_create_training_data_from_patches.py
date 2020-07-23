"""
A Script to build training sets for "hand-held" training.

"""
import os
import argparse
from numpy.random import shuffle

# Argparse setup
parser = argparse.ArgumentParser(description="Create training set of patches from n images")
parser.add_argument("-n", type=int, default=1, help="Number of images in training set")
parser.add_argument("--base_dir", type=str, default="./data/", help="Path to data directory")
parser.add_argument("--split", type=float, default=0.8, help="Percent split")
parser.add_argument("--dim", type=int, default=512, help="Dimension of patches")
args = parser.parse_args()

# Patient doubles
DOUBLES = [
            ['SCC_1', 'SCC_2'],
            ['IEC_2', 'IEC_3'],
            ['BCC_1', 'BCC_2'],
            ['BCC_5', 'BCC_6'],
            ['SCC_5', 'SCC_6'],
            ['BCC_7', 'BCC_8'],
            ['BCC_11', 'BCC_12'],
            ['SCC_10', 'SCC_11'],
            ['BCC_19', 'BCC_20', 'BCC_21'],
            ['BCC_23', 'BCC_24'],
            ['BCC_25', 'BCC_26'],
            ['BCC_27', 'BCC_28'],
            ['IEC_4', 'IEC_5'],
            ['IEC_6', 'IEC_7'],
            ['IEC_8', 'IEC_9'],
            ['IEC_10', 'IEC_11'],
            ['BCC_33', 'BCC_34'],
            ['IEC_13', 'IEC_14'],
            ['BCC_37', 'BCC_38'],
            ['BCC_40', 'BCC_41'],
            ['IEC_19', 'IEC_20'],
            ['IEC_22', 'IEC_23'],
            ['IEC_27', 'IEC_28'],
            ['IEC_29', 'IEC_30'],
            ['BCC_46', 'BCC_47'],
            ['SCC_15', 'SCC_16'],
            ['IEC_34', 'IEC_35'],
            ['BCC_55', 'BCC_56'],
            ['IEC_43', 'IEC_44'],
            ['IEC_51', 'IEC_52'],
            ['BCC_60', 'BCC_61'],
            ['SCC_17', 'SCC_18'],
            ['BCC_83', 'BCC_84'],
            ['SCC_34', 'SCC_35']
            ]

# Assign to global names
base_dir = args.base_dir
num_files = args.n
split = args.split
dim = args.dim

# Directory setup
patch_dir = os.path.join(base_dir, "Patches_Overlapped_" + str(dim))
training_dir = os.path.join(base_dir, "TrainingData_" + str(int(split*100)))
cmd = "mkdir -p " + training_dir
os.system(cmd)

# Create Training Set
set_name = "Data_" + str(num_files)
set_dir = os.path.join(training_dir, set_name)
cmd = "mkdir -p " + set_dir
os.system(cmd)

# Create X_train and y_train directories
for dir in ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]:
    cmd = "mkdir -p " + os.path.join(set_dir, dir)
    os.system(cmd)

# Get files and shuffle
files = os.listdir(patch_dir)[0:num_files]
shuffle(files)

# Remove doubles - add again later...
for double in DOUBLES:
    for file in double:
        try:
            files.remove(file)
        except ValueError:
            continue

# Get Train, Validation and Test splits

val_pos = int(split*len(files))
test_pos = val_pos + (len(files) - val_pos) // 2
# Get set file names
training_files = files[0:val_pos]
validation_files = files[val_pos:test_pos]
test_files = files[test_pos::]

if num_files > 30:
    # Add doubles back in
    val_pos = int(split*len(DOUBLES))
    test_pos = val_pos + (len(DOUBLES) - val_pos) // 2
    shuffle(DOUBLES)
    # Train
    for double in DOUBLES[0:val_pos]:
        for file in double:
            training_files.append(file)
    # Val
    for double in DOUBLES[val_pos:test_pos]:
        for file in double:
            validation_files.append(file)
    # Test
    for double in DOUBLES[test_pos::]:
        for file in double:
            test_files.append(file)

# Assign patches to Sets

# TRAINING
for file in training_files:

    # Create list of files in set
    cmd = "echo " + file + " >> " + set_dir + "/train_files.txt"
    os.system(cmd)

    patches = os.listdir(os.path.join(patch_dir, file + "/X"))
    for patch in patches:

        # Source names
        img_patch = os.path.join(patch_dir, file + "/X/" + patch)
        mask_patch = os.path.join(patch_dir, file + "/y/" + patch)

        # Destination names
        img_out = os.path.join(set_dir, "X_train/" + patch)
        mask_out = os.path.join(set_dir, "y_train/" + patch)

        # Create links
        os.symlink(img_patch, img_out)
        os.symlink(mask_patch, mask_out)

# VALIDATION
for file in validation_files:

    # Create list of files in set
    cmd = "echo " + file + " >> " + set_dir + "/validation_files.txt"
    os.system(cmd)

    patches = os.listdir(os.path.join(patch_dir, file + "/X"))
    for patch in patches:

        # Source names
        img_patch = os.path.join(patch_dir, file + "/X/" + patch)
        mask_patch = os.path.join(patch_dir, file + "/y/" + patch)

        # Destination names
        img_out = os.path.join(set_dir, "X_val/" + patch)
        mask_out = os.path.join(set_dir, "y_val/" + patch)

        # Create links
        os.symlink(img_patch, img_out)
        os.symlink(mask_patch, mask_out)

# TEST
for file in test_files:

    # Create list of files in set
    cmd = "echo " + file + " >> " + set_dir + "/test_files.txt"
    os.system(cmd)

    patches = os.listdir(os.path.join(patch_dir, file + "/X"))
    for patch in patches:

        # Source names
        img_patch = os.path.join(patch_dir, file + "/X/" + patch)
        mask_patch = os.path.join(patch_dir, file + "/y/" + patch)

        # Destination names
        img_out = os.path.join(set_dir, "X_test/" + patch)
        mask_out = os.path.join(set_dir, "y_test/" + patch)

        # Create links
        os.symlink(img_patch, img_out)
        os.symlink(mask_patch, mask_out)




