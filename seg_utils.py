"""

A collection of classes and functions for training fully-convolutional
CNNs for semantic segmentation. Includes a custom generator class for
performing model.fit_genertor() method in Keras. This is specifically used
for segementation ground truth labels which requires sample weights
to be used. (See https://github.com/keras-team/keras/issues/3653)

Author: Simon Thomas
Email: simon.thomas@uq.edu.au

Start Date: 24/10/18
Last Update: 24/06/19

"""

import numpy as np
from sys import stderr
import h5py
import os
import io as IO

import skimage.io as io
from skimage.measure import regionprops
from skimage.morphology import disk
from skimage.transform import rotate
from skimage.filters import median

from sklearn.linear_model import LinearRegression


from cv2 import resize, cvtColor, COLOR_RGB2BGR, imwrite

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as cols
from matplotlib.pyplot import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

from keras.callbacks import Callback, TensorBoard
from keras.models import Model
import keras.backend as K
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight

from pandas_ml import ConfusionMatrix


class Palette(object):
    """
    A color pallete which is essentially a channel
    colour LUT.
    """
    def __init__(self, ordered_list):
        """
        Takes and order list of colours and stores in dictionary
        format.

        Input:

            ordered_list - list of rgb tuples in class order

        Output:

            self[index] - rgb tuple associated with index/class
        """

        self.colors = dict((i, color) for (i, color) in enumerate(ordered_list))

    def __getitem__(self, arg):
        """
        Returns item with input key, i.e. channel
        """
        return self.colors[arg]


    def __str__(self):
        """
        Print representation
        """
        return "Channel Color Palette:\n" + str(self.colors)

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.colors.keys())


class SegmentationGen(object):

    """
    A generator that returns X, y & sampe_weight data in designated batch sizes,
    specifically for segmentation problems. It converts y images 2D arrays
    to work with sample_weights which are calculated on the fly for each batch.

    The output predictions are in shape (batch_size, dim*dim*, num_classes)
    and therefore need to be reshaped inorder to be interpreted / visualised.

    Validation needs to be done separately due to implementation differences
    with keras, but remains fairly straight forward.

    Training is very sensitive to batchsize and traininging rate.

    See again:  https://github.com/keras-team/keras/issues/3653

    Example:

        >>> colours = [(0,0,255),(0,255,0),(255,0,0)]
        >>> palette = Palette(colours)
        >>> batch_size = 10
        >>> dim = 128
        >>> train_gen = SegmentationGen(batch_size, X_dir, y_dir, palette,
        ...                        x_dim=dim, y_dim=dim)
        >>> # Compile mode - include sampe_weights_mode="temporal"
        ... model.compile(  optimizer=SGD(lr=0.001),
        ...                 loss="categorical_crossentropy",
        ...                 sample_weight_mode="temporal",
        ...                 metrics=["accuracy"])
        >>> # Train
        ... history = model.fit_generator(
        ...                 generator = train_gen,
        ...                 steps_per_epoch = train_gen.n // batch_size
        ...                 validation_data = val_gen,
        ...                 validation_steps = val_gen.n // batch_size )
        >>> # Evaluate
        ... loss, acc = model.evaluate_generator(
        ...                 generator = test_gen,
        ...                 steps = 2)


    Input:

        batch_size - number of images in a batch

        X_dir - full path directory of training images

        y_dir - full path directory of training labels

        palette - color palette object where each index (range(n-classes))
                is the class colour from the segmented ground truth. Get be
                obtained from the LUT of come standard segmentaiton datasets.

        dim - batches require images to be stacked so for
                batch_size > 1 image_size is required.

        suffix - the image type in the raw images. Default is ".png"

        weight_mod -  a dictionary to modify certain weights by index
                      i.e. weight_mod = {0 : 1.02} increases weight 0 by 2%.
                      Default is None.

    Output:
        using the global next() function or internal next() function the class
        returns X_train, y_train numpy arrays:
            X_train.shape = (batch_size, image_size, dim, 3)
            y_train.shape = (batch_size, image_size, dim, num_classes)
    """
    def __init__(self,
                    batch_size, X_dir, y_dir,
                    palette, x_dim, y_dim,
                    suffix=".png", weight_mod=None):

        self.batch_size = batch_size
        self.X_dir = X_dir
        self.y_dir = y_dir
        self.palette = palette
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.suffix = suffix
        self.weight_mod = weight_mod
        self.files = np.array(os.listdir(X_dir))
        self.num_classes = len(palette)
        self.n = len(self.files)
        self.cur = 0
        self.order = list(range(self.n))
        np.random.shuffle(self.order)
        self.files_in_batch = None

    # Helper functions
    def _getClassMask(self, rgb, image):
        """
        Takes an rgb tuple and returns a binary mask of size
        im.shape[0] x im.shape[1] indicated where each color
        is present.

        Input:
            rgb - tuple of (r, g, b)

            image - segmentation ground truth image

        Output:
            mask - binary mask
        """
        # Colour mask
        if len(rgb) == 3:
            return np.all(image == rgb, axis=-1)
            #r, g, b = rgb
            #r_mask = im[:,:, 0] == r
            #g_mask = im[:,:, 1] == g
            #b_mask = im[:,:, 2] == b
            #mask = r_mask & g_mask & b_mask
            #return mask
        # 8-bit mask
        return image[:, :] == rgb

    def _calculateWeights(self, y_train):
        """
        Calculates the balanced weights of all the classes
        in the batch.

        Input:
            y_train - (dim, dim,num_classes) ground truth

        Ouput:
            weights - a list of the weights for each class
        """
        class_counts = []
        # loop through each class
        for i in range(self.num_classes):
            batch_count = 0
            # Sum up each class count in each batch image
            for b in range(y_train.shape[0]):
                batch_count += np.sum(y_train[b][:,:,i])
            class_counts.append(batch_count)

        # create Counts
        y = []
        present_classes = []
        absent_classes = []
        for i in range(self.num_classes):
            # Adjusts for absence
            if class_counts[i] == 0:
                absent_classes.append(i)
                continue
            else:
                present_classes.append(i)
                y.extend([i]*int(class_counts[i]))
        # Calculate weights
        weights = compute_class_weight("balanced", present_classes, y)
        for c in absent_classes:
            weights = np.insert(weights, c, 0)

        # Modify weight for a particular class
        if self.weight_mod:
            for key in self.weight_mod.keys():
                weights[key] *= self.weight_mod[key]

        return weights

    def _createBatches(self, positions):
        """
        Creates X_train and y_train batches from the given
        positions i.e. files in the directory

        Input:
            positions - list of integers representing files

        Output:
            X_train, y_train - batches
        """
        # Store images in batch
        X_batch = []
        y_batch = []

        # Save file names for batch
        self.files_in_batch = self.files[positions]

        # Loop through current batch
        for pos in positions:
            # Get image name
            fname = self.files[pos][:-4]

            # load X-image
            im = io.imread(os.path.join(self.X_dir, fname + self.suffix))[:,:,0:3]    # drop alpha
            im = resize(im, (self.x_dim, self.x_dim))
            X_batch.append(im)

            # Load y-image
            im = io.imread(os.path.join(self.y_dir, fname + ".png"))[:,:,0:3]    # drop alpha
            im = resize(im, (self.y_dim, self.y_dim))
            # Convert to 3D ground truth
            y = np.zeros((im.shape[0], im.shape[1], self.num_classes), dtype=np.float32)
            # Loop through colors in palette and assign to new array
            for i in range(self.num_classes):
                rgb = self.palette[i]
                mask = self._getClassMask(rgb, im)
                y[mask, i] = 1.

            y_batch.append(y)

        # Combine images into batches and normalise
        X_train = np.stack(X_batch, axis=0).astype(np.float32)
        y_train = np.stack(y_batch, axis=0)

        # Preprocess X_train
        X_train /= 255.
        X_train -= 0.5
        X_train *= 2.

        # Calculate sample weights
        weights = self._calculateWeights(y_train)

        # Take weight for each correct position
        sample_weights = np.take(weights, np.argmax(y_train, axis=-1))

        # Reshape to suit keras
        sample_weights = sample_weights.reshape(y_train.shape[0], self.y_dim*self.y_dim)
        y_train = y_train.reshape(y_train.shape[0],
                                  self.y_dim*self.y_dim,
                                  self.num_classes)

        return X_train, y_train, sample_weights

    def __next__(self):
        """
        Returns a batch when the `next()` function is called on it.
        """
        while True:

            # Most batches will be equal to batch_size
            if self.cur < (self.n - self.batch_size):
                # Get positions of files in batch
                positions = self.order[self.cur:self.cur + self.batch_size]

                self.cur += self.batch_size

                # create Batches
                X_train, y_train, sample_weights = self._createBatches(positions)

                return (X_train, y_train, sample_weights)

            # Final batch is smaller than batch_size
            else:
                # Have sufficient data in each batch is good on multi-GPUs
                np.random.shuffle(self.order)
                self.cur = 0
                continue



def predict_image(model, image):
    """
    Simplifies image prediction for segmentation models. Automatically
    reshapes output so it can be visualised.

    Input:

        model - ResNet training model where model.layers[-1] is a reshape
                layer.

        image - rgb image of shape (dim, dim, 3) where dim == model.input_shape
                image should already be pre-processed using load_image() function.

    Output:

        preds - probability heatmap of shape (dim, dim, num_classes)

        class_img - argmax of preds of shape (dim, dim, 1)
    """

    if len(image.shape) < 4:
        # Add new axis to conform to model input
        image = image[np.newaxis, ::]

    # Prediction
    preds = model.predict(image)[0].reshape(
                                    image.shape[0],
                                    image.shape[0],
                                    model.layers[-1].output_shape[-1])
    # class_img
    class_img = np.argmax(preds, axis=-1)

    return preds, class_img


def get_color_map(colors):
    """
    Returns a matplotlib color map of the list of RGB values

    Input:

        colors - a list of RGB colors

    Output:

        cmap -  a matplotlib color map object
    """
    # Normalise RGBs
    norm_colors = []
    for color in colors:
        norm_colors.append([val / 255. for val in color])
    # create color map
    cmap = cols.ListedColormap(norm_colors)

    return cmap


def apply_color_map(colors, image):
    """
    Applies the color specified by colors to the input image.

    Input:

        colors - list of colors in color map

        image - image to apply color map to with shape (n, n)

    Output:

        color_image - image with shape (n, n, 3)

    """
    cmap = get_color_map(colors)
    norm = Normalize(vmin=0, vmax=len(colors))
    color_image = cmap(norm(image))[:, :, 0:3]  # drop alpha
    return color_image


def load_image(fname, pre=True):
    """
    Loads an image, with optional resize and pre-processing
    for ResNet50.

    Input:

        fname - path + name of file to load

        pre - whether to pre-process image

    Output:

        im - image as numpy array
    """
    im = io.imread(fname).astype("float32")[:, :, 0:3]
    if pre:
            im /= 255.
            im -= 0.5
            im *= 2.
    return im


def set_weights_for_training(model, fine_tune, layer_num=[81, 174]):
    """
    Takes a model and a training state i.e. fine_tune = True
    and sets weights accordingly. Fine-tuning unlocks
    from layer 81 - res4a_branch2a


    Input:

        model - ResNet_UNet model by default, can be any model

        fine_tune - bool to signify training state

        layer_num - layer to lock/unlock from. default is
                    173 add_16, where 174 is up_sampling2d_1

    Output:

        None
    """
    if not fine_tune:
        print("[INFO] base model...")
        # ResNet layers
        for layer in model.layers[0:layer_num[1]]:
            # Opens up mean and variance for training
            if hasattr(layer, 'moving_mean') and hasattr(layer, 'moving_variance'):
                layer.trainable = True
                K.eval(K.update(layer.moving_mean, K.zeros_like(layer.moving_mean)))
                K.eval(K.update(layer.moving_variance, K.zeros_like(layer.moving_variance)))
            else:
                layer.trainable = False

        # UNet layers
        for layer in model.layers[layer_num[1]::]:
            layer.trainable = True
    else:
        print("[INFO] fine tuning model...")
        # ResNet layers
        for layer in model.layers[layer_num[0]:layer_num[1]]:
            layer.trainable = True
            # Opens up mean and variance for training
            if hasattr(layer, 'moving_mean') and hasattr(layer, 'moving_variance'):
                K.eval(K.update(layer.moving_mean, K.zeros_like(layer.moving_mean)))
                K.eval(K.update(layer.moving_variance, K.zeros_like(layer.moving_variance)))
        # UNet layers
        for layer in model.layers[layer_num[1]::]:
            layer.trainable = True


def get_number_of_images(dir):
    """
    Returns number of files in given directory

    Input:

        dir - full path of directory

    Output:

        number of files in directory
    """
    return len([name for name in os.listdir(dir) if os.path.isfile(os.path.join(dir, name))])


def load_multigpu_checkpoint_weights(model, h5py_file):
    """
    Loads the weights of a weight checkpoint from a multi-gpu
    keras model.

    Input:

        model - keras model to load weights into

        h5py_file - path to the h5py weights file

    Output:
        None
    """

    print("Setting weights...")
    with h5py.File(h5py_file, "r") as file:

        # Get model subset in file - other layers are empty
        weight_file = file["model_1"]

        for layer in model.layers:

            try:
                layer_weights = weight_file[layer.name]

            except:
                # No weights saved for layer
                continue

            try:
                weights = []
                # Extract weights
                for term in layer_weights:
                    if isinstance(layer_weights[term], h5py.Dataset):
                        # Convert weights to numpy array and prepend to list
                        weights.insert(0, np.array(layer_weights[term]))

                # Load weights to model
                layer.set_weights(weights)

            except Exception as e:
                print("Error: Could not load weights for layer:", layer.name, file=stderr)


def create_prob_map_from_mask(filename, palette):
    """

    Creates a probability map with the input mask

    Input:

        filename - path to mask file

        pallette - color palette of mask

    Output:
        prob_map - numpy array of size image.h x image.w x num_classes
    """

    # Helper functions
    def _get_class_mask(rgb, im):
        """
        Takes an rgb tuple and returns a binary mask of size
        im.shape[0] x im.shape[1] indicated where each color
        is present.

        Input:
            rgb - tuple of (r, g, b)

            im - segmentation ground truth image

        Output:
            mask - binary mask
        """
        # Colour mask
        if len(rgb) == 3:
            r, g, b = rgb
            r_mask = im[:, :, 0] == r
            g_mask = im[:, :, 1] == g
            b_mask = im[:, :, 2] == b
            mask = r_mask & g_mask & b_mask
            return mask
        # 8-bit mask
        return im[:, :] == rgb
    # -------------------------#

    num_classes = len(palette)
    # Load y-image
    im = io.imread(filename)
    # Convert to 3D ground truth
    prob_map = np.zeros((im.shape[0], im.shape[1], num_classes), dtype=np.float32)
    # Loop through colors in palette and assign to new array
    for i in range(num_classes):
        rgb = palette[i]
        mask = _get_class_mask(rgb, im)
        prob_map[mask, i] = 1.
    return prob_map


# def generate_ROC_AUC(true_map, prob_map, color_dict, colors):
#     """
#     Generates ROC curves and AUC values for all class in image, as well
#     as keeps raw data for later use.
#
#     Input:
#         true_map - map of true values, generated from mask using
#                     create_prob_map_from_mask()
#         prob map - 3 dimensional prob_map created from model.predict()
#
#         color_dict - color dictionary containing names and colors
#
#         colors - list of colors
#     Output:
#
#         ROC - dictionary:
#                 "AUC" - scalar AUC value
#                 "TPR" - array of trp for different thresholds
#                 "FPR" - array of fpr for different thresholds
#                 "raw_data" - type of (true, pred) where each are arrays
#
#     ! NEED TO INCLUDE SAMPLE WEIGHTS !
#
#     """
#     # Create ROC curves for all tissue types
#     ROC = {}
#     for tissue_class in color_dict.keys():
#         # Get class index
#         class_idx = colors.index(color_dict[tissue_class])
#
#         true = np.ravel(true_map[:, :, class_idx])
#         pred = np.ravel(prob_map[:, :, class_idx])
#
#         # Get FPR and TPR
#         fpr, tpr, thresholds = roc_curve(true, pred)
#         roc_auc = auc(fpr, tpr)
#         if np.isnan(roc_auc):
#             # class not present
#             continue
#         # Update values
#         ROC[tissue_class] = {"AUC": roc_auc, "TPR": tpr, "FPR": fpr, "raw_data": (true, pred)}
#
#     return ROC

def calculate_tile_size(image_shape, lower=50, upper=150):
    """
    Calculates a tile size with optimal overlap

    Input:

        image - original histo image (large size)

        lower - lowerbound threshold for overlap

        upper - upper-bound threshold for overlap

    Output:

        dim - dimension of tile

        threshold - calculated overlap for tile and input image
    """

    def smallest_non_zero(values, threshold=10):
        for tile, overlap in values:
            if overlap > threshold:
                return tile, overlap

    dims = [x*(2**5) for x in range(6, 45)]
    w = image_shape[1]
    h = image_shape[0]
    thresholds = {}
    for d in dims:
        w_steps = w // d
        if w_steps == 0:
            continue
        w_overlap = (d - (w % d)) // w_steps
        h_steps = h // d
        if h_steps == 0:
            continue
        h_overlap = (d - (h % d)) // h_steps
        # Threshold is half the minimum overlap
        thresholds[d] = min(w_overlap, h_overlap) // 2
    # Loop through pairs and take first that satisfies
    sorted_thresholds = sorted(thresholds.items(), key=lambda x: x[1])
    for d, t in sorted_thresholds:
        if lower < t < upper:
                return w, h, d, t  # dim, threshold
    # Else - get largest overlap value
    print("[INFO] - title overlap threshold not met. Defaulting to Smallest non-zero overlap")
    return w, h, smallest_non_zero(sorted_thresholds, threshold=10)


def whole_image_predict(files, model, output_directory, colors, compare=True, pad_val=50, prob_map=False):
    """
    Generates a segmentation mask for each of the images in files
    and saves them in the output directory.

    Input:

        files - list of files including their full paths

        model - model to use to predict.
                >> Must be of type K.function NOT keras.models.Model
                >> Must have output shape (1, dim, dim, 12)

        output_directory -  path to directory to save files, include / on end

        colors - list of RGB values to apply to segmentation

    Output:

        None

    """
    image_num = 1
    for file in files:
        # Get name
        name = file.split("/")[-1].split(".")[0]
        print("Whole Image Segmentation:", name, "Num:", image_num, "of", len(files))
        image_num += 1

        try:
            # Load image
            histo = load_image(file, pre=True)

            # Pad image with minimum threshold
            # https://stackoverflow.com/questions/35751306/python-how-to-pad-numpy-array-with-zeros
            histo = np.pad(histo, [(pad_val, pad_val),(pad_val, pad_val), (0, 0)], mode="constant", constant_values=0.99)

            # Create canvas to add predictions
            if prob_map:
                canvas = np.zeros((histo.shape[0],histo.shape[1], len(colors)))
            else:
                canvas = np.zeros_like(histo)

            # Tile info
            w, h, dim, threshold = calculate_tile_size(histo.shape, lower=50, upper=100)

            print("Tile size:", dim)
            print("Tile threshold:", threshold)

        except Exception as e:
            print("Failed to process:", name, e, file=stderr)
            continue

        # Compute number of vertical and horizontal steps
        w_steps = w // dim
        w_overlap = (dim - (w % dim)) // w_steps
        h_steps = h // dim
        h_overlap = (dim - (h % dim)) // h_steps
        # starting positions
        w_x, w_y = 0, dim
        h_x, h_y = 0, dim


        # Loop through all tiles and predict
        step = 1
        for i in range(h_steps + 1):

            for j in range(w_steps + 1):

                print("Processing tile", step, "of", (h_steps + 1) * (w_steps + 1), )
                step += 1

                # Grab a tile
                tile = histo[h_x:h_y, w_x:w_y, :][np.newaxis, ::]

                # Check and correct shape
                orig_shape = tile.shape


                if prob_map:

                    if tile.shape != (dim, dim, len(colors)):
                        tile = resize(tile[0], dsize=(dim, dim))[np.newaxis, ::]
                    # Predict
                    probs = model([tile])[0]

                    # Add prediction to canvas
                    canvas[h_x + threshold: h_y - threshold,
                                w_x + threshold: w_y - threshold, :] = probs[0][threshold:-threshold,
                                                                                threshold:-threshold, :]
                else:

                    if tile.shape != (dim, dim, 3):
                        tile = resize(tile[0], dsize=(dim, dim))[np.newaxis, ::]
                    # Predict
                    probs = model([tile])[0]

                    class_pred = np.argmax(probs[0], axis=-1)

                    segmentation = apply_color_map(colors, class_pred)

                    # Add prediction to canvas
                    canvas[h_x + threshold: h_y - threshold,
                            w_x + threshold: w_y - threshold, :] = segmentation[threshold:-threshold,
                                                                               threshold:-threshold, :]
                # Update column positions
                w_x += dim - w_overlap
                w_y += dim - w_overlap

            # Update row positions
            h_x += dim - h_overlap
            h_y += dim - h_overlap
            w_x, w_y = 0, dim

        # Save Segmentation
        fname = output_directory + name + ".png"

        # Crop canvas by removing padding
        canvas = canvas[pad_val:-pad_val, pad_val:-pad_val, :]

        if compare:
            # Load in ground truth
            file = "/".join(file.split("/")[0:-2]) + "/Masks/" + name + ".png"
            mask = io.imread(file)
            fig, axes = plt.subplots(1, 2, figsize=(12, 8), frameon=False)
            axes[0].imshow(mask)
            axes[0].set_title("Ground Truth")
            axes[0].set_axis_off()
            axes[1].imshow(canvas)
            axes[1].set_title("Predicted")
            axes[1].set_axis_off()
            plt.tight_layout()
            plt.savefig(fname, dpi=300)
            plt.close()
            # Wipe canvas from memory
            del canvas

        elif prob_map:
            file = fname.split(".")[0] + ".npy"
            np.save(file, canvas)
            del canvas

        else:
            # Scale values to RGB
            #canvas *= 255.
            # Convert canvas to BGR color space for cv2.imwrite
            #canvas = cvtColor(canvas, COLOR_RGB2BGR)
            #print("saving...", fname)
            #imwrite(fname, canvas)
            io.imsave(fname, canvas)

            # Wipe canvas from memory
            del canvas

class Validation(TensorBoard):
    """
    A custom callback to perform validation at the
    end of each epoch. Also writes useful class
    metrics to tensorboard logs.
    """

    def __init__(self, generator, steps, classes, run_name,
                 color_list, WSI=False, model_to_save=None,
                 weight_path=None, interval=5,
                 **kwargs):
        """

        Initialises the callback

        Input:

            generator -  validation generator of type SegmentationGen()

            steps - number of steps in validation e.g. n // batch_size

            classes - an ordered list of classes ie. [ "EPI", "GLD" etc ]

            run_name - str of the unique run identifier

            color_list - list of RGB values for applying colors to predictions

            model_to_save - model to save weights as checkpoint (useful for multi-GPU models)
                            where the model passed should be the non-parallel model.

            log_dir - Tensorboard log directory
        """
        super().__init__(**kwargs)
        self.validation_data = generator
        self.validation_steps = steps
        self.classes = np.asarray(classes)
        self.cms = []
        self.run_name = run_name
        self.color_list = color_list
        self.WSI = WSI
        self.model_to_save = model_to_save
        self.weight_path = weight_path
        self.interval = interval

    # Helper functions ------------------------------------------------ #

    def write_confusion_matrix_to_buffer(self, matrix, classes):
        """
         Writes a confusion matrix to the tensorboard session

         Input:
            matrix - numpy confusion matrix

            classes - ordered list of classes

        Output:
            buffer - buffer where plot is written to
        """
        # Compute row sums for Recall
        row_sums = matrix.sum(axis=1)
        matrix = np.round(matrix / row_sums[:, np.newaxis], 3)

        # Import colors
        color = [255, 118, 25]
        orange = [c / 255. for c in color]
        white_orange = LinearSegmentedColormap.from_list("", ["white", orange])

        fig = plt.figure(figsize=(12, 14))
        ax = fig.add_subplot(111)
        cax = ax.matshow(matrix, interpolation='nearest', cmap=white_orange)
        fig.colorbar(cax)

        ax.set_xticklabels([''] + classes, fontsize=8)
        ax.set_yticklabels([''] + classes, fontsize=8)

        # Get ticks to show properly
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(1))

        ax.set_title("Recall")
        ax.set_ylabel("Ground Truth")
        ax.set_xlabel("Predicted")

        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j - 0.1, i, str(matrix[i, j]), fontsize=8)

        buffer = IO.BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        plt.close(fig)

        return buffer

    def write_current_predict(self, mask, prediction, image_num, epoch):
        """
        Write mask and prediction to Tensorboard
        """
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(apply_color_map(self.color_list, mask))
        axes[0].set_title("Ground Truth")
        plt.axis('off')
        axes[1].imshow(apply_color_map(self.color_list, prediction))
        axes[1].set_title("Predict")
        plt.axis('off')

        # save to buffer
        plot_buffer = IO.BytesIO()
        plt.savefig(plot_buffer, format="png")
        plot_buffer.seek(0)
        plt.close(fig)

        # Create an Image object
        img_sum = tf.Summary.Image(encoded_image_string=plot_buffer.getvalue(),
                                   height=mask.shape[0],
                                   width=mask.shape[1])
        # Create a Summary value
        im_summary = tf.Summary.Value(image=img_sum, tag="Segmentation/Segmentation_E_" + str(epoch))

        summary = tf.Summary(value=[im_summary])
        self.writer.add_summary(summary, str(epoch))

    def write_current_cm(self, epoch):
        """
        Write confusion matrix to Tensorboard
        """
        # Get the matrix
        matrix = self.cms[-1]

        # Prepare the plot
        plot_buffer = self.write_confusion_matrix_to_buffer(matrix, list(self.classes))

        # Create an Image object
        img_sum = tf.Summary.Image(encoded_image_string=plot_buffer.getvalue(),
                                   height=800,
                                   width=800)
        # Create a Summary value
        im_summary = tf.Summary.Value(image=img_sum, tag="Confusion_Matrix")

        summary = tf.Summary(value=[im_summary])

        self.writer.add_summary(summary, str(epoch))

    def compute_stats(self, epoch_cm, logs):
        """
        Computes the precision and recall for
        the epoch, and prints them.
        """
        # End of epoch - compute stats
        val_weighted_acc = []
        weighted_F1 = []
        print("Validation")
        for i in range(12):

            # Precision - TP / (TP + FP)
            try:
                precision = np.round(epoch_cm[i, i] / np.sum(epoch_cm[:, i]), 5)
            except ZeroDivisionError:
                precision = 0
            # Update Logs
            name = self.classes[i] + "_P"
            print(self.classes[i], "P:", precision, end=" ")
            logs[name] = precision

            # Recall - TP / (TP + FN)
            try:
                recall = np.round(epoch_cm[i, i] / np.sum(epoch_cm[i, :]), 5)
            except ZeroDivisionError:
                recall = 0
            # Update Logs
            name = self.classes[i] + "_R"
            print("R:", recall)
            logs[name] = recall
            val_weighted_acc.append(recall)

            # F1 - score
            F1 = 2 * ((precision*recall) / (precision + recall))
            weighted_F1.append(F1)

        # update weighed average
        #epoch_val_weighted_acc = np.nanmean(val_weighted_acc)
        #epoch_weighted_F1 = np.nanmean(weighted_F1)
        #print("Weighted Val Acc:", epoch_val_weighted_acc)
        #print("Weighted F1:", epoch_weighted_F1)

        # Add to logs
        #logs["val_weighted_acc"] = epoch_val_weighted_acc
        #logs["val_weights_F1"] = epoch_weighted_F1



    def sample_predict(self, val_model, epoch):
        """
        Predict segmentation mask from the next
        batch of images.
        """
        # Grab next batch
        X, y_true, _ = next(self.validation_data)

        # Make prediction with model
        learning_phase = 1
        y_pred = val_model([X, learning_phase])[0]

        # Reshape prediction
        y_pred = y_pred.reshape(
            y_pred.shape[0],  # batch_size
            X[0].shape[0],  # dim
            X[0].shape[1],  # dim
            y_pred.shape[-1]  # number of classes
        )

        # Reshape ground-truth
        y_true = y_true.reshape(
            y_pred.shape[0],  # batch_size
            X[0].shape[0],  # dim
            X[0].shape[1],  # dim
            y_pred.shape[-1]
        )

        # Loop through each image in batch
        for i in range(y_pred.shape[0]):
            # Create mask and prediction mask
            mask, pred = np.argmax(y_true[i], axis=-1), np.argmax(y_pred[i], axis=-1)
            # Save image i in batch
            self.write_current_predict(mask, pred, i, epoch)

    def validate_epoch(self, val_model, epoch_cm):
        """
        Computes the batch validation confusion matrix
        and then updates the epoch confusion matrix.
        """
        # Loop through validation set
        for n in range(self.validation_steps):

            # Grab next batch
            X, y_true, _ = next(self.validation_data)

            # Make prediction with model
            y_pred = val_model([X])[0]

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

    # End of helper functions ----------------------------------------- #

    # Overloaded parent methods
    def on_train_end(self, logs={}):
        # 1. Print final confusion matrix
        print()
        print("Confusion Matrix (final epoch):")
        for i in range(12):
            for j in range(12):
                print(self.cms[-1][i, j], end=", ")
            print()

        # # Predict whole images
        # if self.WSI:
        #
        #     # Create prediction model
        #     model_in = self.model.layers[0].get_input_at(0)
        #     model_out = self.model.layers[-2].output
        #     model = K.function(inputs=[model_in, K.learning_phase()], outputs=[model_out])
        #
        #     files = [
        #         "./WSI_test/images/histo_demo_1.tif",
        #         "./WSI_test/images/histo_demo_2.tif",
        #         "./WSI_test/images/histo_demo_3.tif"
        #     ]
        #
        #     whole_image_predict(files, model, "./WSI_test/segmentations/", self.color_list)

    def on_epoch_end(self, epoch, logs={}):

        # Create new validation model
        val_model = K.function(inputs=[self.model.input], outputs=[self.model.output], )

        # Confusion Matrix
        epoch_cm = np.zeros((len(self.classes), len(self.classes)))

        # Loop through validation data and update epoch_cm
        self.validate_epoch(val_model, epoch_cm)

        # Compute stats
        self.compute_stats(epoch_cm, logs)
        # Save epoch CM
        self.cms.append(epoch_cm)

        # Predict segmentation mask for a sample of images
        self.sample_predict(val_model, epoch)

        # Write confusion matrix to tensorboard
        self.write_current_cm(epoch)

        # Clear memory of val model
        del val_model

        # Call parent class method - Tensorboard writer
        super().on_epoch_end(epoch, logs)

        # Save weights for model checkpoint
        if epoch % self.interval == 0 and epoch != 0:
            if self.model_to_save:
                weight_name = self.weight_path.format(epoch=epoch)
                print("Saving model checkpoint:", weight_name)
                self.model_to_save.save_weights(weight_name)

        return


