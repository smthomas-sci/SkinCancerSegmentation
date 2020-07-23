"""

Temperature scaling on a validation set....

From this...

https://arxiv.org/pdf/1706.04599.pdf



"""

from seg_utils import *
from seg_models import ResNet_UNet, ResNet_UNet_ExtraConv, ResNet_UNet_Dropout

from keras.models import Model
from keras.layers import Softmax, Reshape, Layer
from keras.initializers import Constant
from keras.optimizers import Adam

from numpy.random import seed
from tensorflow import set_random_seed
# Set seed
seed(1)
set_random_seed(2)

num_classes = 12
gpus = 1


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
colors = [color_dict[key] for key in color_dict.keys()]


val_path = "/home/simon/Desktop/10x_290_Over_Aug/"
# Validation
X_val_dir = os.path.join(val_path, "X_val")
y_val_dir = os.path.join(val_path, "y_val")

# Create palette for generators
palette = Palette(colors)

dim = 256
batch_size = 6
val_gen = SegmentationGen(
                batch_size, X_val_dir,
                y_val_dir, palette,
                x_dim=dim, y_dim=dim,
                )


class TemperatureScaling(Layer):
    def __init__(self, T=1, T_is_trainable=True, **kwargs):
        self.T = T
        self.T_is_trainable = T_is_trainable
        super(TemperatureScaling, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='T',
                                      shape=(1,),
                                      initializer=Constant(value=self.T),
                                      trainable=self.T_is_trainable)
        super(TemperatureScaling, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return x / self.kernel

    def compute_output_shape(self, input_shape):
        return input_shape
# -------------------------------------------------------- #

# Import model
K.set_learning_phase(0) #  Since training but not using BN layers, need to force them to use learned pop mean

# Load model without softmax function
model = ResNet_UNet_Dropout(num_classes=num_classes, dropout=0.5, dim=256, final_activation=True)

model.load_weights("/home/simon/Desktop/10x_Experiments_Over_Aug/weights/10x_290_Over_Aug_BS_24_PS_256_C_12_FT_True_E_5_LR_1e-06_WM_F_model_ResNet_UNet_Dropout_less_params_all_32_seed_1_DO_0.5_checkpoint_001.h5")


def generate_temperature_model(model, dim=256, T=1, trainable=True):
    # Add Temperature Scaling
    inputs = model.get_input_at(0)
    x = model.layers[-2].output
    x = TemperatureScaling(T, trainable)(x)
    x = Reshape((dim*dim, 12))(x)
    activation = Softmax(axis=-1)(x)

    return Model(inputs=[inputs], outputs=[activation])


# temp_model = generate_temperature_model(model, dim=256, T=1, trainable=True)
#
# # Load weights
# for layer in temp_model.layers[0:-3]:
#    layer.trainable = False
# temp_model.layers[-3].trainable = True
# temp_model.summary()


temp_model = model
temp_model.trainable = False

# # Compile model for training
# temp_model.compile(
#             optimizer=Adam(lr=0.05),
#             loss="categorical_crossentropy",
#             sample_weight_mode="temporal",
#             metrics=["accuracy"],
#             weighted_metrics=["accuracy"]
#             )
#
#
#
# # Train
# history = temp_model.fit_generator(
#                         epochs=1,
#                         generator=val_gen,
#                         steps_per_epoch=val_gen.n // val_gen.batch_size,
#                         )


# ------------ CALIBRATION PLOT




def tally_predictions(model, generator):
    """
    Tallys the predictions into bins within the range(0, 1.1, 0.1).

    Input:
        model - the model to use (final layer needs to be a Reshape)
        generator - the data generator to use (validation set)

    Output:
        all_preds - a tally of all the predictions (dict)
        correct_preds - a tally of all the correction predictions (dict)
    """
    # Set all counts to zero
    all_preds = np.zeros(10)
    correct_preds = np.zeros(10)
    pred_means = np.zeros(10)

    # Loop through whole data set
    n = 0
    for batch in range(generator.n // generator.batch_size):

        (X_train, y_true, sample_weights) = next(val_gen)

        y_pred = model.predict_on_batch(X_train)

        # Get the proportion that are correct
        all_max_values = np.max(y_pred, axis=-1)
        mask = np.equal(np.argmax(y_true, axis=-1), np.argmax(y_pred, axis=-1))
        correct_max_values = np.max(y_pred, axis=-1)[mask]

        correct_hist, _ = np.histogram(correct_max_values, bins=np.arange(0, 1.1, 0.1))
        all_hist, _ = np.histogram(all_max_values, bins=np.arange(0, 1.1, 0.1))

        all_preds += all_hist
        correct_preds += correct_hist

        # Get the mean prediction value for max predictions for each bin
        bins = np.arange(0, 1.1, 0.1)
        digitized = np.digitize(all_max_values, bins)
        bin_means = [all_max_values[digitized == i].mean() for i in range(1, len(bins))]

        pred_means += np.asarray(bin_means)
        n += 1

    proportion_correct = correct_hist / all_hist
    all_bin_means = pred_means / n

    return all_bin_means, proportion_correct, correct_hist, all_hist

def calculate_ECE(bin_means, proportion_correct, all_hist):
    """
    Calculates Expected Calibration Error (Naeini et al. 2015). This is simply
    the sum of the weighted average of each bin's accuracy / confidence difference.

    Input:
        bin_means - list means for the 10 bins
        proportion_correct - list of correct proportions for the 10 bins
        all_hist - the number of samples in each bin

    Output:
        ECE - Expected Calibration Error
    """
    # Check for nans
    bin_means[np.isnan(bin_means)] = 0
    proportion_correct[np.isnan(proportion_correct)] = 0
    ece = 0
    # Get total predictions
    n = sum(all_hist)
    # Loop through each bin
    for bin in range(len(bin_means)):
        w = all_hist[bin] / n
        ece += w * abs(bin_means[bin] - proportion_correct[bin])
    return ece


def plot_accuracy_confidence(mean_p_of_all_ps, proportion_correct, ece):
    """
    Plots it
    """
    import matplotlib.patches as mpatches
    purple = (127 / 255., 63 / 255., 191 / 255.)

    plt.axes(axisbelow=True)
    plt.grid(ls="--")  # np.arange(0.05, 1.05, 0.1)

    # Mean argmax p for each bin
    plt.bar(np.arange(0, 1, 0.1), mean_p_of_all_ps,
            width=0.1, align="edge", color="red",
            alpha=0.5, edgecolor="black",
            label="Mean Output Confidence")

    # Proportion Correct
    plt.bar(np.arange(0, 1, 0.1), proportion_correct,
            width=0.1, align="edge", color="blue",
            alpha=0.5, edgecolor="black",
            label="Proportion Correct")

    plt.plot(range(1), range(1), color=purple, label="Concordance")
    # Ideal output
    plt.plot([0, 1], [0, 1], color="lightgray", ls="--", lw=2)

    plt.title("Uncalibrated - ECE = {0:.3f}".format(ece))
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.margins(x=0)
    plt.margins(y=0)

    # Legend
    red_patch = mpatches.Patch(color='red', label='Mean Confidence', alpha=0.5)
    blue_patch = mpatches.Patch(color='blue', label='Accuracy', alpha=0.5)
    purple_patch = mpatches.Patch(color=purple, label='Acc./Conf. Concordance')
    legend = plt.legend(handles=[red_patch, blue_patch, purple_patch],
               frameon=True, framealpha=1,
               fancybox=None)
    legend.get_frame().set_linewidth(0.0)
    plt.savefig("/home/simon/Desktop/Validation_Uncalibrated.png", dpi=300)
    #plt.show()
    plt.close()



# Run
bin_means, proportion_correct, correct_hist, all_hist = tally_predictions(temp_model, val_gen)


#temp_model.save_weights("/home/simon/Desktop/10x_Experiments_Over_Aug/weights/10x_temperature_scaled.h5")


# T = temp_model.layers[-3].get_weights()[0][0]
# print("T:", T)

#np.set_printoptions(precision=3)
print("Bin Means:", bin_means)
print("Porpriont:", proportion_correct)
print(correct_hist)
print(all_hist)

ece = calculate_ECE(bin_means, proportion_correct, all_hist)

plot_accuracy_confidence(bin_means, proportion_correct, ece)

# Get T parameter





