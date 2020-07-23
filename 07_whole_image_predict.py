"""

Implementation of whole image prediction with variable overlap.


"""

from seg_utils import *
from seg_models import *

from numpy.random import seed
from tensorflow import set_random_seed
# Set seed
seed(1)
set_random_seed(2)

num_classes = 12
gpus = 1

# Import model
model = ResNet_UNet_Dropout(num_classes=num_classes, dropout=0.5, final_activation=False)


model = generate_temperature_model(model, dim=256, T=1, trainable=False)

#model.load_weights("/home/simon/Desktop/10x_Experiments_Over_Aug/weights/10x_290_Over_Aug_BS_24_PS_256_C_12_FT_True_E_5_LR_1e-06_WM_F_model_ResNet_UNet_Dropout_less_params_all_32_seed_1_DO_0.5_checkpoint_001.h5")
#model.load_weights("/home/simon/Desktop/10x_Experiments_Over_Aug/weights/10x_temperature_scaled.h5")
model.load_weights("/media/simon/UNTITLED/Segmentation_Experiments/Segmentation/10x_Experiments_Over_Aug/weights/10x_temperature_scaled.h5")

model.summary()

# Create Keras function instead of model - helps with Learning Phase errors
model_in = model.layers[0].get_input_at(0)
model_out = model.layers[-3].output # Get output with softmax include in temp_scaling layer (ad-hoc fix)
model = K.function(inputs=[model_in], outputs=[model_out])

K.set_learning_phase(0)

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

base_dir = "/home/simon/Desktop/10x_Slides/"

fnames = os.listdir(base_dir)

# with open("/home/simon/Desktop/10x_Experiments_Over_Aug/validation_files.txt", "r") as fh:
#     fnames = [line.strip() + ".tif" for line in fh.readlines()]

files = [ os.path.join(base_dir, file) for file in fnames if ( "BCC" in file or "IEC" in file or "SCC" in file or "KA" in file)]

output_directory = "/home/simon/Desktop/SegmentationRevision/EXTERNAL_PROB_MAPS/"

whole_image_predict(files, model, output_directory, colors, compare=False, pad_val=100, prob_map=True)


