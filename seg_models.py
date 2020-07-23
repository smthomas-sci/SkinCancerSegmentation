"""

A collection of Encoder-Decoder networks, namely U-net and
U-net like decoders combined with regular CNNs e.g. VGG, ResNEt etc.)
The model architectures are suitbale for training Semantic Segmentation only.
You will need to save the trained model and rebuilt so it can take any input
size. 

Author: Simon Thomas
Email: simon.thomas@uq.edu.au

Start Date: 26/10/18
Last Update: 04/02/19

"""

# Required for custom layer / model
from keras.layers import Softmax, Reshape, Layer
from keras.initializers import Constant
from keras.models import Model


def VGG_UNet(dim, num_classes, channels=3):
    """
    Returns a VGG16 Nework with a U-Net
    like upsampling stage. Inlcudes 3 skip connections
    from previous VGG layers.

    Input:

        dim - the size of the input image. Note that is should be
              a square of 2 so that downsampling and upsampling
              always match. ie. 128 -> 64 -> 32 -> 64 -> 128

        num_classes - the number of classes in the whole problem. Used to
                      determine the dimension of output map. i.e. model.predict()
                      returns array that can be reshaped to (dim, dim,
                      num_classes).

        channels - number of channels in input image. Defaut of 3 for RGB

    Output:

        model - an uncompied keras model. Check output shape before use.

    """

    import keras.backend as K
    from keras.models import Model
    from keras.layers import Input
    from keras.layers import Conv2D, MaxPooling2D
    from keras.layers import UpSampling2D, Reshape, concatenate
    from keras.applications.vgg16 import VGG16

    # Import a headless VGG16 - extract weighs and then delete
    vgg16 = VGG16(include_top=False)

    weights = []
    for layer in vgg16.layers[1::]:
        weights.append(layer.get_weights())

    del vgg16
    K.clear_session()

    # Build VGG-Unet using functional API

    input_image = Input(shape=(dim, dim, channels))

    # Conv Block 1
    block1_conv1 = Conv2D(64, (3, 3), activation='relu',
                          padding='same',name='block1_conv1')(input_image)
    block1_conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',
                          name='block1_conv2')(block1_conv1)
    block1_pool = MaxPooling2D((2, 2), strides=(2, 2),
                               name="block1_pool")(block1_conv2)

    # Conv Block 2
    block2_conv1 = Conv2D(128, (3, 3), activation='relu', padding='same',
                          name='block2_conv1')(block1_pool)
    block2_conv2 = Conv2D(128, (3, 3), activation='relu', padding='same',
                          name='block2_conv2')(block2_conv1)
    block2_pool = MaxPooling2D((2, 2), strides=(2, 2),
                               name="block2_pool")(block2_conv2)

    # Conv Block 3
    block3_conv1 = Conv2D(256, (3, 3), activation='relu', padding='same',
                          name='block3_conv1')(block2_pool)
    block3_conv2 = Conv2D(256, (3, 3), activation='relu', padding='same',
                          name='block3_conv2')(block3_conv1)
    block3_conv3 = Conv2D(256, (3, 3),activation='relu',padding='same',
                          name='block3_conv3')(block3_conv2)
    block3_pool = MaxPooling2D((2, 2), strides=(2, 2),
                               name="block3_pool")(block3_conv3)

    # Conv Block 4
    block4_conv1 = Conv2D(512, (3, 3), activation='relu', padding='same',
                          name='block4_conv1')(block3_pool)
    block4_conv2 = Conv2D(512, (3, 3), activation='relu', padding='same',
                          name='block4_conv2')(block4_conv1)
    block4_conv3 = Conv2D(512, (3, 3), activation='relu', padding='same',
                          name='block4_conv3')(block4_conv2)
    block4_pool = MaxPooling2D((2, 2), strides=(2, 2),
                               name="block4_pool")(block4_conv3)


    # Conv Block 5
    block5_conv1 = Conv2D(512, (3, 3), activation='relu', padding='same',
                          name='block5_conv1')(block4_pool)
    block5_conv2 = Conv2D(512, (3, 3), activation='relu', padding='same',
                          name='block5_conv2')(block5_conv1)
    block5_conv3 = Conv2D(512, (3, 3), activation='relu', padding='same',
                          name='block5_conv3')(block5_conv2)
    block5_pool = MaxPooling2D((2, 2), strides=(2, 2),
                               name="block5_pool")(block5_conv3)


    # Upsampling 1
    up1 = UpSampling2D(size=(2,2))(block5_pool)
    up1_conv = Conv2D(512, 2, activation = 'relu', padding = 'same',
                 kernel_initializer = 'he_normal')(up1)
    merge1 = concatenate([block5_conv3,up1_conv], axis = 3)
    merge1_conv1 = Conv2D(512, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = 'he_normal')(merge1)
    merge1_conv2 = Conv2D(512, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = 'he_normal')(merge1_conv1)

    # Upsampling 2
    up2 = UpSampling2D(size = (2,2))(merge1_conv2)
    up2_conv = Conv2D(256, 2, activation = 'relu', padding = 'same',
                 kernel_initializer = 'he_normal')(up2)
    merge2 = concatenate([block4_conv3,up2_conv], axis = 3)
    merge2_conv1 = Conv2D(256, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = 'he_normal')(merge2)
    merge2_conv2 = Conv2D(256, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = 'he_normal')(merge2_conv1)

    # Upsampling 3
    up3 = UpSampling2D(size = (2,2))(merge2_conv2)
    up3_conv = Conv2D(128, 2, activation = 'relu', padding = 'same',
                 kernel_initializer = 'he_normal')(up3)
    merge3 = concatenate([block3_conv3,up3_conv], axis = 3)
    merge3_conv1 = Conv2D(128, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = 'he_normal')(merge3)
    merge3_conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = 'he_normal')(merge3_conv1)

    # Upsampling 4
    up4 = UpSampling2D(size=(2,2))(merge3_conv2)
    up4_conv = Conv2D(64, 2, activation = 'relu', padding = 'same',
                 kernel_initializer = 'he_normal')(up4)
    merge4 = concatenate([block2_conv2,up4_conv], axis = 3)
    merge4_conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = 'he_normal')(merge4)
    merge4_conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = 'he_normal')(merge4_conv1)

    # Upsamplig 5
    up5 = UpSampling2D(size = (2,2))(merge4_conv2)
    up5_conv = Conv2D(64, 2, activation = 'relu', padding = 'same',
                  kernel_initializer = 'he_normal')(up5)
    merge5 = concatenate([block1_conv2,up5_conv], axis = 3)
    merge5_conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same',
                    kernel_initializer = 'he_normal')(merge5)
    merge5_conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same',
                    kernel_initializer = 'he_normal')(merge5_conv1)

    # Activation and reshape for training
    activation = Conv2D(num_classes, 1, activation = "softmax")(merge5_conv2)
    output = Reshape((dim*dim, num_classes))(activation)

    # Link model
    model = Model(inputs=[input_image], outputs=output)

    # Set VGG weights and lock from training
    for layer, weight in zip(model.layers[1:19], weights):
        # Set
        layer.set_weights(weight)
        # Lock
        layer.trainable = False

    return model

def ResNet_UNet(dim=512, num_classes=6):
    """
    Returns a ResNet50 Nework with a U-Net
    like upsampling stage. Inlcudes 3 skip connections
    from previous VGG layers.

    Input:

        dim - the size of the input image. Note that is should be
              a square of 2 so that downsampling and upsampling
              always match. ie. 128 -> 64 -> 32 -> 64 -> 128
              This is only needed for training.

        num_classes - the number of classes in the whole problem. Used to
                      determine the dimension of output map. i.e. model.predict()
                      returns array that can be reshaped to (dim, dim,
                      num_classes).

    Output:

        model - an uncompiled keras model. Check output shape before use.


    """
    from keras.models import Model
    from keras.layers import Conv2D
    from keras.layers import UpSampling2D, Reshape, concatenate
    from keras.applications.resnet50 import ResNet50

    # Import a headless ResNet50
    resnet = ResNet50(input_shape = (None, None, 3), include_top=False)

    # Attached U-net from second last layer - activation_49
    res_out = resnet.layers[-2].output

    # Standard U-Net upsampling 512 -> 256 -> 128 -> 64

    # Upsampling 1 - 512
    fs = 32
    up1 = UpSampling2D(size=(2,2))(res_out)
    up1_conv = Conv2D(fs, 2, activation = 'relu', padding = 'same',
                 kernel_initializer = 'he_normal')(up1)

    prev_layer = resnet.get_layer("activation_40").output
    merge1 = concatenate([prev_layer,up1_conv], axis = 3)
    merge1_conv1 = Conv2D(fs, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = 'he_normal')(merge1)
    merge1_conv2 = Conv2D(fs, 3, activation = 'relu', padding = 'same',
                   kernel_initializer='he_normal')(merge1_conv1)

    # Upsampling 2 - 256
    fs = 32
    up2 = UpSampling2D(size = (2,2))(merge1_conv2)
    up2_conv = Conv2D(fs, 2, activation = 'relu', padding = 'same',
                 kernel_initializer = 'he_normal')(up2)
    prev_layer = resnet.get_layer("activation_22").output
    merge2 = concatenate([prev_layer,up2_conv], axis = 3)
    merge2_conv1 = Conv2D(fs, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = 'he_normal')(merge2)
    merge2_conv2 = Conv2D(fs, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = 'he_normal')(merge2_conv1)

    # Upsampling 3 & 4 - 128
    fs = 32
    up3 = UpSampling2D(size = (2,2))(merge2_conv2)
    up3_conv1 = Conv2D(fs, 2, activation = 'relu', padding = 'same',
                 kernel_initializer = 'he_normal')(up3)
    up3_conv2 = Conv2D(fs, 2, activation = 'relu', padding = 'same',
                 kernel_initializer = 'he_normal')(up3_conv1)
    up4 = UpSampling2D(size = (2,2))(up3_conv2)
    up4_conv = Conv2D(fs, 2, activation = 'relu', padding = 'same',
                 kernel_initializer = 'he_normal')(up4)
    prev_layer = resnet.get_layer("activation_1").output
    merge3 = concatenate([prev_layer,up4_conv], axis = 3)
    merge3_conv1 = Conv2D(fs, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = 'he_normal')(merge3)
    merge3_conv2 = Conv2D(fs, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = 'he_normal')(merge3_conv1)

    # Upsample 5 - 64
    fs = 32
    up5 = UpSampling2D(size = (2,2))(merge3_conv2)
    up5_conv = Conv2D(fs, 2, activation = 'relu', padding = 'same',
                  kernel_initializer = 'he_normal')(up5)
    merge5_conv1 = Conv2D(fs, 3, activation = 'relu', padding = 'same',
                    kernel_initializer = 'he_normal')(up5_conv)
    merge5_conv2 = Conv2D(fs, 3, activation = 'relu', padding = 'same',
                    kernel_initializer = 'he_normal')(merge5_conv1)

    # Activation and reshape for training
    activation = Conv2D(num_classes, 1, activation = "softmax")(merge5_conv2)
    output = Reshape((dim*dim, num_classes))(activation)

    # Build model
    model = Model(inputs=[resnet.input], outputs=[output])

    return model

def ResNet_UNet_ExtraConv(dim=512, num_classes=6):
    """
    Returns a ResNet50 Nework with a U-Net
    like upsampling stage. Inlcudes 3 skip connections
    from previous VGG layers.

    Input:

        dim - the size of the input image. Note that is should be
              a square of 2 so that downsampling and upsampling
              always match. ie. 128 -> 64 -> 32 -> 64 -> 128
              This is only needed for training.

        num_classes - the number of classes in the whole problem. Used to
                      determine the dimension of output map. i.e. model.predict()
                      returns array that can be reshaped to (dim, dim,
                      num_classes).

    Output:

        model - an uncompiled keras model. Check output shape before use.


    """
    from keras.models import Model
    from keras.layers import Conv2D
    from keras.layers import UpSampling2D, Reshape, concatenate
    from keras.applications.resnet50 import ResNet50

    # Import a headless ResNet50
    resnet = ResNet50(input_shape = (None, None, 3), include_top=False)

    # Attached U-net from second last layer - activation_49
    res_out = resnet.layers[-2].output

    # Standard U-Net upsampling 512 -> 256 -> 128 -> 64

    # Upsampling 1
    up1 = UpSampling2D(size=(2,2))(res_out)
    up1_conv = Conv2D(512, 2, activation = 'relu', padding = 'same',
                 kernel_initializer = 'he_normal')(up1)

    prev_layer = resnet.get_layer("activation_40").output
    merge1 = concatenate([prev_layer,up1_conv], axis = 3)
    merge1_conv1 = Conv2D(512, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = 'he_normal')(merge1)
    merge1_conv2 = Conv2D(512, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = 'he_normal')(merge1_conv1)

    # Upsampling 2
    up2 = UpSampling2D(size = (2,2))(merge1_conv2)
    up2_conv = Conv2D(256, 2, activation = 'relu', padding = 'same',
                 kernel_initializer = 'he_normal')(up2)
    prev_layer = resnet.get_layer("activation_22").output
    merge2 = concatenate([prev_layer,up2_conv], axis = 3)
    merge2_conv1 = Conv2D(256, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = 'he_normal')(merge2)
    merge2_conv2 = Conv2D(256, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = 'he_normal')(merge2_conv1)

    # Upsampling 3 & 4
    up3 = UpSampling2D(size = (2,2))(merge2_conv2)
    up3_conv1 = Conv2D(128, 2, activation = 'relu', padding = 'same',
                 kernel_initializer = 'he_normal')(up3)
    up3_conv2 = Conv2D(128, 2, activation = 'relu', padding = 'same',
                 kernel_initializer = 'he_normal')(up3_conv1)
    up4 = UpSampling2D(size = (2,2))(up3_conv2)
    up4_conv = Conv2D(128, 2, activation = 'relu', padding = 'same',
                 kernel_initializer = 'he_normal')(up4)
    prev_layer = resnet.get_layer("activation_1").output
    merge3 = concatenate([prev_layer,up4_conv], axis = 3)
    merge3_conv1 = Conv2D(128, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = 'he_normal')(merge3)
    merge3_conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = 'he_normal')(merge3_conv1)

    # Upsample 5
    up5 = UpSampling2D(size = (2,2))(merge3_conv2)
    up5_conv = Conv2D(64, 2, activation = 'relu', padding = 'same',
                  kernel_initializer = 'he_normal')(up5)
    merge5_conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same',
                    kernel_initializer = 'he_normal')(up5_conv)
    merge5_conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same',
                    kernel_initializer = 'he_normal')(merge5_conv1)

    # Activation and reshape for training
    activation = Conv2D(num_classes, 1, activation = "softmax")(merge5_conv2)

    # Smoothing
    smooth_conv1 = Conv2D(12, 7, activation='relu', padding='same',
                          kernel_initializer='he_normal')(activation)
    smooth_conv2 = Conv2D(12, 7, activation='relu', padding='same',
                          kernel_initializer='he_normal')(smooth_conv1)

    # Final classification
    classification = Conv2D(num_classes, 1, activation = "softmax")(smooth_conv2)

    output = Reshape((dim*dim, num_classes))(classification)

    # Build model
    model = Model(inputs=[resnet.input], outputs=[output])

    return model


def ResNet_UNet_More_Params(dim=512, num_classes=6):
    """
    Returns a ResNet50 Nework with a U-Net
    like upsampling stage. Inlcudes 3 skip connections
    from previous VGG layers.

    Input:

        dim - the size of the input image. Note that is should be
              a square of 2 so that downsampling and upsampling
              always match. ie. 128 -> 64 -> 32 -> 64 -> 128
              This is only needed for training.

        num_classes - the number of classes in the whole problem. Used to
                      determine the dimension of output map. i.e. model.predict()
                      returns array that can be reshaped to (dim, dim,
                      num_classes).

    Output:

        model - an uncompiled keras model. Check output shape before use.


    """
    from keras.models import Model
    from keras.layers import Conv2D
    from keras.layers import UpSampling2D, Reshape, concatenate
    from keras.applications.resnet50 import ResNet50

    # Import a headless ResNet50
    resnet = ResNet50(input_shape = (None, None, 3), include_top=False)

    # Attached U-net from second last layer - activation_49
    res_out = resnet.layers[-2].output

    # Standard U-Net upsampling 512 -> 256 -> 128 -> 64

    # Upsampling 1
    up1 = UpSampling2D(size=(2,2))(res_out)
    up1_conv = Conv2D(512, 2, activation = 'relu', padding = 'same',
                 kernel_initializer = 'he_normal')(up1)

    prev_layer = resnet.get_layer("activation_40").output
    merge1 = concatenate([prev_layer,up1_conv], axis = 3)
    merge1_conv1 = Conv2D(512, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = 'he_normal')(merge1)
    merge1_conv2 = Conv2D(512, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = 'he_normal')(merge1_conv1)

    # Upsampling 2
    up2 = UpSampling2D(size = (2,2))(merge1_conv2)
    up2_conv = Conv2D(256, 2, activation = 'relu', padding = 'same',
                 kernel_initializer = 'he_normal')(up2)
    prev_layer = resnet.get_layer("activation_22").output
    merge2 = concatenate([prev_layer,up2_conv], axis = 3)
    merge2_conv1 = Conv2D(256, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = 'he_normal')(merge2)
    merge2_conv2 = Conv2D(256, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = 'he_normal')(merge2_conv1)

    # Upsampling 3 & 4
    up3 = UpSampling2D(size = (2,2))(merge2_conv2)
    up3_conv1 = Conv2D(256, 2, activation = 'relu', padding = 'same',
                 kernel_initializer = 'he_normal')(up3)
    up3_conv2 = Conv2D(256, 2, activation = 'relu', padding = 'same',
                 kernel_initializer = 'he_normal')(up3_conv1)
    up4 = UpSampling2D(size = (2,2))(up3_conv2)
    up4_conv = Conv2D(256, 2, activation = 'relu', padding = 'same',
                 kernel_initializer = 'he_normal')(up4)
    prev_layer = resnet.get_layer("activation_1").output
    merge3 = concatenate([prev_layer,up4_conv], axis = 3)
    merge3_conv1 = Conv2D(256, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = 'he_normal')(merge3)
    merge3_conv2 = Conv2D(256, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = 'he_normal')(merge3_conv1)

    # Upsample 5
    up5 = UpSampling2D(size = (2,2))(merge3_conv2)
    up5_conv = Conv2D(256, 2, activation = 'relu', padding = 'same',
                  kernel_initializer = 'he_normal')(up5)
    merge5_conv1 = Conv2D(256, 3, activation = 'relu', padding = 'same',
                    kernel_initializer = 'he_normal')(up5_conv)
    merge5_conv2 = Conv2D(256, 3, activation = 'relu', padding = 'same',
                    kernel_initializer = 'he_normal')(merge5_conv1)

    # Activation and reshape for training
    activation = Conv2D(num_classes, 1, activation = "softmax")(merge5_conv2)
    output = Reshape((dim*dim, num_classes))(activation)

    # Build model
    model = Model(inputs=[resnet.input], outputs=[output])

    return model


def ResNet_UNet_BN(dim=512, num_classes=6):
    """
    Returns a ResNet50 Nework with a U-Net
    like upsampling stage. Inlcudes 3 skip connections
    from previous VGG layers.

    Input:

        dim - the size of the input image. Note that is should be
              a square of 2 so that downsampling and upsampling
              always match. ie. 128 -> 64 -> 32 -> 64 -> 128
              This is only needed for training.

        num_classes - the number of classes in the whole problem. Used to
                      determine the dimension of output map. i.e. model.predict()
                      returns array that can be reshaped to (dim, dim,
                      num_classes).

    Output:

        model - an uncompiled keras model. Check output shape before use.


    """
    from keras.models import Model
    from keras.layers import Conv2D, BatchNormalization
    from keras.layers import UpSampling2D, Reshape, concatenate
    from keras.activations import relu
    from keras.applications.resnet50 import ResNet50

    # Import a headless ResNet50
    resnet = ResNet50(input_shape = (None, None, 3), include_top=False)

    # Attached U-net from second last layer - activation_49
    res_out = resnet.layers[-2].output

    # Standard U-Net upsampling 512 -> 256 -> 128 -> 64

    # Upsampling 1
    up1 = UpSampling2D(size=(2, 2))(res_out)
    up1_conv = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(up1)
    up1_conv = BatchNormalization()(up1_conv)
    #up1_conv = relu(up1_conv)

    prev_layer = resnet.get_layer("activation_40").output
    merge1 = concatenate([prev_layer,up1_conv], axis = 3)
    merge1_conv1 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge1)
    merge1_conv1 = BatchNormalization()(merge1_conv1)
    #merge1_conv1 = relu(merge1_conv1)

    merge1_conv2 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer = 'he_normal')(merge1_conv1)
    merge1_conv2 = BatchNormalization()(merge1_conv2)
    #merge1_conv2 = relu(merge1_conv2)

    # Upsampling 2
    up2 = UpSampling2D(size=(2, 2))(merge1_conv2)
    up2_conv = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(up2)
    up2_conv = BatchNormalization()(up2_conv)
    #up2_conv = relu(up2_conv)

    prev_layer = resnet.get_layer("activation_22").output
    merge2 = concatenate([prev_layer,up2_conv], axis = 3)
    merge2_conv1 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge2)
    merge2_conv1 = BatchNormalization()(merge2_conv1)
    #merge2_conv1 = relu(merge2_conv1)

    merge2_conv2 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge2_conv1)
    merge2_conv2 = BatchNormalization()(merge2_conv2)
    #merge2_conv2 = relu(merge2_conv2)

    # Upsampling 3
    up3 = UpSampling2D(size=(2,2))(merge2_conv2)
    up3_conv1 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(up3)
    up3_conv1 = BatchNormalization()(up3_conv1)
    #up3_conv1 = relu(up3_conv1)
    up3_conv2 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(up3_conv1)
    up3_conv2 = BatchNormalization()(up3_conv2)
    #up3_conv2 = relu(up3_conv2)

    # Upsampling 4
    up4 = UpSampling2D(size=(2,2))(up3_conv2)
    up4_conv = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(up4)
    up4_conv = BatchNormalization()(up4_conv)
    #up4_conv = relu(up4_conv)

    prev_layer = resnet.get_layer("activation_1").output
    merge3 = concatenate([prev_layer, up4_conv], axis=3)
    merge3_conv1 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge3)
    merge3_conv1 = BatchNormalization()(merge3_conv1)
    #merge3_conv1 = relu(merge3_conv1)

    merge3_conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge3_conv1)
    merge3_conv2 = BatchNormalization()(merge3_conv2)
    #merge3_conv2 = relu(merge3_conv2)

    # Upsample 5
    up5 = UpSampling2D(size=(2,2))(merge3_conv2)
    up5_conv = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(up5)
    up5_conv = BatchNormalization()(up5_conv)
    #up5_conv = relu(up5_conv)
    merge5_conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up5_conv)
    merge5_conv1 = BatchNormalization()(merge5_conv1)
    #merge5_conv1 = relu(merge5_conv1)

    merge5_conv2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge5_conv1)
    merge5_conv2 = BatchNormalization()(merge5_conv2)
    #merge5_conv2 = relu(merge5_conv2)

    # Activation and reshape for training
    activation = Conv2D(num_classes, 1, activation="softmax")(merge5_conv2)
    output = Reshape((dim*dim, num_classes))(activation)

    # Build model
    model = Model(inputs=[resnet.input], outputs=[output])

    return model


def ResNet_UNet_Dropout(dim=512, num_classes=6, dropout=0.5, final_activation=True):
    """
    Returns a ResNet50 Nework with a U-Net
    like upsampling stage. Inlcudes skip connections
    from previous ResNet50 layers.

    Uses a SpatialDrop on the final layer as introduced
    in https://arxiv.org/pdf/1411.4280.pdf, 2015.

    Input:

        dim - the size of the input image. Note that is should be
              a square of 2 so that downsampling and upsampling
              always match. ie. 128 -> 64 -> 32 -> 64 -> 128
              This is only needed for training.

        num_classes - the number of classes in the whole problem. Used to
                      determine the dimension of output map. i.e. model.predict()
                      returns array that can be reshaped to (dim, dim,
                      num_classes).

    Output:

        model - an uncompiled keras model. Check output shape before use.


    """
    from keras.models import Model
    from keras.layers import Conv2D, SpatialDropout2D
    from keras.layers import UpSampling2D, Reshape, concatenate
    from keras.applications.resnet50 import ResNet50

    # Import a headless ResNet50
    resnet = ResNet50(input_shape = (None, None, 3), include_top=False)

    # Attached U-net from second last layer - activation_49
    res_out = resnet.layers[-2].output

    # Standard U-Net upsampling 512 -> 256 -> 128 -> 64

    # Upsampling 1 - 512
    fs = 32
    up1 = UpSampling2D(size=(2,2))(res_out)
    up1_conv = Conv2D(fs, 2, activation = 'relu', padding = 'same',
                 kernel_initializer = 'he_normal')(up1)

    prev_layer = resnet.get_layer("activation_40").output
    merge1 = concatenate([prev_layer,up1_conv], axis = 3)
    merge1_conv1 = Conv2D(fs, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = 'he_normal')(merge1)
    merge1_conv2 = Conv2D(fs, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = 'he_normal')(merge1_conv1)

    # Upsampling 2 - 256
    fs = 32
    up2 = UpSampling2D(size = (2,2))(merge1_conv2)
    up2_conv = Conv2D(fs, 2, activation = 'relu', padding = 'same',
                 kernel_initializer = 'he_normal')(up2)
    prev_layer = resnet.get_layer("activation_22").output
    merge2 = concatenate([prev_layer,up2_conv], axis = 3)
    merge2_conv1 = Conv2D(fs, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = 'he_normal')(merge2)
    merge2_conv2 = Conv2D(fs, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = 'he_normal')(merge2_conv1)

    # Upsampling 3 & 4 - 128
    fs = 32
    up3 = UpSampling2D(size = (2,2))(merge2_conv2)
    up3_conv1 = Conv2D(fs, 2, activation = 'relu', padding = 'same',
                 kernel_initializer = 'he_normal')(up3)
    up3_conv2 = Conv2D(fs, 2, activation = 'relu', padding = 'same',
                 kernel_initializer = 'he_normal')(up3_conv1)
    up4 = UpSampling2D(size = (2,2))(up3_conv2)
    up4_conv = Conv2D(fs, 2, activation = 'relu', padding = 'same',
                 kernel_initializer = 'he_normal')(up4)
    prev_layer = resnet.get_layer("activation_1").output
    merge3 = concatenate([prev_layer,up4_conv], axis = 3)
    merge3_conv1 = Conv2D(fs, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = 'he_normal')(merge3)
    merge3_conv2 = Conv2D(fs, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = 'he_normal')(merge3_conv1)

    # Upsample 5 - 64
    fs = 32
    up5 = UpSampling2D(size=(2,2))(merge3_conv2)
    up5_conv = Conv2D(fs, 2, activation = 'relu', padding = 'same',
                  kernel_initializer = 'he_normal')(up5)
    merge5_conv1 = Conv2D(fs, 3, activation = 'relu', padding = 'same',
                    kernel_initializer = 'he_normal')(up5_conv)
    merge5_conv2 = Conv2D(fs, 3, activation = 'relu', padding = 'same',
                    kernel_initializer = 'he_normal')(merge5_conv1)

    # Drop Out
    do = SpatialDropout2D(dropout)(merge5_conv2)
    # Activation and reshape for training
    if final_activation:
        activation = Conv2D(num_classes, 1, activation="softmax")(do)
    else:
        activation = Conv2D(num_classes, 1, activation=None)(do)
    output = Reshape((dim*dim, num_classes))(activation)

    # Build model
    model = Model(inputs=[resnet.input], outputs=[output])

    return model



def ResNet_UNet_Reg(dim=512, num_classes=6, reg=5e-4):
    """
    Returns a ResNet50 Nework with a U-Net
    like upsampling stage. Inlcudes skip connections
    from previous ResNet50 layers.

    Uses a SpatialDrop on the final layer as introduced
    in https://arxiv.org/pdf/1411.4280.pdf, 2015.

    Input:

        dim - the size of the input image. Note that is should be
              a square of 2 so that downsampling and upsampling
              always match. ie. 128 -> 64 -> 32 -> 64 -> 128
              This is only needed for training.

        num_classes - the number of classes in the whole problem. Used to
                      determine the dimension of output map. i.e. model.predict()
                      returns array that can be reshaped to (dim, dim,
                      num_classes).

    Output:

        model - an uncompiled keras model. Check output shape before use.


    """
    from keras.models import Model
    from keras.layers import Conv2D, SpatialDropout2D
    from keras.layers import UpSampling2D, Reshape, concatenate
    from keras.regularizers import l2
    from keras.applications.resnet50 import ResNet50


    # Import a headless ResNet50
    resnet = ResNet50(input_shape = (None, None, 3), include_top=False)

    # Attached U-net from second last layer - activation_49
    res_out = resnet.layers[-2].output

    # Standard U-Net upsampling 512 -> 256 -> 128 -> 64

    # Upsampling 1 - 512
    fs = 32
    up1 = UpSampling2D(size=(2,2))(res_out)
    up1_conv = Conv2D(fs, 2, activation = 'relu', padding = 'same',
                 kernel_initializer = 'he_normal', kernel_regularizer = l2(reg), bias_regularizer = l2(reg))(up1)

    prev_layer = resnet.get_layer("activation_40").output
    merge1 = concatenate([prev_layer,up1_conv], axis = 3)
    merge1_conv1 = Conv2D(fs, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = 'he_normal', kernel_regularizer = l2(reg), bias_regularizer = l2(reg))(merge1)
    merge1_conv2 = Conv2D(fs, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = 'he_normal', kernel_regularizer = l2(reg), bias_regularizer = l2(reg))(merge1_conv1)

    # Upsampling 2 - 256
    fs = 32
    up2 = UpSampling2D(size = (2,2))(merge1_conv2)
    up2_conv = Conv2D(fs, 2, activation = 'relu', padding = 'same',
                 kernel_initializer = 'he_normal', kernel_regularizer = l2(reg), bias_regularizer = l2(reg))(up2)
    prev_layer = resnet.get_layer("activation_22").output
    merge2 = concatenate([prev_layer,up2_conv], axis = 3)
    merge2_conv1 = Conv2D(fs, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = 'he_normal', kernel_regularizer = l2(reg), bias_regularizer = l2(reg))(merge2)
    merge2_conv2 = Conv2D(fs, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = 'he_normal', kernel_regularizer = l2(reg), bias_regularizer = l2(reg))(merge2_conv1)

    # Upsampling 3 & 4 - 128
    fs = 32
    up3 = UpSampling2D(size = (2,2))(merge2_conv2)
    up3_conv1 = Conv2D(fs, 2, activation = 'relu', padding = 'same',
                 kernel_initializer = 'he_normal', kernel_regularizer = l2(reg), bias_regularizer = l2(reg))(up3)
    up3_conv2 = Conv2D(fs, 2, activation = 'relu', padding = 'same',
                 kernel_initializer = 'he_normal', kernel_regularizer = l2(reg), bias_regularizer = l2(reg))(up3_conv1)
    up4 = UpSampling2D(size = (2,2))(up3_conv2)
    up4_conv = Conv2D(fs, 2, activation = 'relu', padding = 'same',
                 kernel_initializer = 'he_normal', kernel_regularizer = l2(reg), bias_regularizer = l2(reg))(up4)
    prev_layer = resnet.get_layer("activation_1").output
    merge3 = concatenate([prev_layer,up4_conv], axis = 3)
    merge3_conv1 = Conv2D(fs, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = 'he_normal', kernel_regularizer = l2(reg), bias_regularizer = l2(reg))(merge3)
    merge3_conv2 = Conv2D(fs, 3, activation = 'relu', padding = 'same',
                   kernel_initializer = 'he_normal', kernel_regularizer = l2(reg), bias_regularizer = l2(reg))(merge3_conv1)

    # Upsample 5 - 64
    fs = 32
    up5 = UpSampling2D(size=(2,2))(merge3_conv2)
    up5_conv = Conv2D(fs, 2, activation = 'relu', padding = 'same',
                  kernel_initializer = 'he_normal', kernel_regularizer = l2(reg), bias_regularizer = l2(reg))(up5)
    merge5_conv1 = Conv2D(fs, 3, activation = 'relu', padding = 'same',
                    kernel_initializer = 'he_normal', kernel_regularizer = l2(reg), bias_regularizer = l2(reg))(up5_conv)
    merge5_conv2 = Conv2D(fs, 3, activation = 'relu', padding = 'same',
                    kernel_initializer = 'he_normal', kernel_regularizer = l2(reg), bias_regularizer = l2(reg))(merge5_conv1)


    # Activation and reshape for training
    activation = Conv2D(num_classes, 1, activation = "softmax")(merge5_conv2)
    output = Reshape((dim*dim, num_classes))(activation)

    # Build model
    model = Model(inputs=[resnet.input], outputs=[output])

    return model


def UNet(dim=512, num_classes=12):
    """
    Standard U-Net architecture for segmentation
    """

    from keras.models import Model
    from keras.layers import Input
    from keras.layers import Conv2D, SpatialDropout2D, MaxPool2D
    from keras.layers import UpSampling2D, Reshape, concatenate
    from keras.regularizers import l2


    input = Input(shape=(None, None, 3))

    # Down 1
    fs = 64
    conv1 = Conv2D(fs, 3, activation="relu", padding="same")(input)
    conv2 = Conv2D(fs, 3, activation="relu", padding="same")(conv1)
    pool1 = MaxPool2D((2, 2))(conv2)

    # Down 2
    fs = 128
    conv3 = Conv2D(fs, 3, activation="relu", padding="same")(pool1)
    conv4 = Conv2D(fs, 3, activation="relu", padding="same")(conv3)
    pool2 = MaxPool2D((2, 2))(conv4)

    # Down 3
    fs = 256
    conv5 = Conv2D(fs, 3, activation="relu", padding="same")(pool2)
    conv6 = Conv2D(fs, 3, activation="relu", padding="same")(conv5)
    pool3 = MaxPool2D((2, 2))(conv6)

    # Down 4
    fs = 512
    conv7 = Conv2D(fs, 3, activation="relu", padding="same")(pool3)
    conv8 = Conv2D(fs, 3, activation="relu", padding="same")(conv7)
    pool4 = MaxPool2D((2, 2))(conv8)

    # Bottom
    fs = 1024
    conv9 = Conv2D(fs, 3, activation="relu", padding="same")(pool4)
    conv10 = Conv2D(fs, 3, activation="relu", padding="same")(conv9)

    # Up 1
    f2 = 512
    up1 = UpSampling2D(size=(2,2))(conv10)
    up1_merge = concatenate([up1, conv8], axis=3)
    up1_conv1 = Conv2D(fs, 3, activation="relu", padding="same")(up1_merge)
    up1_conv2 = Conv2D(fs, 3, activation="relu", padding="same")(up1_conv1)

    # Up 2
    fs = 256
    up2 = UpSampling2D(size=(2, 2))(up1_conv2)
    up2_merge = concatenate([up2, conv6], axis=3)
    up2_conv1 = Conv2D(fs, 3, activation="relu", padding="same")(up2_merge)
    up2_conv2 = Conv2D(fs, 3, activation="relu", padding="same")(up2_conv1)

    # Up 3
    fs = 128
    up3 = UpSampling2D(size=(2, 2))(up2_conv2)
    up3_merge = concatenate([up3, conv4], axis=3)
    up3_conv1 = Conv2D(fs, 3, activation="relu", padding="same")(up3_merge)
    up3_conv2 = Conv2D(fs, 3, activation="relu", padding="same")(up3_conv1)

    # Up 4
    fs = 64
    up4 = UpSampling2D(size=(2, 2))(up3_conv2)
    up4_merge = concatenate([up4, conv2], axis=3)
    up4_conv1 = Conv2D(fs, 3, activation="relu", padding="same")(up4_merge)
    up4_conv2 = Conv2D(fs, 3, activation="relu", padding="same")(up4_conv1)

    # Activation and reshape for training
    if num_classes > 2:
        activation = Conv2D(num_classes, 1, activation="softmax")(up4_conv2)
        output = Reshape((dim * dim, num_classes))(activation)

    else:
        activation = Conv2D(1, 1, activation="sigmoid")(up4_conv2)
        output = Reshape((dim * dim, 1))(activation)

    # Build model
    model = Model(inputs=[input], outputs=[output])

    return model

# ---------------------------------------------------------------------------- #
def ResNet_UNet_Generator(dim=512, num_classes=6):
    """
    Returns a ResNet50 Nework with a U-Net
    like upsampling stage. Inlcudes 3 skip connections
    from previous VGG layers.

    Input:

        dim - the size of the input image. Note that is should be
              a square of 2 so that downsampling and upsampling
              always match. ie. 128 -> 64 -> 32 -> 64 -> 128
              This is only needed for training.

        num_classes - the number of classes in the whole problem. Used to
                      determine the dimension of output map. i.e. model.predict()
                      returns array that can be reshaped to (dim, dim,
                      num_classes).

    Output:

        model - an uncompiled keras model. Check output shape before use.


    """
    from keras.models import Model
    from keras.layers import Conv2D, LeakyReLU, Softmax
    from keras.layers import UpSampling2D, Reshape, concatenate
    from keras.applications.resnet50 import ResNet50

    # Import a headless ResNet50
    resnet = ResNet50(input_shape = (None, None, 3), include_top=False)

    # Attached U-net from second last layer - activation_49
    res_out = resnet.layers[-2].output

    # Standard U-Net upsampling 512 -> 256 -> 128 -> 64

    # Upsampling 1
    up1 = UpSampling2D(size=(2,2))(res_out)
    up1_conv = Conv2D(512, 2, activation=None, padding = 'same',
                 kernel_initializer='he_normal')(up1)
    up1_conv = LeakyReLU()(up1_conv)

    prev_layer = resnet.get_layer("activation_40").output
    merge1 = concatenate([prev_layer, up1_conv], axis = 3)
    merge1_conv1 = Conv2D(512, 3, activation=None, padding = 'same',
                   kernel_initializer='he_normal')(merge1)
    merge1_conv1 = LeakyReLU()(merge1_conv1)
    merge1_conv2 = Conv2D(512, 3, activation=None, padding = 'same',
                   kernel_initializer='he_normal')(merge1_conv1)
    merge1_conv2 = LeakyReLU()(merge1_conv2)

    # Upsampling 2
    up2 = UpSampling2D(size = (2,2))(merge1_conv2)
    up2_conv = Conv2D(256, 2, activation=None, padding = 'same',
                 kernel_initializer='he_normal')(up2)
    up2_conv = LeakyReLU()(up2_conv)
    prev_layer = resnet.get_layer("activation_22").output
    merge2 = concatenate([prev_layer,up2_conv], axis = 3)
    merge2_conv1 = Conv2D(256, 3, activation=None, padding = 'same',
                   kernel_initializer='he_normal')(merge2)
    merge2_conv1 = LeakyReLU()(merge2_conv1)
    merge2_conv2 = Conv2D(256, 3, activation=None, padding = 'same',
                   kernel_initializer='he_normal')(merge2_conv1)
    merge2_conv2 = LeakyReLU()(merge2_conv2)

    # Upsampling 3 & 4
    up3 = UpSampling2D(size=(2, 2))(merge2_conv2)
    up3_conv1 = Conv2D(128, 2, activation=None, padding='same',
                 kernel_initializer='he_normal')(up3)
    up3_conv1 = LeakyReLU()(up3_conv1)

    up3_conv2 = Conv2D(128, 2, activation=None, padding='same',
                 kernel_initializer='he_normal')(up3_conv1)
    up3_conv2 = LeakyReLU()(up3_conv2)

    up4 = UpSampling2D(size=(2, 2))(up3_conv2)
    up4_conv = Conv2D(128, 2, activation=None, padding='same',
                 kernel_initializer='he_normal')(up4)
    up4_conv = LeakyReLU()(up4_conv)

    prev_layer = resnet.get_layer("activation_1").output
    merge3 = concatenate([prev_layer, up4_conv], axis = 3)
    merge3_conv1 = Conv2D(128, 3, activation=None, padding = 'same',
                   kernel_initializer='he_normal')(merge3)
    merge3_conv1 = LeakyReLU()(merge3_conv1)

    merge3_conv2 = Conv2D(128, 3, activation=None, padding = 'same',
                   kernel_initializer='he_normal')(merge3_conv1)
    merge3_conv2 = LeakyReLU()(merge3_conv2)


    # Upsample 5
    up5 = UpSampling2D(size=(2, 2))(merge3_conv2)
    up5_conv = Conv2D(64, 2, activation=None, padding='same',
                  kernel_initializer='he_normal')(up5)
    up5_conv = LeakyReLU()(up5_conv)

    merge5_conv1 = Conv2D(64, 3, activation=None, padding='same',
                    kernel_initializer='he_normal')(up5_conv)
    merge5_conv1 = LeakyReLU()(merge5_conv1)

    merge5_conv2 = Conv2D(64, 3, activation=None, padding='same',
                    kernel_initializer='he_normal')(merge5_conv1)
    merge5_conv2 = LeakyReLU()(merge5_conv2)

    # Activation and reshape for training
    activation = Conv2D(num_classes, 1, activation="softmax")(merge5_conv2)

    # NOT RESHAPE

    # Build model
    model = Model(inputs=[resnet.input], outputs=[activation])

    return model


def ResNet_UNet_Descriminator(dim=512, num_classes=12):
    """
    Descriminator for adversarial training...
    """
    from keras.models import Model
    from keras.layers import Conv2D, LeakyReLU, Dropout, Input, Flatten, Dense

    discriminator_input = Input(shape=(dim, dim, num_classes))
    x = Conv2D(128, 3)(discriminator_input)
    x = LeakyReLU()(x)
    x = Conv2D(128, 4, strides=2)(x)
    x = LeakyReLU()(x)
    x = Conv2D(128, 4, strides=2)(x)
    x = LeakyReLU()(x)
    x = Conv2D(128, 4, strides=2)(x)
    x = LeakyReLU()(x)
    x = Flatten()(x)

    # One dropout layer - an important trick
    x = Dropout(0.5)(x)

    # Classificiation layer
    x = Dense(1, activation='sigmoid')(x)

    model = Model(discriminator_input, x)

    return model





from keras.activations import softmax

class TemperatureScaling(Layer):
    def __init__(self, T=1, T_is_trainable=True, use_activation=False, **kwargs):
        self.T = T
        self.T_is_trainable = T_is_trainable
        self.use_activation = use_activation
        super(TemperatureScaling, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='T',
                                      shape=(1,),
                                      initializer=Constant(value=self.T),
                                      trainable=self.T_is_trainable)
        super(TemperatureScaling, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        if self.use_activation:
            return softmax(x / self.kernel) # Cust
        return x / self.kernel

    def compute_output_shape(self, input_shape):
        return input_shape


def generate_temperature_model(model, dim=256, T=1, trainable=True, extra_reshape=False):
    # Add Temperature Scaling
    inputs = model.get_input_at(0)
    x = model.layers[-2].output
    x = TemperatureScaling(T, trainable)(x)
    x = Reshape((dim*dim, 12))(x)
    activation = Softmax(axis=-1)(x)

    if extra_reshape:
        activation = Reshape((dim, dim, 12))(activation)

    return Model(inputs=[inputs], outputs=[activation])


if __name__ == "__main__":

    dim = 256
    num_classes = 2
    model = UNet(dim, num_classes)
    model.summary()





