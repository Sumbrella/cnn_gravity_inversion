import keras.src.backend
from keras.models import Model
from keras.layers import Input, Conv2D, Conv3D, MaxPooling2D, UpSampling2D, \
    concatenate, BatchNormalization, Dropout, ELU, Conv3DTranspose, Reshape, Add, LeakyReLU, LayerNormalization, \
    Permute


def res_block(x, filters):
    x = Conv2D(filters, kernel_size=4, strides=2, padding='same')(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x1 = Conv2D(filters, kernel_size=3, padding='same')(x)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU()(x1)

    return Add()([x, x1])


def encoder(x):
    x = res_block(x, 64)
    x = res_block(x, 128)
    x = res_block(x, 256)
    x = res_block(x, 512)
    x = res_block(x, 1024)
    # x = ResBlock(x, 2048)
    return x


def Dimension_Transform(x):
    reconstructed_features = Conv2D(filters=1024, kernel_size=1)(x)
    reconstructed_features = LeakyReLU()(reconstructed_features)

    # Reshape the feature maps to prepare for conversion to 3D
    reshaped_features = Reshape((1, 2, 2, 1024))(reconstructed_features)

    # Nonlinear transformation between 3D features using 1x1x1 convolution and leaky ReLU
    converted_features = Conv3D(filters=1024, kernel_size=1, strides=(1, 1, 1))(reshaped_features)
    converted_features = LeakyReLU()(converted_features)
    return converted_features


def deconv_block(inputs, filters):
    x = Conv3D(filters=filters, kernel_size=3, strides=(1, 1, 1), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv3DTranspose(filters=filters//2, kernel_size=4, strides=(2, 2, 2), padding='same')(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    return x


def decoder(inputs):
    # x = deconv_block(inputs, filters=1024)
    x = deconv_block(inputs, filters=512)
    x = deconv_block(x, filters=256)
    x = deconv_block(x, filters=128)
    x = deconv_block(x, filters=64)
    x = deconv_block(x, filters=32)

    # 1x1x1 convolution and leaky ReLU to obtain reconstructed feature map
    x = Conv3D(filters=1, kernel_size=1, padding='same')(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    # Reshape feature map to size 32x64x64
    # X = Reshape((32, 64, 64))(x)
    # x = Reshape((64, 64, 32))(x)
    x = Permute((4, 2, 3, 1))(x)
    x = Reshape((64, 64, 32))(x)

    # 2D convolution to predict 3D models
    output = Conv2D(filters=32, kernel_size=1, activation='sigmoid')(x)  # Change C to the desired number of depth meshes
    output = Dropout(0.3)(output)
    return output


def create_conv_autoencoder_with_skip_connections():
    inputs = Input(shape=(None, None, 3))

    x = encoder(inputs)
    x = Dimension_Transform(x)
    outputs = decoder(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model

#
# model = create_conv_autoencoder_with_skip_connections()
# model.summary()
