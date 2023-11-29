from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, BatchNormalization, Dropout


def create_conv_autoencoder_with_skip_connections():
    input_img = Input(shape=(None, None, 3))

    # 编码器部分
    conv1 = Conv2D(64, (3, 3), activation='leaky_relu', padding='same')(input_img)
    conv1 = Dropout(0.3)(conv1)
    conv1 = Conv2D(64, (3, 3), activation='leaky_relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, (3, 3), activation='leaky_relu', padding='same')(pool1)
    conv2 = Dropout(0.3)(conv2)
    conv2 = Conv2D(128, (3, 3), activation='leaky_relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = BatchNormalization()(pool2)

    conv3 = Conv2D(256, (3, 3), activation='leaky_relu', padding='same')(pool2)
    conv3 = Dropout(0.3)(conv3)
    conv3 = Conv2D(256, (3, 3), activation='leaky_relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = BatchNormalization()(pool3)

    conv4 = Conv2D(512, (2, 2), activation='leaky_relu', padding='same')(pool3)
    conv4 = Dropout(0.3)(conv4)
    conv4 = Conv2D(512, (2, 2), activation='leaky_relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = BatchNormalization()(pool4)

    conv5 = Conv2D(1024, (2, 2), activation='leaky_relu', padding='same')(conv4)
    conv5 = Dropout(0.3)(conv5)
    conv6 = Conv2D(1024, (2, 2), activation='leaky_relu', padding='same')(conv5)
    conv6 = Dropout(0.3)(conv6)
    conv6 = BatchNormalization()(conv6)

    # 解码器部分，同时添加skip-connections
    bconv0 = Conv2D(512, (3, 3), activation='leaky_relu', padding='same')(conv6)
    bconv0 = Dropout(0.3)(bconv0)
    bconv0 = Conv2D(512, (3, 3), activation='leaky_relu', padding='same')(bconv0)
    bconv0 = Dropout(0.3)(bconv0)
    bconv0 = BatchNormalization()(bconv0)
    bconv0 = concatenate([bconv0, conv4], axis=-1)

    up1 = UpSampling2D(size=(2, 2))(bconv0)
    bconv1 = Conv2D(256, (3, 3), activation='leaky_relu', padding='same')(up1)
    bconv1 = Dropout(0.3)(bconv1)
    bconv1 = Conv2D(256, (3, 3), activation='leaky_relu', padding='same')(bconv1)
    bconv1 = Dropout(0.3)(bconv1)
    bconv1 = BatchNormalization()(bconv1)
    bconv1 = concatenate([bconv1, conv3], axis=-1)

    up2 = UpSampling2D(size=(2, 2))(bconv1)
    bconv2 = Conv2D(128, (3, 3), activation='leaky_relu', padding='same')(up2)
    bconv2 = Conv2D(128, (3, 3), activation='leaky_relu', padding='same')(bconv2)
    bconv2 = Dropout(0.3)(bconv2)
    bconv2 = BatchNormalization()(bconv2)
    bconv2 = concatenate([bconv2, conv2], axis=-1)

    up3 = UpSampling2D(size=(2, 2))(bconv2)
    bconv3 = Conv2D(64, (3, 3), activation='leaky_relu', padding='same')(up3)
    bconv3 = Conv2D(64, (3, 3), activation='leaky_relu', padding='same')(bconv3)
    bconv3 = Dropout(0.3)(bconv3)
    bconv3 = BatchNormalization()(bconv3)
    bconv3 = concatenate([bconv3, conv1], axis=-1)

    decoded = Conv2D(32, (3, 3), activation='leaky_relu', padding='same')(bconv3)
    decoded = Dropout(0.3)(decoded)
    decoded = Conv2D(32, 1, activation='sigmoid', padding='same')(bconv3)

    autoencoder = Model(inputs=input_img, outputs=decoded)

    return autoencoder

# #
# # 构建模型
autoencoder_with_skip = create_conv_autoencoder_with_skip_connections()
# #
# # 查看模型概要
autoencoder_with_skip.summary()


