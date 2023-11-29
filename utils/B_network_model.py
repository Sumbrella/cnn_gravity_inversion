from keras.models import Model
from keras.layers import Input, Conv2D, Conv3D, MaxPooling2D, UpSampling2D, \
    concatenate, BatchNormalization, Dropout, ELU, Conv3DTranspose, Reshape, Add

def LUConv(x, filters):
    relu1 = ELU(0.8)
    conv1 = Conv3D(filters, kernel_size=3, padding=1)
    bn1 = BatchNormalization()
    return bn1(conv1(relu1(x)))


def down_transition(x, in_chans, n_convs):
    outChans = 2 * in_chans
    down_conv = Conv3D(outChans, kernel_size=2, strides=2)
    bn1 = BatchNormalization()
    relu1 = ELU(0.8)
    relu2 = ELU(0.8)
    dp = Dropout(0.3)

    down = down_conv(x)
    down = dp(down)
    out = bn1(down)
    out = relu1(out)

    for i in range(n_convs):
        out = LUConv(out, in_chans)

    out = relu2(out)
    return out


def up_transition(x, skip_input, in_chans, out_chans, n_convs):
    up_conv = Conv3DTranspose(out_chans//2, kernel_size=2, strides=2)
    bn1 = BatchNormalization()
    dp = Dropout(0.3)
    relu1 = ELU(0.8)
    relu2 = ELU(0.8)

    up = up_conv(x)
    out = bn1(up)
    out = relu1(out)
    out = concatenate([out, skip_input], axis=-1)
    for i in range(n_convs):
        out = LUConv(out, out_chans)
    out = relu2(out)
    out = Dropout(0.3)
    return out


def output_transition(x):
    conv1 = Conv3D(1, kernel_size=3, padding=1)
    bn1 = BatchNormalization()
    conv2 = Conv3D(1, kernel_size=1)
    relu = ELU(0.8)

    out = conv1(x)
    out = bn1(x)
    out = relu(x)
    out = conv2(x, activation='sigmoid')
    out = Reshape((32, 64, 64))(out)

    return out

def input_transition(x, out_chans):
    conv0 = Conv2D(16 * 3, 3, 1, padding='same', use_bias=True)
    conv1 = Conv3D(out_chans, kernel_size=3, padding='same')
    bn1 = BatchNormalization()
    elu1 = ELU(0.8)
    elu2 = ELU(0.8)

    x = conv0(x)
    x = bn1(x)
    x = elu1(x)
    x = Reshape((3, 16, 64, 64))(x)

    x1 = conv1(x)
    out = bn1(x1)
    x16 = concatenate([x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x], axis=1)

    out = elu2(x)
    out = Add([out, x16])
    return out

def v_net():
    x = Input(shape=(64, 64, 3))

    x = input_transition(x, 16)
    down_tr32 = down_transition(x, 16, 2)
    down_tr64 = down_tr32(x, 32, 3)
    down_tr128 = down_transition(x, 64, 3)
    up_tr128 = up_transition(x, down_tr128, 128, 128, 3)
    up_tr64 = up_transition(x, down_tr64, 128, 64, 2)
    up_tr32 = up_transition(x, down_tr32, 64, 32, 1)
    out = output_transition(up_tr32)

    autoencoder = Model(inputs=x, outputs=out)

    return autoencoder

autoencoder = v_net()
autoencoder.summary()