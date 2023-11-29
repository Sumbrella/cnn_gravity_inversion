import datetime
import os
from itertools import repeat

import keras.metrics
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TerminateOnNaN
from keras.losses import MeanSquaredError, Loss, mean_squared_error
from keras.preprocessing.image import ImageDataGenerator, random_rotation, random_brightness, random_zoom


import tensorflow as tf
import keras.backend as K
import numpy as np

# from utils.A_network_model import create_conv_autoencoder_with_skip_connections
from utils.A_network_model_imporved import create_conv_autoencoder_with_skip_connections
# from utils.C_network_model import create_conv_autoencoder_with_skip_connections
# from utils.A_network_model_big import create_conv_autoencoder_with_skip_connections


class DiceScore(Loss):
    def call(self, y_true, y_pred):
        smooth = 1.0
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        score = 1 - (2. * intersection + smooth) / (
                    K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + smooth)
        return score * 0.8 + mean_squared_error(y_true, y_pred) * 0.2


class AttenMSE(Loss):
    def call(self, y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        coff = K.abs(y_true_f)
        coff = coff / K.sum(coff)
        loss = K.abs(y_true_f - y_pred_f) * coff
        return 0.8 * K.sum(loss) + 0.2 * mean_squared_error(y_true, y_pred)


train_shape = (64, 64, 3)
out_shape = (64, 64, 32)

# X_train = np.load("data/simple_data_32.npy")
# y_train = np.load("data/simple_label_32.npy")

# X_train = np.load("data/one_ball_data_32.npy")
# y_train = np.load("data/one_ball_label_32.npy")

# X_train = np.load("data/500_one_ball_data_32.npy")
# y_train = np.load("data/500_one_ball_label_32.npy")

# X_train = np.load("data/0-1_travel_data_32.npy")
# y_train = np.load("data/0-1_travel_label_32.npy")

# X_train = np.load("data/1000_travel_data_32.npy")
# y_train = np.load("data/1000_travel_label_32.npy")

# X_train = np.load("data/2000_0-1_travel_data_32.npy")
# y_train = np.load("data/2000_0-1_travel_label_32.npy")

# X_train = np.load("data/5000_travel_data_32.npy")
# y_train = np.load("data/5000_travel_label_32.npy")

X_train = np.load("data/5000_01_travel_data_32.npy")
y_train = np.load("data/5000_01_travel_label_32.npy")

X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min())

N = X_train.shape[0] // 2
# X_train = np.tile(X_train, (2, 1, 1, 1))
# y_train = np.tile(y_train, (2, 1, 1, 1))

for i in range(N):
    X_train[i] += 0.012 * np.max(X_train[i]) * np.random.normal(0, 1, (64, 64, 3))

# X_train[N:] = np.asarray([random_brightness(x, (0.99, 1.01)) for x in X_train[N:]])
# X_train[N:] = np.asarray([random_zoom(x, (0.99, 1.01)) for x in X_train[N:]])


# # == Generator ==
# data_gen_args = dict(
#     # rotation_range=3,
#     # brightness_range=(0.99, 1.01),
#     horizontal_flip=True,
#     vertical_flip=True,
#     validation_split=0.1
# )

# X_train = np.asarray([random_brightness(x, (0.98, 1.02)) for x in X_train])
# X_train = np.asarray([random_zoom(x, (0.98, 1.02)) for x in X_train])
# X_train = np.asarray([random_rotation(x, 2) for x in X_train])

# image_datagen = ImageDataGenerator(**data_gen_args)
# mask_datagen = ImageDataGenerator(**data_gen_args)

# # Provide the same seed and keyword arguments to the fit and flow methods
# seed = np.random.randint(10000)
# image_datagen.fit(X_train, augment=True, seed=seed)
# mask_datagen.fit(y_train, augment=True, seed=seed)

# image_train_generator = image_datagen.flow(X_train, subset='training')
# mask_train_generator = mask_datagen.flow(y_train, subset='training')

# image_val_generator = image_datagen.flow(X_train, subset='validation')
# mask_val_generator = mask_datagen.flow(y_train, subset='validation')

# train_generator = zip(image_train_generator, mask_train_generator)
# val_generator = zip(image_val_generator, mask_val_generator)

# == end generator ===

# 实例化模型
autoencoder_with_skip = create_conv_autoencoder_with_skip_connections()

# 编译模型
# autoencoder_with_skip.compile(optimizer='adam', loss=MeanSquaredError())
autoencoder_with_skip.compile(
    optimizer='adam',
    loss=DiceScore(),
    # loss=MeanSquaredError(),
    # loss=AttenMSE(),
    metrics=[
        keras.metrics.mean_squared_error,
        keras.metrics.binary_accuracy,
    ]
)

model_dir = os.path.join("models", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

# 选择模型存储的位置
checkpoint = ModelCheckpoint(
    os.path.join(model_dir, "best_model.h5"),
    verbose=1,
    save_best_only=True
)

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1
)

# 使用早停法来结束训练当验证损失不再改善
earlystopping = EarlyStopping(patience=100)

# Nan时停止
terminalNan = TerminateOnNaN()

# 训练模型
# with tf.device("/gpu:0"):
#     history = autoencoder_with_skip.fit(
#         # X_train, y_train,
#         train_generator,
#         validation_data=val_generator,
#         steps_per_epoch=300,
#         validation_steps=10,
#         epochs=100,
#         batch_size=8,
#         shuffle=True,
#         # validation_split=0.1, 
#         callbacks=[tensorboard_callback, earlystopping, checkpoint, terminalNan]
#     )

with tf.device("/gpu:0"):
    history = autoencoder_with_skip.fit(
        X_train, y_train,
        epochs=200,
        batch_size=64,
        shuffle=True,
        validation_split=0.05, 
        callbacks=[tensorboard_callback, earlystopping, checkpoint, terminalNan]
    )


# 保存模型
autoencoder_with_skip.save(os.path.join(model_dir, 'final_model.h5'))

