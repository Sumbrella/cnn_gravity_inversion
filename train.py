import datetime
import os
from itertools import repeat

import keras.metrics
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, TerminateOnNaN
from keras.losses import MeanSquaredError, Loss, mean_squared_error
from keras.preprocessing.image import ImageDataGenerator, random_rotation, random_brightness, random_zoom


import tensorflow as tf
import keras.backend as K
import numpy as np

from data_generator import data_generator, val_generator
# from utils.A_network_model import create_conv_autoencoder_with_skip_connections
from utils.A_network_model_imporved import create_conv_autoencoder_with_skip_connections
# from utils.C_network_model import create_conv_autoencoder_with_skip_connections
# from utils.A_network_model_big import create_conv_autoencoder_with_skip_connections
# from utils.pix2pix import create_conv_autoencoder_with_skip_connections


class DiceScore(Loss):
    def call(self, y_true, y_pred):
        smooth = 0.1
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        score = 1 - (2. * intersection + smooth) / (
                    K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + smooth)
        return score * 0.75 + mean_squared_error(y_true, y_pred) * 0.25


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

# X_train = np.load("data/5000_01_travel_data_32.npy")
# y_train = np.load("data/5000_01_travel_label_32.npy")

# X_train = np.load("data/test_model_data.npy")
# y_train = np.load("data/test_model_label.npy")

# X_train = np.load("data/5000_01_travel_data_32.npy")
# y_train = np.load("data/5000_01_travel_label_32.npy")

# X_train = np.concatenate([X_train, np.load("data/5000_01_travel_data2_32.npy")], axis=0)
# y_train = np.concatenate([y_train, np.load("data/5000_01_travel_label2_32.npy")], axis=0)

# X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min())

# N = X_train.shape[0] // 2

# for i in range(N):
#     X_train[i+N] += 0.002 * np.max(X_train[i+N]) * np.random.normal(0, 1, (64, 64, 3))
#
# X_train[N:] = np.asarray([random_brightness(x, (0.98, 1.02)) for x in X_train[N:]])
# X_train[N:] = np.asarray([random_zoom(x, (0.98, 1.02)) for x in X_train[N:]])


# 文件路径列表
train_data_paths = ["data/500_01_one_ball_data_32.npy", "data/5000_01_travel_data_32.npy", "data/5000_01_travel_data2_32.npy", "data/test_model_data.npy", "data/2000_0-1_travel_data_32.npy"]
train_label_paths = ["data/500_01_one_ball_label_32.npy", "data/5000_01_travel_label_32.npy", "data/5000_01_travel_label2_32.npy", "data/test_model_label.npy", "data/2000_0-1_travel_label_32.npy"]

val_data_paths = ["data/2000_0-1_travel_data_32.npy", "data/test_model_data.npy"]
val_label_paths = ["data/2000_0-1_travel_label_32.npy", "data/test_model_label.npy"]

# 创建Dataset对象
train_dataset = tf.data.Dataset.from_generator(
    generator=lambda: data_generator(train_data_paths, train_label_paths),
    output_signature=(tf.TensorSpec(shape=(64, 64, 3), dtype=tf.float32), tf.TensorSpec(shape=(64, 64, 32), dtype=tf.float32))
)
train_dataset = train_dataset.batch(64)
# train_dataset = train_dataset.shuffle(10000)

val_dataset = tf.data.Dataset.from_generator(
    generator=lambda: val_generator(val_data_paths, val_label_paths),
    output_signature=(tf.TensorSpec(shape=(64, 64, 3), dtype=tf.float32), tf.TensorSpec(shape=(64, 64, 32), dtype=tf.float32))
)
val_dataset = val_dataset.batch(64)
# val_dataset = val_dataset.shuffle(2000)

# == end generator ===

# 实例化模型
autoencoder_with_skip = create_conv_autoencoder_with_skip_connections()

# 编译模型
autoencoder_with_skip.compile(
    optimizer=Adam(learning_rate=0.001, beta_1=0.85),
    # optimizer=SGD(),
    # optimizer=RMSprop(),
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
    save_best_only=True,
    save_weights_only=True
)

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1
)

# 使用早停法来结束训练当验证损失不再改善
# earlystopping = EarlyStopping(patience=100)

# Nan时停止
terminalNan = TerminateOnNaN()

# 训练模型
with tf.device("/gpu:0"):
    history = autoencoder_with_skip.fit(
        # X_train, y_train,
        train_dataset.repeat(),
        validation_data=val_dataset.repeat(),
        steps_per_epoch=200,
        validation_steps=30,
        epochs=200,
        batch_size=64,
        shuffle=True,
        # validation_split=0.1,
        callbacks=[tensorboard_callback, checkpoint, terminalNan]
    )

# with tf.device("/gpu:0"):
#     history = autoencoder_with_skip.fit(
#         X_train, y_train,
#         epochs=500,
#         batch_size=32,
#         shuffle=True,
#         validation_split=0.05,
#         callbacks=[tensorboard_callback, checkpoint, terminalNan] #, earlystopping]
#     )


# 保存模型
autoencoder_with_skip.save(os.path.join(model_dir, 'final_model.h5'))

