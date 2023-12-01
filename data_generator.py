import os

from keras.preprocessing.image import random_rotation, random_brightness, random_zoom
import numpy as np

import tensorflow as tf

#
# def data_generator(data_folder, batch_size):
#     files = [file for file in os.listdir(data_folder)]
#     num_samples = len(files) // 2
#     steps_per_epoch = num_samples // batch_size
#
#     while True:
#         # 打乱数据集索引
#         indices = np.random.permutation(num_samples)
#         data_files = tf.gather(files, indices)
#
#         for step in range(steps_per_epoch):
#             # 获取当前批次的图像和标签路径
#             batch_image_paths = image_paths[step*batch_size: (step+1)*batch_size]
#             batch_label_paths = label_paths[step*batch_size: (step+1)*batch_size]
#
#             # 初始化批次的图像和标签列表
#             batch_images = []
#             batch_labels = []
#
#             # 逐个读取和处理图像和标签
#             for image_path, label_path in zip(batch_image_paths, batch_label_paths):
#                 image = process_image(image_path)  # 处理图像的自定义函数
#                 label = process_label(label_path)  # 处理标签的自定义函数
#
#                 batch_images.append(image)
#                 batch_labels.append(label)
#
#             # 转换为NumPy数组，并调整形状
#             batch_images = np.array(batch_images)
#             batch_labels = np.array(batch_labels)
#
#             yield batch_images, batch_labels
#


def data_generator(data_paths, label_paths,
                   lr_flip_rate=0.05,
                   up_flip_rate=0.05,
                   brightness_range=(0.98, 1.02)
                   ):
    for data_path, label_path in zip(data_paths, label_paths):
        X_train = np.load(data_path)
        y_train = np.load(label_path)
        X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min())
        for x, y in zip(X_train, y_train):
            if tf.random.uniform(()) < lr_flip_rate:
                x = tf.image.flip_left_right(x)
                y = tf.image.flip_left_right(y)

            if tf.random.uniform(()) < up_flip_rate:
                x = tf.image.flip_up_down(x)
                y = tf.image.flip_up_down(y)

                # x[:, :, i:i+1] = random_brightness(x[:, :, i:i+1], brightness_range, scale=False)

            x += 0.001 * np.max(x) * np.random.normal(0, 1, (64, 64, 3))

            yield x, y


def val_generator(data_paths, label_paths,
                   lr_flip_rate=0.05,
                   up_flip_rate=0.05,
                   ):
    for data_path, label_path in zip(data_paths, label_paths):
        X_train = np.load(data_path)
        y_train = np.load(label_path)
        X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min())
        for x, y in zip(X_train, y_train):

                # x[:, :, i:i+1] = random_brightness(x[:, :, i:i+1], brightness_range, scale=False)

            # x += 0.001 * np.max(x) * np.random.normal(0, 1, (64, 64, 3))

            yield x, y
