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

from data_generator import data_generator, val_generator
from utils.pix2pix import *
from IPython import display


class DiceScore(Loss):
    def call(self, y_true, y_pred):
        smooth = 0.1
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        score = 1 - (2. * intersection + smooth) / (
                K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + smooth)
        return score * 0.9 + mean_squared_error(y_true, y_pred) * 0.1


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

X_train = np.load("data/2000_0-1_travel_data_32.npy")
y_train = np.load("data/2000_0-1_travel_label_32.npy")

X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min())
# 实例化模型
generator = create_conv_autoencoder_with_skip_connections()
discriminator = Discriminator()

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

log_dir = "logs/"

summary_writer = tf.summary.create_file_writer(
    log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


@tf.function
def train_step(input_image, target, step):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss,
                                            generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                                 discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,
                                            generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                discriminator.trainable_variables))

    with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=step // 1000)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step // 1000)
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step // 1000)
        tf.summary.scalar('disc_loss', disc_loss, step=step // 1000)


def generate_images(model, test_input, tar):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        # Getting the pixel values in the [0, 1] range to plot.
        plt.imshow(display_list[i][:, :, 0])
        plt.axis('off')
    plt.show()


def fit(train_ds, test_ds, steps):
    example_input, example_target = next(iter(test_ds.take(1)))
    start = time.time()

    for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
        if (step) % 1000 == 0:
            display.clear_output(wait=True)

            if step != 0:
                print(f'Time taken for 1000 steps: {time.time() - start:.2f} sec\n')

            start = time.time()
            generate_images(generator, example_input, example_target)
            print(f"Step: {step // 1000}k")

        train_step(input_image, target, step)

        # Training step
        if (step + 1) % 10 == 0:
            print('.', end='', flush=True)

        # Save (checkpoint) the model every 5k steps
        if (step + 1) % 5000 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)


def load_data_train(data_file):
    matrix = tf.io.read_file(data_file)
    matrix = tf.image.decode_png(matrix, channels=35)
    X, y = matrix[:, :, :3], matrix[:, :, 3:]
    # X = (X - X.min()) / (X.max() - X.min())
    X = tf.cast(X, tf.float32)
    y = tf.cast(y, tf.float32)
    return X, y


BUFFER_SIZE = 400
# The batch size of 1 produced better results for the U-Net in the original pix2pix experiment
BATCH_SIZE = 1
# Each image is 64 * 64 in size
IMG_WIDTH = 64
IMG_HEIGHT = 64

train_data_paths = ["data/500_01_one_ball_data_32.npy", "data/5000_01_travel_data_32.npy", "data/5000_01_travel_data2_32.npy", "data/test_model_data.npy", "data/2000_0-1_travel_data_32.npy"]
train_label_paths = ["data/500_01_one_ball_label_32.npy", "data/5000_01_travel_label_32.npy", "data/5000_01_travel_label2_32.npy", "data/test_model_label.npy", "data/2000_0-1_travel_label_32.npy"]

val_data_paths = ["data/2000_0-1_travel_data_32.npy", "data/test_model_data.npy"]
val_label_paths = ["data/2000_0-1_travel_label_32.npy", "data/test_model_label.npy"]

# 创建Dataset对象
train_dataset = tf.data.Dataset.from_generator(
    generator=lambda: data_generator(train_data_paths, train_label_paths),
    output_signature=(tf.TensorSpec(shape=(64, 64, 3), dtype=tf.float32), tf.TensorSpec(shape=(64, 64, 32), dtype=tf.float32))
)
train_dataset = train_dataset.batch(32)
# train_dataset = train_dataset.shuffle(10000)

val_dataset = tf.data.Dataset.from_generator(
    generator=lambda: val_generator(val_data_paths, val_label_paths),
    output_signature=(tf.TensorSpec(shape=(64, 64, 3), dtype=tf.float32), tf.TensorSpec(shape=(64, 64, 32), dtype=tf.float32))
)
val_dataset = val_dataset.batch(32)

# test_dataset = test_dataset.map(load_data_train, num_parallel_calls=tf.data.AUTOTUNE)

fit(train_dataset, val_dataset, steps=40000*5)
