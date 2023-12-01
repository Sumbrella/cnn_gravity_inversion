import tensorflow as tf

import os
import pathlib
import time
import datetime

from matplotlib import pyplot as plt


def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result


def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result

OUTPUT_CHANNELS = 32


def create_conv_autoencoder_with_skip_connections():
  inputs = tf.keras.layers.Input(shape=[64, 64, 3])

  down_stack = [
    downsample(32, 4, apply_batchnorm=False),  # (batch_size, 32, 32, 64)
    downsample(64, 4),  # (batch_size, 16, 16, 128)
    downsample(128, 4),  # (batch_size, 8, 8, 256)
    downsample(256, 4),  # (batch_size, 4, 4, 512)
    downsample(256, 4),  # (batch_size, 2, 2, 512)
    # downsample(512, 4),  # (batch_size, 4, 4, 512)
    # downsample(512, 4),  # (batch_size, 2, 2, 512)
    # downsample(512, 4),  # (batch_size, 1, 1, 512)
  ]

  up_stack = [
    # upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
    # upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
    # upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
    upsample(256, 4, apply_dropout=True),  # (batch_size, 16, 16, 1024)
    upsample(128, 4),  # (batch_size, 32, 32, 512)
    upsample(64, 4),  # (batch_size, 64, 64, 256)
    upsample(32, 4),  # (batch_size, 128, 128, 128)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='sigmoid')  # (batch_size, 256, 256, 3)

  x = inputs

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)


loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

LAMBDA = 100

def generator_loss(disc_generated_output, gen_output, target):
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  # Mean absolute error
  gen_output = tf.cast(gen_output, tf.float32)
  target = tf.cast(target, tf.float32)
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

  total_gen_loss = gan_loss + (LAMBDA * l1_loss)

  return total_gen_loss, gan_loss, l1_loss

def Discriminator():
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[64, 64, 3], name='input_image')
  tar = tf.keras.layers.Input(shape=[64, 64, 32], name='target_image')

  x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)

  down1 = downsample(64, 4, False)(x)  # (batch_size, 128, 128, 64)
  down2 = downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
  down3 = downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
  conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

  last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

  return tf.keras.Model(inputs=[inp, tar], outputs=last)

def discriminator_loss(disc_real_output, disc_generated_output):
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss

# generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
# discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
#
#
# generator = create_conv_autoencoder_with_skip_connections()
# generator.summary()
# tf.keras.utils.plot_model(generator, show_shapes=True, dpi=200)