# -*- coding:utf-8 -*-
from WS_conv import WS_Conv2D
from WS_convtranspose import WS_Conv2DTranspose
from WS_depthwise import WS_Conv2DDepthwise

import tensorflow as tf

l1_l2 = tf.keras.regularizers.L1L2(0.00001, 0.000001)
l1 = tf.keras.regularizers.l1(0.00001)

def sharpening_filter(center, input_dim):

    filter = tf.constant([[-1., -1., -1., -1., -1.],
                         [-1., 2., 2., 2., -1.],
                         [-1., 2., center, 2., -1.],
                         [-1., 2., 2., 2., -1.],
                         [-1., -1., -1., -1., -1.]], dtype=tf.float32) / center
    filter = tf.expand_dims(filter, -1)
    filter = tf.expand_dims(filter, -1)
    filter = tf.tile(filter, [1, 1, input_dim, 1])
    # [5, 5, 1, 1]
    return filter

class InstanceNormalization(tf.keras.layers.Layer):
  #"""Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon
    
    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(0., 0.02),
            trainable=True)
        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)
    
    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset


def shuffle_unit(x , groups):

    height, width, ch = x.shape[1], x.shape[2], x.shape[3]

    x = tf.keras.layers.Reshape([height, width, groups, ch // groups])(x)
    x = tf.transpose(x, [0, 1, 2, 4, 3])
    x = tf.keras.layers.Reshape([height, width, ch])(x)

    return x

def SE_layer(input, filters, ratio):

    s = tf.keras.layers.GlobalAveragePooling2D()(input)

    e = tf.keras.layers.Dense(filters // ratio)(s)
    e = tf.keras.layers.ReLU()(e)
    e = tf.keras.layers.Dense(filters)(e)
    e = tf.nn.sigmoid(e)
    e = tf.keras.layers.Reshape([1, 1, filters])(e)

    out = input * e

    return out

def spatial_attention_layer(input):

    kernel_size = 7

    avg = tf.reduce_mean(input, -1, keepdims=True)
    max = tf.reduce_max(input, -1, keepdims=True)

    feature = tf.concat([avg, max], -1)

    feature = WS_Conv2D(filters=1,
                        kernel_size=kernel_size,
                        padding="same",
                        kernel_regularizer=l1_l2,
                        activity_regularizer=l1)(feature)
    feature = tf.keras.layers.BatchNormalization()(feature)
    feature = tf.nn.sigmoid(feature)

    out = input * feature

    return out

def blocks(input, filters, shuffle_group):

    top, bottom = tf.split(input, num_or_size_splits=2, axis=-1)
    half_ch = filters // 2

    top = WS_Conv2D(filters=half_ch,
                    kernel_size=1,
                    kernel_regularizer=l1_l2,
                    activity_regularizer=l1)(top)
    top = InstanceNormalization()(top)
    top = tf.keras.layers.ReLU()(top)
    top = WS_Conv2DDepthwise(kernel_size=3,
                             padding="same",
                             kernel_regularizer=l1_l2,
                             activity_regularizer=l1)(top)
    top = InstanceNormalization()(top)
    top = tf.keras.layers.ReLU()(top)
    top = WS_Conv2D(filters=half_ch,
                    kernel_size=1,
                    kernel_regularizer=l1_l2,
                    activity_regularizer=l1)(top)
    top = InstanceNormalization()(top)
    top = tf.keras.layers.ReLU()(top)

    top = WS_Conv2DDepthwise(kernel_size=3,
                             padding="same",
                             kernel_regularizer=l1_l2,
                             activity_regularizer=l1)(top)
    top = InstanceNormalization()(top)
    top = tf.keras.layers.ReLU()(top)
    top = WS_Conv2D(filters=half_ch,
                    kernel_size=1,
                    kernel_regularizer=l1_l2,
                    activity_regularizer=l1)(top)
    top = InstanceNormalization()(top)
    top = tf.keras.layers.ReLU()(top)

    top = SE_layer(top, half_ch, 8)
    top = spatial_attention_layer(top)

    bottom = WS_Conv2D(filters=half_ch,
                       kernel_size=1,
                       kernel_regularizer=l1_l2,
                       activity_regularizer=l1)(bottom)
    bottom = InstanceNormalization()(bottom)
    bottom = tf.keras.layers.ReLU()(bottom)
    bottom = WS_Conv2DDepthwise(kernel_size=3,
                                padding="same",
                                kernel_regularizer=l1_l2,
                                activity_regularizer=l1)(bottom)
    bottom = InstanceNormalization()(bottom)
    bottom = tf.keras.layers.ReLU()(bottom)
    bottom = WS_Conv2D(filters=half_ch,
                       kernel_size=1,
                       kernel_regularizer=l1_l2,
                       activity_regularizer=l1)(bottom)
    bottom = InstanceNormalization()(bottom)
    bottom = tf.keras.layers.ReLU()(bottom)

    out = tf.concat([top, bottom], -1)
    out = shuffle_unit(out, shuffle_group)

    return out

def attention_residual_block(input, dilation=1, filters=256):

    h = input
    h_attenion_layer = tf.reduce_mean(input, axis=-1, keepdims=True)
    h_attenion_layer = tf.nn.sigmoid(h_attenion_layer)    # attenion map !

    h = tf.keras.layers.ReLU()(h)
    h = tf.pad(h, [[0,0],[dilation,dilation],[0,0],[0,0]], mode='REFLECT', constant_values=0)
    h = WS_Conv2D(filters=filters, kernel_size=(3, 1), padding="VALID",
                  kernel_regularizer=l1_l2, activity_regularizer=l1, dilation_rate=dilation)(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.pad(h, [[0,0],[0,0],[dilation,dilation],[0,0]], mode='REFLECT', constant_values=0)
    h = WS_Conv2D(filters=filters, kernel_size=(1, 3), padding="VALID",
                  kernel_regularizer=l1_l2, activity_regularizer=l1, dilation_rate=dilation)(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.pad(h, [[0,0],[dilation,dilation],[dilation,dilation],[0,0]], mode='REFLECT', constant_values=0)
    h = tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding="valid",
                                        depthwise_regularizer=l1_l2, activity_regularizer=l1, dilation_rate=dilation)(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = WS_Conv2D(filters=filters, kernel_size=1, strides=1, padding="VALID",
                  kernel_regularizer=l1_l2, activity_regularizer=l1)(h)
    h = InstanceNormalization()(h)
    h = shuffle_unit(h*h_attenion_layer, 2)

    return h + input

def gaussian_and_sharpen(inputs, sigma=1.0, kernel_size=5, sharpen_center=4):

    def _get_gaussian_kernel(sigma, kernel_size):
        """Compute 1D Gaussian kernel."""
        sigma = tf.convert_to_tensor(sigma)
        x = tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1)
        x = tf.cast(x ** 2, sigma.dtype)
        x = tf.nn.softmax(-x / (2.0 * (sigma ** 2)))
        return x

    def _get_gaussian_kernel_2d(gaussian_filter_x, gaussian_filter_y):
        """Compute 2D Gaussian kernel given 1D kernels."""
        gaussian_kernel = tf.matmul(gaussian_filter_x, gaussian_filter_y)
        return gaussian_kernel

    channels = tf.shape(inputs)[3]
    sigma = tf.cast(1.0, tf.float32)
    gaussian_kernel_x = _get_gaussian_kernel(sigma, kernel_size)
    gaussian_kernel_x = gaussian_kernel_x[tf.newaxis, :]

    gaussian_kernel_y = _get_gaussian_kernel(sigma, kernel_size)
    gaussian_kernel_y = gaussian_kernel_y[:, tf.newaxis]

    gaussian_kernel_2d = _get_gaussian_kernel_2d(
        gaussian_kernel_y, gaussian_kernel_x
    )

    gaussian_kernel_2d = gaussian_kernel_2d[:, :, tf.newaxis, tf.newaxis]
    gaussian_kernel_2d = tf.tile(gaussian_kernel_2d, [1, 1, channels, 1])

    output = tf.pad(inputs, [[0,0],[2,2],[2,2],[0,0]], "REFLECT")
    output = tf.nn.depthwise_conv2d(output, filter=gaussian_kernel_2d, strides=(1,1,1,1), padding="VALID")

    sharpen_filters = sharpening_filter(4.0, channels)
    output = tf.pad(output, [[0,0],[2,2],[2,2],[0,0]], "REFLECT")
    output = tf.nn.depthwise_conv2d(output, filter=sharpen_filters, strides=(1,1,1,1), padding="VALID")

    return output

def decode_residual_block(input, dilation=1, filters=256):

    h = input

    h_attenion_layer = tf.reduce_mean(input, axis=-1, keepdims=True)
    h_attenion_layer = tf.nn.sigmoid(h_attenion_layer)    # attenion map !

    h = tf.nn.tanh(h)
    h = WS_Conv2D(filters=filters, kernel_size=1, strides=1, padding="VALID",
                                kernel_regularizer=l1_l2, activity_regularizer=l1)(h)
    h = InstanceNormalization()(h)
    h = tf.nn.tanh(h)

    h = tf.pad(h, [[0,0],[dilation,dilation],[dilation,dilation],[0,0]], mode='REFLECT', constant_values=0)
    h = WS_Conv2DDepthwise(kernel_size=3, padding="valid",
                                        depthwise_regularizer=l1_l2, activity_regularizer=l1, dilation_rate=dilation)(h)
    h = InstanceNormalization()(h)
    h = tf.nn.tanh(h)

    h = tf.pad(h, [[0,0],[dilation,dilation],[0,0],[0,0]], mode='REFLECT', constant_values=0)
    h = WS_Conv2D(filters=filters, kernel_size=(3, 1), padding="VALID",
                                kernel_regularizer=l1_l2, activity_regularizer=l1, dilation_rate=dilation)(h)
    h = InstanceNormalization()(h)
    h = tf.nn.tanh(h)
    h = tf.pad(h, [[0,0],[0,0],[dilation,dilation],[0,0]], mode='REFLECT', constant_values=0)
    h = WS_Conv2D(filters=filters, kernel_size=(1, 3), padding="VALID",
                                kernel_regularizer=l1_l2, activity_regularizer=l1, dilation_rate=dilation)(h)
    h = InstanceNormalization()(h)
    h = shuffle_unit(h*h_attenion_layer, 2)

    return h + input

def sigmoid_4x(x):
    return 1/(tf.math.exp(-4*x))

def F2M_generator(input_shape=(256, 256, 3), shuffle_group=2):

    h = inputs = tf.keras.Input(input_shape)

    first_sharpen = gaussian_and_sharpen(h * 0.5 + 0.5, sigma=1.0, kernel_size=5, sharpen_center=4) / 127.5 - 1.
    first_sharpen = sigmoid_4x(tf.reduce_mean(first_sharpen, -1, keepdims=True))
    second_sharpen = tf.image.resize(h * 0.5 + 0.5, [128, 128])
    second_sharpen = gaussian_and_sharpen(second_sharpen, sigma=1.0, kernel_size=5, sharpen_center=2.5) / 127.5 - 1.
    second_sharpen = sigmoid_4x(tf.reduce_mean(second_sharpen, -1, keepdims=True))
    thrid_sharpen = tf.image.resize(h * 0.5 + 0.5, [64, 64])
    thrid_sharpen = gaussian_and_sharpen(thrid_sharpen, sigma=1.0, kernel_size=5, sharpen_center=1) / 127.5 - 1.
    thrid_sharpen = sigmoid_4x(tf.reduce_mean(thrid_sharpen, -1, keepdims=True))

    h = tf.keras.layers.ZeroPadding2D((3,3))(h)
    h = WS_Conv2D(filters=64, kernel_size=7, strides=1, padding="VALID",
                  kernel_regularizer=l1_l2, activity_regularizer=l1, name="conv_en_1")(h)
    h = InstanceNormalization()(h * first_sharpen)
    h = shuffle_unit(h, 2)
    h = tf.keras.layers.ReLU()(h)
    h = h

    h = WS_Conv2D(filters=128, kernel_size=3, strides=2, padding="SAME",
                  kernel_regularizer=l1_l2, activity_regularizer=l1, name="conv_en_2")(h)
    h = InstanceNormalization()(h * second_sharpen)
    h = shuffle_unit(h, 2)
    h = tf.keras.layers.ReLU()(h)

    h = WS_Conv2D(filters=256, kernel_size=3, strides=2, padding="SAME",
                  kernel_regularizer=l1_l2, activity_regularizer=l1, name="conv_en_3")(h)
    h = InstanceNormalization()(h * thrid_sharpen)
    h = shuffle_unit(h, 2)
    h = tf.keras.layers.ReLU()(h)

    h = blocks(h, 256, shuffle_group=shuffle_group*4)

    for i in range(1, 7):
        h = attention_residual_block(h, dilation=i * 4, filters=256)

    h = WS_Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding="same",
                           kernel_regularizer=l1_l2, activity_regularizer=l1)(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    for i in range(1, 2):
        h = decode_residual_block(h, dilation=4, filters=128)

    h = WS_Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding="same",
                           kernel_regularizer=l1_l2, activity_regularizer=l1)(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = WS_Conv2D(filters=64, kernel_size=1, padding="VALID",
                  kernel_regularizer=l1_l2, activity_regularizer=l1)(h)

    h = tf.keras.layers.ZeroPadding2D((3,3))(h)
    h = WS_Conv2D(filters=3*16, kernel_size=7, padding="VALID",
                  kernel_regularizer=l1_l2, activity_regularizer=l1)(h)
    #h = tf.nn.tanh(h)

    return tf.keras.Model(inputs=inputs, outputs=h)

def F2M_discriminator(input_shape=(256, 256, 3)):

    h = inputs = tf.keras.Input(input_shape)

    h = WS_Conv2D(filters=64, kernel_size=4, strides=2, padding="SAME", 
                               kernel_regularizer=l1_l2, activity_regularizer=l1)(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)

    h = WS_Conv2D(filters=128, kernel_size=4, strides=2, padding="SAME",
                               kernel_regularizer=l1_l2, activity_regularizer=l1)(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)

    h = WS_Conv2D(filters=256, kernel_size=4, strides=2, padding="SAME",
                               kernel_regularizer=l1_l2, activity_regularizer=l1)(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)

    h = WS_Conv2D(filters=512, kernel_size=4, strides=1, padding="SAME",
                               kernel_regularizer=l1_l2, activity_regularizer=l1)(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)

    h = WS_Conv2D(filters=1, kernel_size=4, strides=1, padding="SAME")(h)

    return tf.keras.Model(inputs=inputs, outputs=h)
