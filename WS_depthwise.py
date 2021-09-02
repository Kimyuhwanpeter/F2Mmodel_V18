import tensorflow as tf
tf = tf.compat.v2

from keras.utils import conv_utils
from tensorflow.python.util.tf_export import keras_export
from tensorflow.keras import activations
from tensorflow.keras import backend
from tensorflow.keras import constraints
from tensorflow.keras import initializers
from tensorflow.keras import regularizers

class WS_Conv2DDepthwise(tf.keras.layers.Layer):
    def __init__(self,
                 kernel_size,
                 strides=[1,1,1,1],
                 padding='valid',
                 depth_multiplier=1,
                 data_format=None,
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 depthwise_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 depthwise_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 depthwise_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(WS_Conv2DDepthwise, self).__init__()
        self.filters = None
        self.kernel_size=conv_utils.normalize_tuple(
            kernel_size, 2, 'kernel_size')
        self.strides=strides
        self.padding=padding
        self.data_format=data_format
        self.dilation_rate=conv_utils.normalize_tuple(
            dilation_rate, 2, 'dilation_rate')
        self.activation=activation
        self.use_bias=use_bias
        self.bias_regularizer=bias_regularizer
        self.activity_regularizer=activity_regularizer
        self.bias_constraint=bias_constraint
        self.depth_multiplier = depth_multiplier
        self.depthwise_initializer = initializers.get(depthwise_initializer)
        self.depthwise_regularizer = regularizers.get(depthwise_regularizer)
        self.depthwise_constraint = constraints.get(depthwise_constraint)
        self.bias_initializer = initializers.get(bias_initializer)

    def build(self, input_shape):

        input_shape = tf.TensorShape(input_shape)
        channel_axis = -1
        input_dim = int(input_shape[channel_axis])
        depthwise_kernel_shape = self.kernel_size + (input_dim,
                                                     self.depth_multiplier)

        self.depthwise_kernel = self.add_weight(
            shape=depthwise_kernel_shape,
            initializer=self.depthwise_initializer,
            name='depthwise_kernel',
            regularizer=self.depthwise_regularizer,
            constraint=self.depthwise_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(input_dim * self.depth_multiplier,),
                                      initializer=self.bias_initializer,
                                      name='bias',
                                      regularizer=self.bias_regularizer,
                                      constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = tf.keras.layers.InputSpec(
            min_ndim=2 + 2, axes={channel_axis: input_dim})
        self.built = True

    def conv_op(self, inputs, kernel):

        if self.padding == "valid":
            tr_pad = "VALID"
        if self.padding == "same":
            tr_pad = "SAME"

        kernel_mean = tf.reduce_mean(kernel, axis=[0], keepdims=True, name="kernel_mean")
        kernel = kernel - kernel_mean
        kernel_std = tf.keras.backend.std(kernel, axis=[0], keepdims=True)
        kernel = kernel / (kernel_std + 1e-5)

        outputs = tf.nn.depthwise_conv2d(
            inputs,
            kernel,
            strides=self.strides,
            padding=tr_pad,
            dilations=self.dilation_rate)

        return outputs

    def call(self, inputs):

        outputs = self.conv_op(inputs, self.depthwise_kernel)
        if self.use_bias:
            outputs = tf.nn.bias_add(
                outputs, self.bias)

        return outputs

    def compute_output_shape(self, input_shape):
        rows = input_shape[1]
        cols = input_shape[2]
        out_filters = input_shape[3] * self.depth_multiplier

        rows = conv_utils.conv_output_length(rows, self.kernel_size[0],
                                             self.padding,
                                             self.strides[0],
                                             self.dilation_rate[0])
        cols = conv_utils.conv_output_length(cols, self.kernel_size[1],
                                             self.padding,
                                             self.strides[1],
                                             self.dilation_rate[1])

        return (input_shape[0], rows, cols, out_filters)