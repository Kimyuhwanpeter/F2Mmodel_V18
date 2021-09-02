import tensorflow as tf
tf = tf.compat.v2

from keras.utils import conv_utils
from tensorflow.python.util.tf_export import keras_export
from tensorflow.keras import activations
from tensorflow.keras import backend
from tensorflow.keras import constraints
from tensorflow.keras import initializers
from tensorflow.keras import regularizers

class WS_Conv2D(tf.keras.layers.Layer):
    def __init__(self,
                filters,
                kernel_size,
                strides=1,
                padding='valid',
                data_format=None,
                dilation_rate=1,
                groups=1,
                activation=None,
                use_bias=True,
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None,
                trainable=True,
                name=None,
                **kwargs):
        super(WS_Conv2D, self).__init__(
            trainable=trainable,
            name=name,
            activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs)

        self.filters = filters
        self.groups = groups or 1
        self.kernel_size = conv_utils.normalize_tuple(
            kernel_size, 2, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        #self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(
            dilation_rate, 2, 'dilation_rate')
        
        self.activation = activations.get(activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = tf.keras.layers.InputSpec(min_ndim=2 + 2)

        self._is_causal = self.padding == 'causal'
        #self._tf_data_format = conv_utils.convert_data_format(
        #    self.data_format, 2 + 2)

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        input_channel = input_shape[-1]
        kernel_shape = self.kernel_size + (input_channel // self.groups,
                                            self.filters)

        self.kernel = self.add_weight(
            name='kernel',
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype)

        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)
        else:
            self.bias = None
        channel_axis = -1
        self.input_spec = tf.keras.layers.InputSpec(min_ndim=2 + 2,
                                    axes={channel_axis: input_channel})
        self.built = True

    def conv_op(self, inputs, kernel):

        if self.padding == 'causal':
            tf_padding = 'VALID'  # Causal padding handled in `call`.
        elif isinstance(self.padding, str):
            tf_padding = self.padding.upper()
        else:
            tf_padding = self.padding

        kernel_mean = tf.reduce_mean(kernel, axis=[0], keepdims=True, name="kernel_mean")
        kernel = kernel - kernel_mean
        kernel_std = tf.keras.backend.std(kernel, axis=[0], keepdims=True)
        kernel = kernel / (kernel_std + 1e-5)

        return tf.nn.conv2d(inputs, kernel, strides=list(self.strides),
                            padding=tf_padding, dilations=list(self.dilation_rate))
    def call(self, inputs):
        input_shape=inputs.shape

        if self._is_causal:  # Apply causal padding to inputs for Conv1D.
            inputs = tf.pad(inputs, self._compute_causal_padding(inputs))

        outputs = self.conv_op(inputs, self.kernel)
        if self.use_bias:
            outputs = tf.nn.bias_add(
                outputs, self.bias)

        return outputs