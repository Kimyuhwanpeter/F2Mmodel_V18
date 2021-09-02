import tensorflow as tf
tf = tf.compat.v2

from keras.utils import conv_utils
from tensorflow.python.util.tf_export import keras_export
from tensorflow.keras import activations
from tensorflow.keras import backend
from tensorflow.keras import constraints
from tensorflow.keras import initializers
from tensorflow.keras import regularizers

class WS_Conv2DTranspose(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 output_padding=None,
                 data_format=None,
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 name=None,
                 **kwargs):
        super(WS_Conv2DTranspose, self).__init__(
            name=name,
            activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs)
        self.filters=filters
        self.kernel_size = conv_utils.normalize_tuple(
            kernel_size, 2, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.padding = padding
        self.dilation_rate = conv_utils.normalize_tuple(
            dilation_rate, 2, 'dilation_rate')
        self.activation=activations.get(activation)
        self.use_bias=use_bias,
        self.kernel_initializer=initializers.get(kernel_initializer)
        self.bias_initializer=initializers.get(bias_initializer)
        self.kernel_regularizer=regularizers.get(kernel_regularizer)
        self.bias_regularizer=regularizers.get(bias_regularizer)
        self.kernel_constraint=constraints.get(kernel_constraint)
        self.bias_constraint=constraints.get(bias_constraint)

        self.output_padding = output_padding
        if self.output_padding is not None:
            self.output_padding = conv_utils.normalize_tuple(
                self.output_padding, 2, 'output_padding')

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        channel_axis = -1
        input_dim = int(input_shape[channel_axis])
        #self.input_spec = tf.keras.layers.InputSpec(ndim=4, axes={channel_axis: input_dim})
        kernel_shape = self.kernel_size + (self.filters, input_dim)

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
              name='bias_transpose',
              shape=(self.filters,),
              initializer=self.bias_initializer,
              regularizer=self.bias_regularizer,
              constraint=self.bias_constraint,
              trainable=True,
              dtype=self.dtype)
        else:
          self.bias = None
        self.built = True

    def conv_op(self, inputs, kernel, output_shape_tensor):
        kernel_mean = tf.reduce_mean(kernel, axis=[0], keepdims=True, name="kernel_mean")
        kernel = kernel - kernel_mean
        kernel_std = tf.keras.backend.std(kernel, axis=[0], keepdims=True)
        kernel = kernel / (kernel_std + 1e-5)

        if self.padding == "valid":
            tr_pad = "VALID"
        if self.padding == "same":
            tr_pad = "SAME"
        outputs = tf.nn.conv2d_transpose(
            inputs,
            self.kernel,
            output_shape_tensor,
            strides=self.strides,
            padding=tr_pad,
            dilations=self.dilation_rate)

        return outputs

    def call(self, inputs):

        inputs_shape = tf.shape(inputs)
        batch_size = inputs_shape[0]
        h_axis, w_axis = 1, 2
        height, width = None, None
        if inputs.shape.rank is not None:
            dims = inputs.shape.as_list()
            height = dims[h_axis]
            width = dims[w_axis]
        height = height if height is not None else inputs_shape[h_axis]
        width = width if width is not None else inputs_shape[w_axis]

        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.strides

        if self.output_padding is None:
          out_pad_h = out_pad_w = None
        else:
          out_pad_h, out_pad_w = self.output_padding
        
        # Infer the dynamic output shape:
        out_height = conv_utils.deconv_length(height,
                                              stride_size=stride_h,
                                              kernel_size=kernel_h,
                                              padding=self.padding,
                                              output_padding=out_pad_h,
                                              dilation=self.dilation_rate[0])

        out_width = conv_utils.deconv_length(width,
                                             stride_size=stride_w,
                                             kernel_size=kernel_w,
                                             padding=self.padding,
                                             output_padding=out_pad_w,
                                             dilation=self.dilation_rate[1])

        output_shape = (batch_size, out_height, out_width, self.filters)

        output_shape_tensor = tf.stack(output_shape)


        outputs = self.conv_op(inputs, self.kernel, output_shape_tensor)
        if not tf.executing_eagerly():
          # Infer the static output shape:
          out_shape = self.compute_output_shape(inputs.shape)
          outputs.set_shape(out_shape)

        if self.use_bias:
          outputs = tf.nn.bias_add(
              outputs,
              self.bias)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        output_shape = list(input_shape)
        c_axis, h_axis, w_axis = 3, 1, 2

        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.strides

        if self.output_padding is None:
          out_pad_h = out_pad_w = None
        else:
          out_pad_h, out_pad_w = self.output_padding

        output_shape[c_axis] = self.filters
        output_shape[h_axis] = conv_utils.deconv_length(
            output_shape[h_axis],
            stride_size=stride_h,
            kernel_size=kernel_h,
            padding=self.padding,
            output_padding=out_pad_h,
            dilation=self.dilation_rate[0])
        output_shape[w_axis] = conv_utils.deconv_length(
            output_shape[w_axis],
            stride_size=stride_w,
            kernel_size=kernel_w,
            padding=self.padding,
            output_padding=out_pad_w,
            dilation=self.dilation_rate[1])
        return tf.TensorShape(output_shape)

    def get_config(self):
        config = super(WS_Conv2DTranspose, self).get_config()
        config['output_padding'] = self.output_padding
        return config

