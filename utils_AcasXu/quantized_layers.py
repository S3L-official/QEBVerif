import tensorflow as tf
from tensorflow.keras.layers import Layer
import utils_AcasXu.quantization_util as qu

if tf.__version__.startswith("1.") or tf.__version__.startswith("0."):
    raise ValueError("Please upgrade to TensorFlow 2.x")


class QuantizedModel(tf.keras.Model):
    def __init__(
            self,
            layers,
            input_bits,
            quantization_bits,
            dropout_rate=0.0,
            last_layer_signed=False,
    ):

        self.quantization_config = {
            "input_bits": input_bits,
            "quantization_bits": quantization_bits,
            "int_bits_weights": 2,
            "int_bits_bias": 2,
            "int_bits_activation": 2,
            "int_bits_input": 0,
        }

        self._last_layer_signed = last_layer_signed
        super(QuantizedModel, self).__init__()

        if not input_bits in [4, 8]:
            raise ValueError("Only 4 and 8 bit inputs supported")

        self.dense_layers = []
        current_bits = input_bits
        for i, l in enumerate(layers):
            if type(l) == int:
                signed = (
                    True if self._last_layer_signed and i == len(layers) - 1 else False
                )
                last_layer = (True if i == len(layers) - 1 else False)
                self.dense_layers.append(
                    QuantizedDense(
                        output_dim=l,
                        input_bits=current_bits,
                        quantization_config=self.quantization_config,
                        signed_output=signed,
                        if_output_layer=last_layer
                    )
                )
            else:
                raise ValueError("Unexpected type {} ({})".format(type(l), str(l)))
            current_bits = quantization_bits
        self.dropout_rate = dropout_rate
        self.dropout_layer = tf.keras.layers.Dropout(dropout_rate)
        self.flatten_layer = tf.keras.layers.Flatten()

    def build(self, input_shape):
        self._input_shape = input_shape
        super(QuantizedModel, self).build(
            input_shape
        )

    def preprocess(self, x):
        x = qu.downscale_op_input(x, self.quantization_config)
        x = qu.fake_quant_op_input(x, self.quantization_config)
        return x

    def call(self, inputs, **kwargs):
        x = self.preprocess(tf.cast(inputs, tf.float32))
        x = self.flatten_layer(x)
        for c in self.dense_layers:
            if self.dropout_rate > 0.0:
                x = self.dropout_layer(x)
            x = c(x)
        return x


class QuantizableLayer(Layer):
    def __init__(self, input_bits=None, quantization_config=None, **kwargs):
        if not input_bits in [3, 4, 5, 6, 7, 8, 9, 10,12,14,16]:
            raise ValueError(
                "Input bit resolution '{}' not supported. (Supported: 4-8)".format(
                    input_bits
                )
            )
        self.input_bits = input_bits
        self.quantization_config = quantization_config

        super(QuantizableLayer, self).__init__(**kwargs)


class QuantizedDense(QuantizableLayer):
    def __init__(
            self,
            output_dim,
            signed_output=False,
            if_output_layer=False,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.2),
            **kwargs
    ):
        self.units = output_dim
        self.kernel_initializer = kernel_initializer

        self.signed_output = signed_output
        self.if_output_layer = if_output_layer
        super(QuantizedDense, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name="kernel",
            shape=(int(input_shape[1]), self.units),
            initializer=self.kernel_initializer,
            trainable=True,
        )
        self.bias = self.add_weight(
            name="bias",
            shape=[self.units],
            initializer=tf.keras.initializers.Constant(0.25),
            trainable=True,
        )

        super(QuantizedDense, self).build(
            input_shape
        )

    def call(self, x, training=None):
        kernel = qu.fake_quant_op_weight(self.kernel, self.quantization_config)
        bias = qu.fake_quant_op_bias(self.bias, self.quantization_config)

        y = tf.matmul(x, kernel) + bias
        y = qu.fake_quant_op_activation(y, self.quantization_config, self.signed_output, self.if_output_layer)
        return y

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

    def get_fixed_point_weights(self):
        kernel = qu.fake_quant_op_weight(self.kernel, self.quantization_config).numpy()
        bias = qu.fake_quant_op_bias(self.bias, self.quantization_config).numpy()
        return (kernel, bias)

    def get_quantized_weights(self):
        w, b = self.get_weights()
        return (
            qu.quantize_weight(w, self.quantization_config),
            qu.quantize_bias(b, self.quantization_config),
        )
