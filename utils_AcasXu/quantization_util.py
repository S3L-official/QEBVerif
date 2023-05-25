import numpy as np
import tensorflow as tf


def quantize_int(float_value, num_bits, frac_bits):
    min_value, max_value = int_get_min_max(num_bits, frac_bits)
    float_value = np.clip(float_value, min_value, max_value)

    scaled = float_value * (2 ** frac_bits)
    quant = np.int32(scaled)
    if type(quant) == np.ndarray:
        incs = (scaled - quant) >= 0.5
        decs = (scaled - quant) <= -0.5
        quant[incs] += 1
        quant[decs] -= 1
    else:
        if scaled - quant >= 0.5:
            quant += 1
        elif scaled - quant <= -0.5:
            quant -= 1

    return np.int32(quant)


def quantize_uint(float_value, num_bits, frac_bits):
    min_value, max_value = uint_get_min_max(num_bits, frac_bits)
    float_value = np.clip(float_value, min_value, max_value)

    scaled = float_value * (2 ** frac_bits)
    quant = np.int32(scaled)
    if type(quant) == np.ndarray:
        incs = (scaled - quant) >= 0.5
        decs = (scaled - quant) <= -0.5
        quant[incs] += 1
        quant[decs] -= 1
    else:
        if scaled - quant >= 0.5:
            quant += 1
        elif scaled - quant <= -0.5:
            quant -= 1

    return np.uint32(quant)


def de_quantize_uint(int_value, num_bits, frac_bits):
    real = np.float32(int_value)
    real = real / (2 ** frac_bits)
    return real


def de_quantize_int(int_value, num_bits, frac_bits):
    real = np.float32(int_value)
    real = real / (2 ** frac_bits)
    return real


def get_activation_eps(quantization_config):
    num_bits = quantization_config["quantization_bits"]
    frac_bits = num_bits - quantization_config["int_bits_weights"]
    eps = 0.2 / (2 ** frac_bits)
    return eps


def quantize_weight(v, quantization_config):
    num_bits = quantization_config["quantization_bits"]
    frac_bits = num_bits - quantization_config["int_bits_weights"]
    return quantize_int(v, num_bits, frac_bits)


def quantize_bias(v, quantization_config):
    num_bits = quantization_config["quantization_bits"]
    frac_bits = num_bits - quantization_config["int_bits_bias"]
    return quantize_int(v, num_bits, frac_bits)


def binary_str_to_int(binary_str):
    value = 0
    twos_complement = False
    if binary_str[0] == "1":
        twos_complement = True

    for i in range(1, len(binary_str)):
        value *= 2
        if (not twos_complement and binary_str[i] == "1") or (
                twos_complement and binary_str[i] == "0"
        ):
            value += 1

    if twos_complement:
        value += 1
        value = -value
    return value


def binary_str_to_uint(binary_str):
    value = 0

    for i in range(len(binary_str)):
        value *= 2
        if binary_str[i] == "1":
            value += 1
    return value


def uint_get_min_max_integer(num_bits, frac_bits):
    min_value = 0
    max_value = 2 ** num_bits - 1
    return (min_value, max_value)


def int_get_min_max_integer(num_bits, frac_bits):
    num_value_bits = num_bits - 1
    min_value = -(2 ** num_value_bits)
    max_value = 2 ** num_value_bits - 1
    return (min_value, max_value)


def uint_get_min_max(num_bits, frac_bits):
    min_value = 0
    max_value = (2 ** num_bits - 1) / (2 ** frac_bits)
    return (min_value, max_value)


def int_get_min_max(num_bits, frac_bits):
    num_value_bits = num_bits - 1
    min_value = -(2 ** num_value_bits) / (2 ** frac_bits)
    max_value = (2 ** num_value_bits - 1) / (2 ** frac_bits)
    return (min_value, max_value)


def fake_quant_op_activation(x, quantization_config, signed_output, if_output):
    if if_output:
        return x

    num_bits = quantization_config["quantization_bits"]
    frac_bits = num_bits - quantization_config["int_bits_activation"]

    if signed_output:
        min_val, max_val = int_get_min_max(num_bits, frac_bits)
    else:
        min_val, max_val = uint_get_min_max(num_bits, frac_bits)

    return tf.quantization.fake_quant_with_min_max_args(
        x, min=min_val, max=max_val, num_bits=num_bits
    )


def fake_quant_bounds_activation(quantization_config, signed_output):
    num_bits = quantization_config["quantization_bits"]
    frac_bits = num_bits - quantization_config["int_bits_activation"]
    if signed_output:
        min_val, max_val = int_get_min_max(num_bits, frac_bits)
    else:
        min_val, max_val = uint_get_min_max(num_bits, frac_bits)
    return min_val, max_val


def fake_quant_op_weight(w, quantization_config):
    num_bits = quantization_config["quantization_bits"]
    frac_bits = num_bits - quantization_config["int_bits_weights"]
    min_val, max_val = int_get_min_max(num_bits, frac_bits)

    return tf.quantization.fake_quant_with_min_max_args(
        w, min=min_val, max=max_val, num_bits=num_bits
    )


def fake_quant_op_bias(b, quantization_config):
    num_bits = quantization_config["quantization_bits"]
    frac_bits = num_bits - quantization_config["int_bits_bias"]
    min_val, max_val = int_get_min_max(num_bits, frac_bits)
    return tf.quantization.fake_quant_with_min_max_args(
        b, min=min_val, max=max_val, num_bits=num_bits
    )


def fake_quant_op_input(x, quantization_config):
    num_bits = quantization_config["input_bits"]
    frac_bits = num_bits - quantization_config["int_bits_input"]
    min_val, max_val = int_get_min_max(num_bits, frac_bits)

    return tf.quantization.fake_quant_with_min_max_args(
        x, min=min_val, max=max_val, num_bits=num_bits
    )


def downscale_op_input(x, quantization_config):
    num_bits = quantization_config["input_bits"]
    frac_bits = num_bits - quantization_config["int_bits_input"]
    min_val, max_val = uint_get_min_max(num_bits, frac_bits)
    return x * max_val / (2 ** num_bits - 1)


def forward_DNN(x, ilp_model, relu):
    if relu == 0:
        relu = 10000
    for i, l in enumerate(ilp_model.dense_layers):
        tf_layer = ilp_model.quantized_model.dense_layers[i]
        w_cont, b_cont = tf_layer.get_weights()
        out_x = []

        if tf_layer.signed_output:
            min_val_cont, max_val_cont = int_get_min_max(
                l.quantization_config["quantization_bits"],
                l.quantization_config["quantization_bits"]
                - l.quantization_config["int_bits_activation"],
            )
        else:
            min_val_cont, max_val_cont = uint_get_min_max(
                l.quantization_config["quantization_bits"],
                l.quantization_config["quantization_bits"]
                - l.quantization_config["int_bits_activation"],
            )
        ifLast = (i == len(ilp_model.dense_layers) - 1)

        for out_index in range(l.layer_size):
            weight_row = np.float32(w_cont[:, out_index])
            bias = np.float32(b_cont[out_index])

            accumulator = np.float32(np.array(weight_row * x).sum() + bias)

            if not ifLast:
                if accumulator < 0:
                    accumulator = 0
                else:
                    accumulator = np.clip(accumulator, min_val_cont, relu)
            out_x.append(accumulator)
        x = np.array(out_x)
    return x


def real_round(x):
    if x < 0:
        return np.ceil(x - 0.5)
    elif x > 0:
        return np.floor(x + 0.5)
    else:
        return 0


def forward_QNN(x, ilp_model):
    for i, l in enumerate(ilp_model.dense_layers):
        tf_layer = ilp_model.quantized_model.dense_layers[i]
        w, b = tf_layer.get_quantized_weights()
        out_x = []

        if tf_layer.signed_output:
            min_val, max_val = int_get_min_max(
                l.quantization_config["quantization_bits"],
                l.quantization_config["quantization_bits"]
                - l.quantization_config["int_bits_activation"],
            )
        else:
            min_val, max_val = uint_get_min_max(
                l.quantization_config["quantization_bits"],
                l.quantization_config["quantization_bits"]
                - l.quantization_config["int_bits_activation"],
            )

        ifLast = (i == len(ilp_model.dense_layers) - 1)

        num_bits = l.quantization_config["quantization_bits"]
        frac_bits = num_bits - l.quantization_config["int_bits_activation"]

        for out_index in range(l.layer_size):
            weight_row = w[:, out_index]

            w_fp = weight_row / (
                    2 ** (l.quantization_config["quantization_bits"] - l.quantization_config["int_bits_weights"]))

            bias = b[out_index]
            bias_fp = bias / (
                    2 ** (l.quantization_config["quantization_bits"] - l.quantization_config["int_bits_bias"]))

            accumulator = np.array(w_fp * x).sum() + bias_fp

            if not ifLast:
                if accumulator < 0:
                    c = 0
                else:
                    c = np.clip(real_round(accumulator * (2 ** frac_bits)), 0, 2 ** num_bits - 1) / 2 ** frac_bits

            else:
                c = accumulator
            out_x.append(c)
        x = np.array(out_x)

    return x