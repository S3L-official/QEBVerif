import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import time
import math
from copy import deepcopy
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed, wait
import utils_mnist.quantization_util as qu


def real_round(x):
    if x < 0:
        return np.ceil(x - 0.5)
    elif x > 0:
        return np.floor(x + 0.5)
    else:
        return 0


class DP_QNN_neuron(object):
    """
    Attributes:
        algebra_lower (numpy ndarray of float): neuron's algebra lower bound(coeffients of previous neurons and a constant)
        algebra_upper (numpy ndarray of float): neuron's algebra upper bound(coeffients of previous neurons and a constant)
        concrete_algebra_lower (numpy ndarray of float): neuron's algebra lower bound(coeffients of input neurons and a constant)
        concrete_algebra_upper (numpy ndarray of float): neuron's algebra upper bound(coeffients of input neurons and a constant)
        concrete_lower (float): neuron's concrete lower bound
        concrete_upper (float): neuron's concrete upper bound
        concrete_highest_lower (float): neuron's highest concrete lower bound
        concrete_lowest_upper (float): neuron's lowest concrete upper bound
        weight (numpy ndarray of float): neuron's weight
        bias (float): neuron's bias
        certain_flag (int): 0 uncertain 1 activated(>=0) 2 deactivated(<=0) 3 over-activated(>=clamp_upper)
        prev_abs_mode (int): indicates abstract mode of relu nodes in previous iteration.0 use first,1 use second
    """

    def __init__(self):
        self.algebra_lower = None
        self.algebra_upper = None
        self.concrete_algebra_lower = None
        self.concrete_algebra_upper = None
        self.concrete_lower = None
        self.concrete_lower_noClip = None
        self.concrete_upper = None
        self.concrete_upper_noClip = None
        self.concrete_highest_lower = None
        self.concrete_lowest_upper = None
        self.weight = None
        self.bias = None
        self.prev_abs_mode = None
        self.prev_abs_mode_min = None
        self.certain_flag = 0

    def clear(self):
        self.certain_flag = 0
        self.concrete_highest_lower = None
        self.concrete_lowest_upper = None
        self.prev_abs_mode = None


class DP_QNN_layer(object):
    """
    Attributes:
        neurons (list of neuron): Layer neurons
        size (int): Layer size
        layer_type (int) : Layer type 0 input 1 affine 2 relu
    """
    INPUT_LAYER = 0
    AFFINE_LAYER = 1
    CLAMP_MAX_LAYER = 2
    CLAMP_MIN_LAYER = 3

    def __init__(self):
        self.size = None
        self.neurons = None
        self.layer_type = None

    def clear(self):
        for i in range(len(self.neurons)):
            self.neurons[i].clear()


class DP_QNN_network(object):
    """
    Attributes:
        numLayers (int): Number of weight matrices or bias vectors in neural network
        layerSizes (list of ints): Size of input layer, hidden layers, and output layer
        inputSize (int): Size of input
        outputSize (int): Size of output
        mins (list of floats): Minimum values of inputs
        maxes (list of floats): Maximum values of inputs
        means (list of floats): Means of inputs and mean of outputs
        ranges (list of floats): Ranges of inputs and range of outputs
        layers (list of layer): Network Layers
        unsafe_region (list of ndarray):coeffient of output and a constant
        property_flag (bool) : indicates the network have verification layer or not
        property_region (float) : Area of the input box
        abs_mode_changed (int) : count of uncertain relu abstract mode changed
        self.MODE_ROBUSTNESS=1
        self.MODE_QUANTITIVE=0
    """

    def __init__(self, ifSignedOutput):
        self.MODE_QUANTITIVE = 0
        self.MODE_ROBUSTNESS = 1

        self.numlayers = None
        self.layerSizes = None
        self.inputSize = None
        self.outputSize = None
        self.mins = None
        self.maxes = None
        self.ranges = None
        self.layers = None
        self.property_flag = None
        self.property_region = None
        self.abs_mode_changed = None
        self.outputSigned = ifSignedOutput

    def clear(self):
        for i in range(len(self.layers)):
            self.layers[i].clear()

    def deeppoly_QNN(self, qnn_encoding):

        def pre(cur_neuron, i):
            if i == 0:
                cur_neuron.concrete_algebra_lower = deepcopy(cur_neuron.algebra_lower)
                cur_neuron.concrete_algebra_upper = deepcopy(cur_neuron.algebra_upper)

            lower_bound = deepcopy(cur_neuron.algebra_lower)
            upper_bound = deepcopy(cur_neuron.algebra_upper)

            for k in range(i + 1)[::-1]:
                tmp_lower = np.zeros(len(self.layers[k].neurons[0].algebra_lower))
                tmp_upper = np.zeros(len(self.layers[k].neurons[0].algebra_lower))
                assert (self.layers[k].size + 1 == len(lower_bound))
                assert (self.layers[k].size + 1 == len(upper_bound))
                for p in range(self.layers[k].size):
                    if lower_bound[p] >= 0:
                        tmp_lower += lower_bound[p] * self.layers[k].neurons[p].algebra_lower
                    else:
                        tmp_lower += lower_bound[p] * self.layers[k].neurons[p].algebra_upper

                    if upper_bound[p] >= 0:
                        tmp_upper += upper_bound[p] * self.layers[k].neurons[p].algebra_upper
                    else:
                        tmp_upper += upper_bound[p] * self.layers[k].neurons[p].algebra_lower

                tmp_lower[-1] += lower_bound[-1]
                tmp_upper[-1] += upper_bound[-1]
                lower_bound = deepcopy(tmp_lower)
                upper_bound = deepcopy(tmp_upper)
                if k == 1:
                    cur_neuron.concrete_algebra_upper = deepcopy(upper_bound)
                    cur_neuron.concrete_algebra_lower = deepcopy(lower_bound)
            assert (len(lower_bound) == 1)
            assert (len(upper_bound) == 1)


            cur_neuron.concrete_lower = math.floor(lower_bound[0])
            cur_neuron.concrete_upper = math.ceil(upper_bound[0])

            if (cur_neuron.concrete_highest_lower == None) or (
                    cur_neuron.concrete_highest_lower < cur_neuron.concrete_lower):
                cur_neuron.concrete_highest_lower = cur_neuron.concrete_lower
            if (cur_neuron.concrete_lowest_upper == None) or (
                    cur_neuron.concrete_lowest_upper > cur_neuron.concrete_upper):
                cur_neuron.concrete_lowest_upper = cur_neuron.concrete_upper

        self.abs_mode_changed = 0
        self.abs_mode_changed_min = 0
        in_layer_gp = qnn_encoding.input_layer

        for i in range(len(self.layers) - 1):
            gp_layer_count = 0
            pre_layer = self.layers[i]
            cur_layer = self.layers[i + 1]

            pre_neuron_list = pre_layer.neurons
            cur_neuron_list = cur_layer.neurons

            if cur_layer.layer_type == DP_QNN_layer.AFFINE_LAYER:

                if i + 1 == (len(self.layers) - 1):
                    rounding_error = 0
                else:
                    rounding_error = 0.5

                if i > 0:
                    in_layer_gp = qnn_encoding.dense_layers[gp_layer_count - 1]

                accumulator_frac = in_layer_gp.frac_bits + (
                        qnn_encoding.quantization_config["quantization_bits"]
                        - qnn_encoding.quantization_config["int_bits_weights"]
                )

                excessive_bits = accumulator_frac - (
                        qnn_encoding.quantization_config["quantization_bits"]
                        - qnn_encoding.quantization_config["int_bits_activation"]
                )

                scale = 2 ** excessive_bits

                bias_factor = (
                        + qnn_encoding.quantization_config["int_bits_activation"]
                        - qnn_encoding.quantization_config["int_bits_bias"]
                )
                bias_scale = 2 ** bias_factor

                for j in range(cur_layer.size):
                    cur_neuron = cur_neuron_list[j]

                    cur_neuron.algebra_lower = np.append(cur_neuron.weight / scale, [
                        cur_neuron.bias / bias_scale - rounding_error])
                    cur_neuron.algebra_upper = np.append(cur_neuron.weight / scale, [
                        cur_neuron.bias / bias_scale + rounding_error])
                    pre(cur_neuron, i)

                    cur_neuron.concrete_lower_noClip = cur_neuron.concrete_lower
                    cur_neuron.concrete_upper_noClip = cur_neuron.concrete_upper
                gp_layer_count = gp_layer_count + 1

            elif cur_layer.layer_type == DP_QNN_layer.CLAMP_MAX_LAYER:
                for j in range(cur_layer.size):
                    cur_neuron = cur_neuron_list[j]
                    pre_neuron = pre_neuron_list[j]
                    cur_neuron.concrete_lower_noClip = pre_neuron.concrete_lower_noClip
                    cur_neuron.concrete_upper_noClip = pre_neuron.concrete_upper_noClip

                    if pre_neuron.concrete_highest_lower >= 0 or cur_neuron.certain_flag == 1:  # activated
                        cur_neuron.algebra_lower = np.zeros(cur_layer.size + 1)
                        cur_neuron.algebra_upper = np.zeros(cur_layer.size + 1)
                        cur_neuron.algebra_lower[j] = 1
                        cur_neuron.algebra_upper[j] = 1
                        cur_neuron.concrete_algebra_lower = deepcopy(pre_neuron.concrete_algebra_lower)
                        cur_neuron.concrete_algebra_upper = deepcopy(pre_neuron.concrete_algebra_upper)
                        cur_neuron.concrete_lower = pre_neuron.concrete_lower
                        cur_neuron.concrete_upper = pre_neuron.concrete_upper

                        cur_neuron.concrete_highest_lower = pre_neuron.concrete_highest_lower
                        cur_neuron.concrete_lowest_upper = pre_neuron.concrete_lowest_upper
                        cur_neuron.certain_flag = 1
                    elif pre_neuron.concrete_lowest_upper <= 0 or cur_neuron.certain_flag == 2:  # deactivated
                        cur_neuron.algebra_lower = np.zeros(cur_layer.size + 1)
                        cur_neuron.algebra_upper = np.zeros(cur_layer.size + 1)
                        cur_neuron.concrete_algebra_lower = np.zeros(self.inputSize + 1)
                        cur_neuron.concrete_algebra_upper = np.zeros(self.inputSize + 1)
                        cur_neuron.concrete_lower = 0
                        cur_neuron.concrete_upper = 0

                        cur_neuron.concrete_highest_lower = 0
                        cur_neuron.concrete_lowest_upper = 0
                        cur_neuron.certain_flag = 2
                    elif pre_neuron.concrete_highest_lower + pre_neuron.concrete_lowest_upper <= 0:  # mode-b

                        if (cur_neuron.prev_abs_mode != None) and (cur_neuron.prev_abs_mode != 0):
                            self.abs_mode_changed += 1
                        cur_neuron.prev_abs_mode = 0

                        cur_neuron.algebra_lower = np.zeros(cur_layer.size + 1)
                        aux = pre_neuron.concrete_lowest_upper / (
                                pre_neuron.concrete_lowest_upper - pre_neuron.concrete_highest_lower)
                        cur_neuron.algebra_upper = np.zeros(cur_layer.size + 1)
                        cur_neuron.algebra_upper[j] = aux
                        cur_neuron.algebra_upper[-1] = -aux * pre_neuron.concrete_highest_lower
                        pre(cur_neuron, i)
                    else:

                        if (cur_neuron.prev_abs_mode != None) and (cur_neuron.prev_abs_mode != 1):
                            self.abs_mode_changed += 1
                        cur_neuron.prev_abs_mode = 1

                        cur_neuron.algebra_lower = np.zeros(cur_layer.size + 1)
                        cur_neuron.algebra_lower[j] = 1
                        aux = pre_neuron.concrete_lowest_upper / (
                                pre_neuron.concrete_lowest_upper - pre_neuron.concrete_highest_lower)
                        cur_neuron.algebra_upper = np.zeros(cur_layer.size + 1)
                        cur_neuron.algebra_upper[j] = aux
                        cur_neuron.algebra_upper[-1] = -aux * pre_neuron.concrete_highest_lower
                        pre(cur_neuron, i)


            elif cur_layer.layer_type == DP_QNN_layer.CLAMP_MIN_LAYER:
                clamp_upper = 2 ** (qnn_encoding.quantization_config["quantization_bits"]) - 1
                for j in range(cur_layer.size):
                    cur_neuron = cur_neuron_list[j]
                    pre_neuron = pre_neuron_list[j]
                    cur_neuron.concrete_lower_noClip = pre_neuron.concrete_lower_noClip
                    cur_neuron.concrete_upper_noClip = pre_neuron.concrete_upper_noClip

                    if pre_neuron.concrete_lowest_upper <= clamp_upper or cur_neuron.certain_flag == 3:
                        cur_neuron.algebra_lower = np.zeros(cur_layer.size + 1)
                        cur_neuron.algebra_upper = np.zeros(cur_layer.size + 1)
                        cur_neuron.algebra_lower[j] = 1
                        cur_neuron.algebra_upper[j] = 1
                        cur_neuron.concrete_algebra_lower = deepcopy(pre_neuron.concrete_algebra_lower)
                        cur_neuron.concrete_algebra_upper = deepcopy(pre_neuron.concrete_algebra_upper)
                        cur_neuron.concrete_lower = pre_neuron.concrete_lower
                        cur_neuron.concrete_upper = pre_neuron.concrete_upper

                        cur_neuron.concrete_highest_lower = pre_neuron.concrete_highest_lower
                        cur_neuron.concrete_lowest_upper = pre_neuron.concrete_lowest_upper
                        cur_neuron.certain_flag = 3
                    elif pre_neuron.concrete_highest_lower >= clamp_upper or cur_neuron.certain_flag == 4:
                        cur_neuron.algebra_lower = np.zeros(cur_layer.size + 1)
                        cur_neuron.algebra_upper = np.zeros(cur_layer.size + 1)
                        cur_neuron.algebra_lower[-1] = clamp_upper
                        cur_neuron.algebra_upper[-1] = clamp_upper
                        cur_neuron.concrete_algebra_lower = np.zeros(self.inputSize + 1)
                        cur_neuron.concrete_algebra_upper = np.zeros(self.inputSize + 1)
                        cur_neuron.concrete_algebra_lower[-1] = clamp_upper
                        cur_neuron.concrete_algebra_upper[-1] = clamp_upper
                        cur_neuron.concrete_lower = clamp_upper
                        cur_neuron.concrete_upper = clamp_upper

                        cur_neuron.concrete_highest_lower = clamp_upper
                        cur_neuron.concrete_lowest_upper = clamp_upper
                        cur_neuron.certain_flag = 4
                    elif pre_neuron.concrete_highest_lower + pre_neuron.concrete_lowest_upper >= 2 * (clamp_upper - 1):

                        if (cur_neuron.prev_abs_mode_min != None) and (cur_neuron.prev_abs_mode_min != 0):
                            self.abs_mode_changed_min += 1
                        cur_neuron.prev_abs_mode = 0

                        cur_neuron.algebra_upper = np.zeros(cur_layer.size + 1)
                        cur_neuron.algebra_upper[-1] = clamp_upper
                        aux = (clamp_upper - pre_neuron.concrete_highest_lower) / (
                                pre_neuron.concrete_lowest_upper - pre_neuron.concrete_highest_lower)
                        cur_neuron.algebra_lower = np.zeros(cur_layer.size + 1)
                        cur_neuron.algebra_lower[j] = aux
                        cur_neuron.algebra_lower[-1] = (pre_neuron.concrete_lowest_upper - clamp_upper) / (
                                pre_neuron.concrete_lowest_upper - pre_neuron.concrete_highest_lower) * pre_neuron.concrete_highest_lower
                        cur_neuron.certain_flag = 5
                        pre(cur_neuron, i)
                    else:

                        if (cur_neuron.prev_abs_mode_min != None) and (cur_neuron.prev_abs_mode_min != 1):
                            self.abs_mode_changed_min += 1
                        cur_neuron.prev_abs_mode_min = 1

                        cur_neuron.algebra_upper = np.zeros(cur_layer.size + 1)
                        cur_neuron.algebra_upper[j] = 1
                        aux = (clamp_upper - pre_neuron.concrete_highest_lower) / (
                                pre_neuron.concrete_lowest_upper - pre_neuron.concrete_highest_lower)
                        cur_neuron.algebra_lower = np.zeros(cur_layer.size + 1)
                        cur_neuron.algebra_lower[j] = aux
                        cur_neuron.algebra_lower[-1] = (pre_neuron.concrete_lowest_upper - clamp_upper) / (
                                pre_neuron.concrete_lowest_upper - pre_neuron.concrete_highest_lower) * pre_neuron.concrete_highest_lower
                        cur_neuron.certain_flag = 6
                        pre(cur_neuron, i)

            if i == len(self.layers) - 2:
                if self.outputSigned:
                    min_val, max_val = qu.int_get_min_max_integer(
                        qnn_encoding.quantization_config["quantization_bits"],
                        qnn_encoding.quantization_config["quantization_bits"]
                        - qnn_encoding.quantization_config["int_bits_activation"],
                    )
                else:
                    min_val, max_val = qu.uint_get_min_max_integer(
                        qnn_encoding.quantization_config["quantization_bits"],
                        qnn_encoding.quantization_config["quantization_bits"]
                        - qnn_encoding.quantization_config["int_bits_activation"],
                    )

                for j in range(cur_layer.size):
                    cur_neuron = cur_neuron_list[j]
                    cur_neuron.concrete_lower = min(max(cur_neuron.concrete_lower, min_val), max_val)
                    cur_neuron.concrete_upper = min(max(cur_neuron.concrete_upper, min_val), max_val)

    def load_qnn(self, quantized_model):

        layersize = []
        self.layers = []

        # input layer
        layersize.append(quantized_model._input_shape[-1])
        new_in_layer = DP_QNN_layer()
        new_in_layer.layer_type = DP_QNN_layer.INPUT_LAYER
        new_in_layer.size = layersize[-1]
        new_in_layer.neurons = []
        for i in range(layersize[-1]):
            new_neuron = DP_QNN_neuron()
            new_in_layer.neurons.append(new_neuron)
        self.layers.append(new_in_layer)

        numDensLayers = len(quantized_model.dense_layers)
        # dense layer:
        for i, l in enumerate(quantized_model.dense_layers):
            tf_layer = quantized_model.dense_layers[i]
            w, b = tf_layer.get_quantized_weights()
            w = w.T
            layersize.append(l.units)
            # hidden layer, 1. add affine layer 2. add relu layer
            if (i < numDensLayers - 1):
                new_hidden_layer = DP_QNN_layer()
                new_hidden_layer.layer_type = DP_QNN_layer.AFFINE_LAYER
                new_hidden_layer.size = layersize[-1]
                new_hidden_layer.neurons = []
                for k in range(layersize[-1]):
                    new_hidden_neuron = DP_QNN_neuron()
                    new_hidden_neuron.weight = w[k]
                    new_hidden_neuron.bias = b[k]
                    new_hidden_layer.neurons.append(new_hidden_neuron)
                self.layers.append(new_hidden_layer)

                # add clamp_max layer
                new_hidden_layer = DP_QNN_layer()
                new_hidden_layer.layer_type = DP_QNN_layer.CLAMP_MAX_LAYER
                new_hidden_layer.size = layersize[-1]
                new_hidden_layer.neurons = []
                for k in range(layersize[-1]):
                    new_hidden_neuron = DP_QNN_neuron()
                    new_hidden_layer.neurons.append(new_hidden_neuron)
                self.layers.append(new_hidden_layer)

                # add clamp_min layer
                new_hidden_layer = DP_QNN_layer()
                new_hidden_layer.layer_type = DP_QNN_layer.CLAMP_MIN_LAYER
                new_hidden_layer.size = layersize[-1]
                new_hidden_layer.neurons = []
                for k in range(layersize[-1]):
                    new_hidden_neuron = DP_QNN_neuron()
                    new_hidden_layer.neurons.append(new_hidden_neuron)
                self.layers.append(new_hidden_layer)

            else:
                # output layer
                new_out_layer = DP_QNN_layer()
                new_out_layer.layer_type = new_out_layer.AFFINE_LAYER
                new_out_layer.size = layersize[-1]
                new_out_layer.neurons = []
                for k in range(layersize[-1]):
                    new_out_neuron = DP_QNN_neuron()
                    new_out_neuron.weight = w[k]
                    new_out_neuron.bias = b[k]
                    new_out_layer.neurons.append(new_out_neuron)
                self.layers.append(new_out_layer)

        self.layerSizes = layersize
        self.inputSize = layersize[0]
        self.outputSize = layersize[-1]
        self.numLayers = len(layersize) - 1

    def process_interval(self, qnn_encoding):
        for l in self.layers:
            if l.layer_type == DP_QNN_layer.CLAMP_MIN_LAYER:
                clamp_upper = 2 ** (qnn_encoding.quantization_config["quantization_bits"]) - 1
                for neu in l.neurons:
                    neu.concrete_lower = np.clip(neu.concrete_lower, 0,
                                                 clamp_upper)  # min(max(0, neu.concrete_lower), clamp_upper)
                    neu.concrete_upper = np.clip(neu.concrete_upper, 0,
                                                 clamp_upper)  # min(max(0, neu.concrete_upper), clamp_upper)
