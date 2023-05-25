from __future__ import division, print_function
import numpy as np
import utils_AcasXu.quantization_util as qu
import time
import sys
import gurobipy as gp
from gurobipy import GRB
from DeepPoly_QNN import *
from DeepPoly_DNN import *


def _renormalize(product, excessive_bits):
    shift_bits = excessive_bits
    residue = product % (2 ** shift_bits)
    c = product // (2 ** shift_bits)
    if residue >= (2 ** (shift_bits - 1)):
        c += 1

    return np.int32(c)


def propagate_dense(in_layer, out_layer, w, b):
    for out_index in range(out_layer.layer_size):
        weight_row = w[:, out_index]
        bias_factor = (
                in_layer.frac_bits
                + in_layer.quantization_config["int_bits_bias"]
                - in_layer.quantization_config["int_bits_weights"]
        )

        bias = np.int32(b[out_index] * (2 ** bias_factor))

        bound_1 = weight_row * in_layer.clipped_lb
        bound_2 = weight_row * in_layer.clipped_ub

        accumulator_lb = np.minimum(bound_1, bound_2).sum() + bias
        accumulator_ub = np.maximum(bound_1, bound_2).sum() + bias

        accumulator_frac = in_layer.frac_bits + (
                in_layer.quantization_config["quantization_bits"]
                - in_layer.quantization_config["int_bits_weights"]
        )

        excessive_bits = accumulator_frac - (
                in_layer.quantization_config["quantization_bits"]
                - in_layer.quantization_config["int_bits_activation"]
        )

        lb = _renormalize(accumulator_lb, excessive_bits)
        ub = _renormalize(accumulator_ub, excessive_bits)

        if out_layer.signed_output:
            min_val, max_val = qu.int_get_min_max_integer(
                out_layer.quantization_config["quantization_bits"],
                out_layer.quantization_config["quantization_bits"]
                - out_layer.quantization_config["int_bits_activation"],
            )
        else:
            min_val, max_val = qu.uint_get_min_max_integer(
                out_layer.quantization_config["quantization_bits"],
                out_layer.quantization_config["quantization_bits"]
                - out_layer.quantization_config["int_bits_activation"],
            )

        clipped_lb = np.clip(lb, min_val, max_val)
        clipped_ub = np.clip(ub, min_val, max_val)

        out_layer.clipped_lb[out_index] = clipped_lb
        out_layer.clipped_ub[out_index] = clipped_ub

        out_layer.lb[out_index] = lb
        out_layer.ub[out_index] = ub


def compute_lb_ub(weight, x_lb, x_ub, bias):
    bound_1 = weight * x_lb
    bound_2 = weight * x_ub
    accumulator_lb = np.minimum(bound_1, bound_2).sum() + bias
    accumulator_ub = np.maximum(bound_1, bound_2).sum() + bias
    return np.float32(accumulator_lb), np.float32(accumulator_ub)


def compute_diff_clamp(lb_dnn_in, ub_dnn_in, lb_qnn_in, ub_qnn_in, lb_diff_in, ub_diff_in, t):
    lb, ub = 0, 0
    if ub_dnn_in <= 0:  # n = A
        lb = np.clip(lb_qnn_in, 0, t)
        ub = np.clip(ub_qnn_in, 0, t)
    elif lb_dnn_in >= 0:  # n = B
        if ub_qnn_in <= t and lb_qnn_in >= 0:  # n = B, n' = B
            lb = lb_diff_in
            ub = ub_diff_in
        elif ub_qnn_in <= 0 or lb_qnn_in >= t:  # n = B, n' = {A, C}
            a = np.clip(lb_qnn_in, 0, t)
            b = np.clip(ub_qnn_in, 0, t)
            lb = a - ub_dnn_in
            ub = b - lb_dnn_in
        elif ub_qnn_in <= t:  # n = B, n' = AB
            lb = max(-ub_dnn_in, lb_diff_in)
            ub = max(-lb_dnn_in, ub_diff_in)
        elif lb_qnn_in >= 0:  # n = B, n' = BC
            lb = min(t - ub_dnn_in, lb_diff_in)
            ub = min(t - lb_dnn_in, ub_diff_in)
        else:  # n = B, n' = ABC
            lb = max(-ub_dnn_in, min(t - ub_dnn_in, lb_diff_in))
            ub = max(-lb_dnn_in, min(t - lb_dnn_in, ub_diff_in))
    else:  # n = AB
        if ub_qnn_in <= t and lb_qnn_in >= 0:  # n = AB, n' = B
            lb = min(lb_qnn_in, lb_diff_in)
            ub = min(ub_qnn_in, ub_diff_in)
        elif lb_qnn_in >= t or ub_qnn_in <= 0:  # n = AB, n' = {A, C}
            a = np.clip(lb_qnn_in, 0, t)
            lb = a - ub_dnn_in
            ub = np.clip(ub_qnn_in, 0, t)
        elif ub_qnn_in <= t:  # n = AB, n' = AB
            lb = max(lb_diff_in, -ub_dnn_in)
            ub = min(ub_diff_in, ub_qnn_in)
            if ub_diff_in <= 0:
                ub = 0
            if lb_diff_in >= 0:
                lb = 0
        elif lb_qnn_in >= 0:  # n = AB, n' = BC
            lb = min(lb_diff_in, lb_qnn_in, t - ub_dnn_in)
            ub = min(ub_diff_in, t)
        else:  # n = AB, n' = ABC
            lb = min(t - ub_dnn_in, 0, max(lb_diff_in, -ub_dnn_in))
            ub = np.clip(ub_diff_in, 0, t)

    raw_lb = np.float32(np.clip(lb_qnn_in, 0, t) - max(ub_dnn_in, 0))
    raw_ub = np.float32(np.clip(ub_qnn_in, 0, t) - max(lb_dnn_in, 0))
    real_lb = np.float32(max(np.float32(lb), raw_lb))
    real_ub = np.float32(min(np.float32(ub), raw_ub))
    return real_lb, real_ub


def compute_diff_relu(lb_dnn_in, ub_dnn_in, lb_qnn_in, ub_qnn_in, lb_diff_in, ub_diff_in):
    raw_lb = lb_qnn_in - ub_dnn_in
    raw_ub = ub_qnn_in - lb_dnn_in
    return np.float32(max(raw_lb, lb_diff_in)), np.float32(min(raw_ub, ub_diff_in))


def propagate_difference(in_layer, out_layer, w, b, w_cont, b_cont, layer_index):
    weight_factor = 2 ** (
            out_layer.quantization_config["quantization_bits"] - out_layer.quantization_config["int_bits_weights"])
    b_fp = b / 2 ** (
            out_layer.quantization_config["quantization_bits"] - out_layer.quantization_config["int_bits_bias"])

    activation_factor = 2 ** (out_layer.quantization_config["quantization_bits"] - out_layer.quantization_config[
        "int_bits_activation"])
    rounding_error = np.float32(0.5 / (activation_factor))
    if not out_layer.signed_output:  # output layer
        min_val, max_val = qu.uint_get_min_max_integer(
            out_layer.quantization_config["quantization_bits"],
            out_layer.quantization_config["quantization_bits"]
            - out_layer.quantization_config["int_bits_activation"],
        )
        assert min_val == 0
        t = max_val / activation_factor
        for out_index in range(out_layer.layer_size):
            weight_int = w[:, out_index]
            weight_fp = weight_int / weight_factor
            weight_cont = w_cont[:, out_index]

            bias_cont = b_cont[out_index]
            bias_fp = b_fp[out_index]

            weight_diff = np.float32(weight_fp - weight_cont)
            bias_diff = np.float32(bias_fp - bias_cont)

            lb_qnn_in = np.float32(out_layer.lb[out_index] / activation_factor)
            ub_qnn_in = np.float32(out_layer.ub[out_index] / activation_factor)

            lb_diff_in_1, ub_diff_in_1 = compute_lb_ub(weight_diff, in_layer.clipped_lb_cont, in_layer.clipped_ub_cont,
                                                       bias_diff)
            lb_diff_in_2, ub_diff_in_2 = compute_lb_ub(weight_fp, in_layer.diff_concrete_lower,
                                                       in_layer.diff_concrete_upper, 0)

            lb_diff_in_sum, ub_diff_in_sum = lb_diff_in_1 + lb_diff_in_2, ub_diff_in_1 + ub_diff_in_2

            lb_diff_in = lb_diff_in_sum - rounding_error
            ub_diff_in = ub_diff_in_sum + rounding_error

            assert lb_diff_in <= ub_diff_in
            out_layer.diff_concrete_lower[out_index], out_layer.diff_concrete_upper[out_index] = compute_diff_clamp(
                out_layer.lb_cont[out_index],
                out_layer.ub_cont[out_index],
                lb_qnn_in, ub_qnn_in, lb_diff_in, ub_diff_in, t)

    else:
        for out_index in range(out_layer.layer_size):
            weight_int = w[:, out_index]
            weight_fp = weight_int / weight_factor
            weight_cont = w_cont[:, out_index]

            bias_cont = b_cont[out_index]
            bias_fp = b_fp[out_index]

            weight_diff = np.float32(weight_fp - weight_cont)
            bias_diff = np.float32(bias_fp - bias_cont)

            lb_qnn_in = np.float32(out_layer.lb[out_index] / activation_factor)
            ub_qnn_in = np.float32(out_layer.ub[out_index] / activation_factor)

            lb_diff_in_1, ub_diff_in_1 = compute_lb_ub(weight_diff, in_layer.clipped_lb_cont, in_layer.clipped_ub_cont,
                                                       bias_diff)
            lb_diff_in_2, ub_diff_in_2 = compute_lb_ub(weight_fp, in_layer.diff_concrete_lower,
                                                       in_layer.diff_concrete_upper, 0)

            lb_diff_in, ub_diff_in = lb_diff_in_1 + lb_diff_in_2, ub_diff_in_1 + ub_diff_in_2

            out_layer.diff_concrete_lower[out_index], out_layer.diff_concrete_upper[out_index] = compute_diff_relu(
                out_layer.lb_cont[out_index],
                out_layer.ub_cont[out_index],
                lb_qnn_in, ub_qnn_in,
                lb_diff_in, ub_diff_in)


class LayerEncoding_gurobi:
    def __init__(
            self,
            layer_size,
            gp_model,
            bit_width,
            frac_bits,
            quantization_config,
            signed_output=False,
            if_last=False,
            if_input=False,
    ):
        self.layer_size = layer_size
        self.bit_width = bit_width
        self.frac_bits = frac_bits
        self.quantization_config = quantization_config
        self.signed_output = signed_output
        self.if_last = if_last
        self.if_input = if_input

        if signed_output:
            if if_last:
                self.gp_vars = [
                    gp_model.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS) for i in range(layer_size)
                ]
            else:
                self.gp_vars = [
                    gp_model.addVar(lb=-GRB.INFINIT, vtype=GRB.INTEGER) for i in range(layer_size)
                ]
            self.gp_DNN_vars = [
                gp_model.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS) for i in range(layer_size)
            ]
        else:
            if if_last:
                print("Now the output layer is signed!")
                exit(0)
            elif if_input:
                self.gp_vars = [
                    gp_model.addVar(lb=-GRB.INFINITY, vtype=GRB.INTEGER) for i in range(layer_size)
                ]
                self.gp_DNN_vars = [
                    gp_model.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS) for i in range(layer_size)
                ]
            else:
                self.gp_vars = [
                    gp_model.addVar(vtype=GRB.INTEGER) for i in range(layer_size)
                ]
                self.gp_DNN_vars = [
                    gp_model.addVar(vtype=GRB.CONTINUOUS) for i in range(layer_size)
                ]

        gp_model.update()

        if self.signed_output:
            min_val, max_val = qu.int_get_min_max_integer(
                self.quantization_config["quantization_bits"],
                self.quantization_config["quantization_bits"]
                - self.quantization_config["int_bits_activation"],
            )
            cont_min_val, cont_max_val = qu.int_get_min_max(self.quantization_config["quantization_bits"],
                                                            self.quantization_config["quantization_bits"] -
                                                            self.quantization_config["int_bits_activation"])
        else:
            min_val, max_val = qu.uint_get_min_max_integer(
                self.quantization_config["quantization_bits"],
                self.quantization_config["quantization_bits"]
                - self.quantization_config["int_bits_activation"],
            )
            cont_min_val, cont_max_val = qu.uint_get_min_max(self.quantization_config["quantization_bits"],
                                                             self.quantization_config["quantization_bits"] -
                                                             self.quantization_config["int_bits_activation"])

        self.clipped_lb = min_val * np.ones(layer_size, dtype=np.int32)
        self.clipped_ub = max_val * np.ones(layer_size, dtype=np.int32)

        self.clipped_lb_cont = cont_min_val * np.ones(layer_size, dtype=np.float32)
        self.clipped_ub_cont = cont_max_val * np.ones(layer_size, dtype=np.float32)

        acc_min, acc_max = qu.int_get_min_max_integer(32, None)

        self.lb = acc_min * np.ones(layer_size, dtype=np.int32)
        self.ub = acc_max * np.ones(layer_size, dtype=np.int32)

        acc_min_cont, acc_max_cont = -GRB.INFINITY, GRB.INFINITY

        self.accumulator_lb_cont = acc_min * np.ones(layer_size, dtype=np.float32)
        self.accumulator_ub_cont = acc_max * np.ones(layer_size, dtype=np.float32)

        self.lb_cont = acc_min_cont * np.ones(layer_size, dtype=np.float32)
        self.ub_cont = acc_max_cont * np.ones(layer_size, dtype=np.float32)

        self.diff_concrete_lower = np.zeros(layer_size, dtype=np.float32)  # list

        self.diff_concrete_upper = np.zeros(layer_size, dtype=np.float32)  # list

    def set_bounds(self, low, high, is_input_layer=False):
        self.lb = low
        self.ub = high

        if is_input_layer:
            min_val, max_val = qu.int_get_min_max_integer(
                self.quantization_config["input_bits"],
                self.quantization_config["input_bits"]
                - self.quantization_config["int_bits_input"],
            )
        elif self.signed_output:
            min_val, max_val = qu.int_get_min_max_integer(
                self.quantization_config["quantization_bits"],
                self.quantization_config["quantization_bits"]
                - self.quantization_config["int_bits_activation"],
            )
        else:
            min_val, max_val = qu.uint_get_min_max_integer(
                self.quantization_config["quantization_bits"],
                self.quantization_config["quantization_bits"]
                - self.quantization_config["int_bits_activation"],
            )

        self.clipped_lb = np.clip(self.lb, min_val, max_val)
        self.clipped_ub = np.clip(self.ub, min_val, max_val)

    def set_bounds_cont(self, low_cont, high_cont, is_input_layer=False):
        self.lb_cont = low_cont
        self.ub_cont = high_cont

        if is_input_layer:
            min_val, max_val = qu.int_get_min_max(
                self.quantization_config["input_bits"],
                self.quantization_config["input_bits"]
                - self.quantization_config["int_bits_input"],
            )
        elif self.signed_output:
            min_val, max_val = -GRB.INFINITY, GRB.INFINITY
        else:
            min_val, max_val = 0, GRB.INFINITY

        self.clipped_lb_cont = np.clip(self.lb_cont, min_val, max_val)
        self.clipped_ub_cont = np.clip(self.ub_cont, min_val, max_val)

    def set_bounds_diff(self, low_cont, high_cont, is_input_layer=False):
        self.diff_concrete_lower = - np.maximum(low_cont, high_cont) / 256
        self.diff_concrete_upper = - np.minimum(low_cont, high_cont) / 256


class QNNEncoding_gurobi:
    def __init__(self, quantized_model, RELUN, mode):
        # print("\n\n******************** IP is: ",IP)
        self.gp_model = gp.Model("qnn_ilp_verifier")
        self.gp_model.Params.IntFeasTol = 1e-9
        self.gp_model.setParam(GRB.Param.Threads, 24)

        self.config = {
            "mode": mode,
            "relu_value": RELUN,
        }

        self._stats = {
            "constant_neurons": 0,
            "linear_neurons": 0,
            "unstable_neurons": 0,
            "reused_expressions": 0,
            "partially_stable_neurons": 0,
            "build_time": 0,
            "smt_sat_time": 0,
            "gp_sat_time": 0,
            "IP_time": 0,
            "encoding_time": 0,
            "total_time": 0,
        }

        self.dense_layers = []
        self.quantized_model = quantized_model

        self._last_layer_signed = quantized_model._last_layer_signed
        self.quantization_config = quantized_model.quantization_config

        self.deepPolyNets_QNN = DP_QNN_network(self._last_layer_signed)
        self.deepPolyNets_DNN = DP_DNN_network(self._last_layer_signed)

        for i, l in enumerate(quantized_model.dense_layers):
            ifLast = False
            if (i == len(quantized_model.dense_layers) - 1):
                ifLast = True
            self.dense_layers.append(
                LayerEncoding_gurobi(
                    layer_size=l.units,
                    gp_model=self.gp_model,
                    bit_width=self.quantization_config["quantization_bits"],
                    frac_bits=self.quantization_config["quantization_bits"]
                              - self.quantization_config["int_bits_activation"],
                    quantization_config=self.quantization_config,
                    signed_output=l.signed_output,
                    if_last=ifLast,
                )
            )

        input_size = quantized_model._input_shape[-1]
        self.input_layer = LayerEncoding_gurobi(
            layer_size=input_size,
            gp_model=self.gp_model,
            bit_width=self.quantization_config["input_bits"],
            frac_bits=self.quantization_config["input_bits"]
                      - self.quantization_config["int_bits_input"],
            quantization_config=self.quantization_config,
            if_input=True,
        )

        self.input_gp_vars = self.input_layer.gp_vars
        self.output_gp_vars = self.dense_layers[-1].gp_vars

        self.input_gp_DNN_vars = self.input_layer.gp_DNN_vars
        self.output_gp_DNN_vars = self.dense_layers[-1].gp_DNN_vars

        self.deepPolyNets_QNN.load_qnn(quantized_model)
        self.deepPolyNets_DNN.load_dnn(quantized_model)

    def propogation(self, mode):

        print("************** DRA mode : ", str(mode), " **************")

        IP_begin_time = time.time()

        self.deepPolyNets_DNN.deeppoly_DNN()

        for i, l in enumerate(self.dense_layers):
            if i == len(self.dense_layers) - 1:
                for out_index in range(l.layer_size):
                    lb_cont = self.deepPolyNets_DNN.layers[-1].neurons[out_index].concrete_lower_noClip
                    ub_cont = self.deepPolyNets_DNN.layers[-1].neurons[out_index].concrete_upper_noClip
                    l.lb_cont[out_index] = lb_cont
                    l.ub_cont[out_index] = ub_cont
            else:
                for out_index in range(l.layer_size):
                    lb_cont = np.float32(
                        self.deepPolyNets_DNN.layers[2 * (i + 1)].neurons[out_index].concrete_lower_noClip)
                    ub_cont = np.float32(
                        self.deepPolyNets_DNN.layers[2 * (i + 1)].neurons[out_index].concrete_upper_noClip)

                    lb_cont_clipped = np.float32(
                        self.deepPolyNets_DNN.layers[2 * (i + 1)].neurons[out_index].concrete_lower)
                    ub_cont_clipped = np.float32(
                        self.deepPolyNets_DNN.layers[2 * (i + 1)].neurons[out_index].concrete_upper)

                    l.lb_cont[out_index] = np.float32(lb_cont)
                    l.ub_cont[out_index] = np.float32(ub_cont)

                    l.clipped_lb_cont[out_index] = max(lb_cont_clipped, 0)
                    l.clipped_ub_cont[out_index] = max(ub_cont_clipped, 0)

        if mode == "Con" or mode == "Base":

            self.propagate_QNN_bounds()

            if mode == "Con":
                self.propagte_diff_bounds()
            else:
                for i, l in enumerate(self.dense_layers):
                    activation_factor = 2 ** (l.quantization_config["quantization_bits"] - l.quantization_config[
                        "int_bits_activation"])
                    if not l.signed_output:  # hidden layer
                        for out_index in range(l.layer_size):
                            l.diff_concrete_lower[out_index] = l.clipped_lb[out_index] / activation_factor - \
                                                               l.clipped_ub_cont[
                                                                   out_index]
                            l.diff_concrete_upper[out_index] = l.clipped_ub[out_index] / activation_factor - \
                                                               l.clipped_lb_cont[
                                                                   out_index]

                    else:
                        for out_index in range(l.layer_size):
                            l.diff_concrete_lower[out_index] = l.lb[out_index] / activation_factor - l.ub_cont[
                                out_index]
                            l.diff_concrete_upper[out_index] = l.ub[out_index] / activation_factor - l.lb_cont[
                                out_index]

        elif mode == "Sym":
            self.deepPolyNets_QNN.deeppoly_QNN(self)
            self.deepPolyNets_QNN.process_interval(self)
            for i, l in enumerate(self.dense_layers):
                if i == len(self.dense_layers) - 1:
                    for out_index in range(l.layer_size):
                        lb = self.deepPolyNets_QNN.layers[-1].neurons[out_index].concrete_lower_noClip
                        ub = self.deepPolyNets_QNN.layers[-1].neurons[out_index].concrete_upper_noClip
                        l.lb[out_index] = lb
                        l.ub[out_index] = ub
                else:
                    for out_index in range(l.layer_size):
                        lb = self.deepPolyNets_QNN.layers[3 * (i + 1)].neurons[out_index].concrete_lower_noClip
                        ub = self.deepPolyNets_QNN.layers[3 * (i + 1)].neurons[out_index].concrete_upper_noClip
                        l.lb[out_index] = lb
                        l.ub[out_index] = ub
                        #
                        lb_clipped = self.deepPolyNets_QNN.layers[3 * (i + 1)].neurons[out_index].concrete_lower
                        ub_clipped = self.deepPolyNets_QNN.layers[3 * (i + 1)].neurons[out_index].concrete_upper
                        l.clipped_lb[out_index] = lb_clipped
                        l.clipped_ub[out_index] = ub_clipped

            self.propagte_diff_bounds_symbolic()

        IP_end_time = time.time()
        self._stats["IP_time"] = IP_end_time - IP_begin_time
        print("DRA time is: ", self._stats["IP_time"])

    def sat(self, args, prediction):
        build_start_time = time.time()

        self.encode(args.ifDiff)

        gp_solving_start_time = time.time()

        print(
            "\n==================================== Now we start do ilp-based solving! ====================================")

        numAllVars = self.gp_model.getAttr("NumVars")
        numIntVars = self.gp_model.getAttr("NumIntVars")
        numBinVars = self.gp_model.getAttr("NumBinVars")
        numConstrs = self.gp_model.getAttr("NumConstrs")
        #
        print("The num of vars: ", str(numAllVars))
        print("The num of numIntVars: ", str(numIntVars))
        print("The num of numBinVars: ", str(numBinVars))
        print("The num of Constraints: ", str(numConstrs))
        self.gp_model.optimize()
        gp_end_time = time.time()

        self._stats["build_time"] = gp_solving_start_time - build_start_time
        self._stats["gp_sat_time"] = gp_end_time - gp_solving_start_time

        print("\n===== ===== ===== =====  DRA takes (seconds): ", str(self._stats["IP_time"]),
              " ===== ===== ===== =====")

        print("\n===== =====  ===== ===== Encoding time is: " + str(
            self._stats["build_time"]) + " ===== =====  ===== =====")
        print("\n===== =====  ===== ===== Gurobi solving time is: " + str(
            self._stats["gp_sat_time"]) + " ===== ===== ===== =====")

        ifgpUNSat = self.gp_model.status == GRB.INFEASIBLE

        if ifgpUNSat:
            vadi = "Property (Error bound is always smaller than r) is True."
        else:
            vadi = "Property (Error bound is always smaller than r) is False."

        errorString = str(args.error).strip('.')
        epsString = str(args.eps).strip('.')

        fo = open(args.outputPath + "/FP_" + str(args.property) + str("_Net_") + str(args.AcasNet) + "_qu_" + str(
            args.qu_bit) + "_eps_" + epsString + "_err_" + errorString + "_mode_" + str(args.mode) + "_ifDiff_" + str(
            args.ifDiff) + "_gp.txt", "w")
        fo.write("The difference bound is [" + str(self.dense_layers[-1].diff_concrete_lower[prediction]) + ", " + str(
            self.dense_layers[-1].diff_concrete_upper[prediction]) + "]\n")
        fo.write("Verification Result: " + str(ifgpUNSat) + "\n")
        fo.write(vadi + "\n")
        fo.write("Input Propagation Time: " + str(self._stats["IP_time"]) + "\n")
        fo.write("Encoding Time: " + str(self._stats["build_time"]) + "\n")
        fo.write("Solving Time: " + str(self._stats["gp_sat_time"]) + "\n")
        fo.write("Total Time: " + str(
            self._stats["IP_time"] + self._stats["build_time"] + self._stats["gp_sat_time"]) + "\n")
        return ifgpUNSat

    def propagte_diff_bounds(self):
        current_layer = self.input_layer

        for i, l in enumerate(self.dense_layers):
            tf_layer = self.quantized_model.dense_layers[i]
            w, b = tf_layer.get_quantized_weights()
            w_cont, b_cont = tf_layer.get_weights()
            propagate_difference(current_layer, l, w, b, w_cont, b_cont, i)
            current_layer = l

    def propagte_diff_bounds_symbolic(self):

        in_layer = self.input_layer

        for i, l in enumerate(self.dense_layers):
            act_factor = 2 ** (l.quantization_config["quantization_bits"] - in_layer.quantization_config[
                "int_bits_activation"])

            min_val, max_val = qu.uint_get_min_max_integer(
                l.quantization_config["quantization_bits"],
                l.quantization_config["quantization_bits"]
                - l.quantization_config["int_bits_activation"],
            )
            t = max_val / act_factor

            if not l.signed_output:

                for out_index in range(l.layer_size):
                    lb_qnn_in = np.float32(l.lb[out_index] / act_factor)
                    ub_qnn_in = np.float32(l.ub[out_index] / act_factor)

                    concrete_lb_diff, concrete_ub_diff = self.compute_diff_in_symbolic(l, i, out_index)

                    l.diff_concrete_lower[out_index], l.diff_concrete_upper[out_index] = compute_diff_clamp(
                        l.lb_cont[out_index],
                        l.ub_cont[out_index],
                        lb_qnn_in, ub_qnn_in, concrete_lb_diff, concrete_ub_diff, t)

                in_layer = l
            else:
                for out_index in range(l.layer_size):
                    lb_qnn_in = np.float32(l.lb[out_index] / act_factor)
                    ub_qnn_in = np.float32(l.ub[out_index] / act_factor)

                    concrete_lb_diff, concrete_ub_diff = self.compute_diff_in_symbolic(l, i, out_index)

                    l.diff_concrete_lower[out_index], l.diff_concrete_upper[out_index] = compute_diff_relu(
                        l.lb_cont[out_index],
                        l.ub_cont[out_index],
                        lb_qnn_in, ub_qnn_in, concrete_lb_diff, concrete_ub_diff)

                in_layer = l

    def compute_diff_in_symbolic(self, out_layer, layer_index, output_index):

        in_factor = self.input_layer.frac_bits
        act_factor = out_layer.frac_bits

        weight_factor = 2 ** (act_factor - in_factor)

        bias_factor = 2 ** (act_factor)

        cur_lb_algebra = self.deepPolyNets_QNN.layers[3 * (layer_index + 1) - 2].neurons[
                             output_index].concrete_algebra_lower / weight_factor
        cur_lb_algebra -= self.deepPolyNets_DNN.layers[2 * (layer_index + 1) - 1].neurons[
            output_index].concrete_algebra_upper
        cur_lb_algebra += (self.deepPolyNets_QNN.layers[3 * (layer_index + 1) - 2].neurons[
                               output_index].concrete_algebra_lower / weight_factor * (-1 / 256))

        bias_lb = self.deepPolyNets_QNN.layers[3 * (layer_index + 1) - 2].neurons[output_index].concrete_algebra_lower[
                      -1] / bias_factor - \
                  self.deepPolyNets_DNN.layers[2 * (layer_index + 1) - 1].neurons[output_index].concrete_algebra_upper[
                      -1]

        cur_ub_algebra = self.deepPolyNets_QNN.layers[3 * (layer_index + 1) - 2].neurons[
                             output_index].concrete_algebra_upper / weight_factor
        cur_ub_algebra -= self.deepPolyNets_DNN.layers[2 * (layer_index + 1) - 1].neurons[
            output_index].concrete_algebra_lower
        cur_ub_algebra += (self.deepPolyNets_QNN.layers[3 * (layer_index + 1) - 2].neurons[
                               output_index].concrete_algebra_upper / weight_factor * (-1 / 256))

        bias_ub = self.deepPolyNets_QNN.layers[3 * (layer_index + 1) - 2].neurons[output_index].concrete_algebra_upper[
                      -1] / bias_factor - \
                  self.deepPolyNets_DNN.layers[2 * (layer_index + 1) - 1].neurons[output_index].concrete_algebra_lower[
                      -1]

        concrete_lb = compute_lb_ub(cur_lb_algebra[:-1], self.input_layer.lb_cont,
                                    self.input_layer.ub_cont, bias_lb)[0]
        concrete_ub = compute_lb_ub(cur_ub_algebra[:-1], self.input_layer.lb_cont,
                                    self.input_layer.ub_cont, bias_ub)[1]

        return np.float32(concrete_lb), np.float32(concrete_ub)

    def propagate_QNN_bounds(self):
        current_layer = self.input_layer

        for i, l in enumerate(self.dense_layers):
            tf_layer = self.quantized_model.dense_layers[i]
            w, b = tf_layer.get_quantized_weights()
            propagate_dense(current_layer, l, w, b)
            current_layer = l

    def encode(self, ifDiff):

        current_layer = self.input_layer

        for i, l in enumerate(self.dense_layers):
            tf_layer = self.quantized_model.dense_layers[i]
            w, b = tf_layer.get_quantized_weights()
            w_cont, b_cont = tf_layer.get_weights()
            self.encode_dense(current_layer, l, w, b, w_cont, b_cont, ifDiff)

            current_layer = l

    def encode_dense(self, in_layer, out_layer, w, b, w_cont, b_cont, ifDiff):

        for out_index in range(out_layer.layer_size):
            weight_row = w[:, out_index]
            weight_cont_row = w_cont[:, out_index]

            bias_factor = (
                    in_layer.frac_bits
                    + self.quantization_config["int_bits_bias"]
                    - self.quantization_config["int_bits_weights"]
            )

            bias = int(b[out_index] * (2 ** bias_factor))

            bias_cont = b_cont[out_index]

            gp_id_var_weight_list = [
                (i, in_layer.gp_vars[i], int(weight_row[i]))
                for i in range(len(in_layer.gp_vars))
            ]

            gp_id_var_DNN_weight_list = [
                (i, in_layer.gp_DNN_vars[i], weight_cont_row[i])
                for i in range(len(in_layer.gp_DNN_vars))
            ]

            gp_accumulator = self.gp_reduce_MAC(gp_id_var_weight_list)

            gp_DNN_accumulator = self.gp_DNN_reduce_MAC(gp_id_var_DNN_weight_list)

            gp_accumulator = gp_accumulator + bias
            gp_DNN_accumulator = gp_DNN_accumulator + bias_cont

            accumulator_frac = in_layer.frac_bits + (
                    self.quantization_config["quantization_bits"]
                    - self.quantization_config["int_bits_weights"]
            )

            excessive_bits = accumulator_frac - (
                    self.quantization_config["quantization_bits"]
                    - self.quantization_config["int_bits_activation"]
            )
            activation_Factor = 2 ** (self.quantization_config["quantization_bits"]
                                      - self.quantization_config["int_bits_activation"])

            if ifDiff > 0:
                if not out_layer.if_last:
                    self.gp_model.addConstr(
                        out_layer.gp_vars[out_index] / activation_Factor - out_layer.gp_DNN_vars[out_index] >=
                        out_layer.diff_concrete_lower[out_index])
                    self.gp_model.addConstr(
                        out_layer.gp_vars[out_index] / activation_Factor - out_layer.gp_DNN_vars[out_index] <=
                        out_layer.diff_concrete_upper[out_index])

            self.gp_QNN_renormalize(
                gp_value=gp_accumulator,
                lb=out_layer.lb[out_index],
                ub=out_layer.ub[out_index],
                gp_output_var=out_layer.gp_vars[out_index],
                signed_output=out_layer.signed_output,
                excessive_bits=excessive_bits,
                if_last=out_layer.if_last,
            )
            self.gp_DNN_renormalize(
                gp_DNN_value=gp_DNN_accumulator,
                lb_cont=out_layer.lb_cont[out_index],
                ub_cont=out_layer.ub_cont[out_index],
                gp_output_DNN_var=out_layer.gp_DNN_vars[out_index],
                signed_output=out_layer.signed_output,
                if_last=out_layer.if_last,
            )

    def gp_reduce_MAC(self, id_var_weight_list):
        if len(id_var_weight_list) == 1:
            i, x, weight_value = id_var_weight_list[0]
            if weight_value == 0:
                return 0

            if weight_value == 1:
                return x
            elif weight_value == -1:
                return -x
            else:
                return x * weight_value
        else:
            center = len(id_var_weight_list) // 2

            left = self.gp_reduce_MAC(id_var_weight_list[:center])
            right = self.gp_reduce_MAC(id_var_weight_list[center:])
            if left is None:
                return right
            elif right is None:
                return left
            return left + right

    def gp_DNN_reduce_MAC(self, id_var_weight_list):
        if len(id_var_weight_list) == 1:
            i, x, weight_value = id_var_weight_list[0]
            return x * weight_value
        else:
            center = len(id_var_weight_list) // 2

            left = self.gp_DNN_reduce_MAC(id_var_weight_list[:center])
            right = self.gp_DNN_reduce_MAC(id_var_weight_list[center:])
            if left is None:
                return right
            elif right is None:
                return left
            return left + right

    def gp_QNN_renormalize(self, gp_value, lb, ub, gp_output_var, signed_output, excessive_bits, if_last):
        assert lb <= ub

        if if_last:
            all_bit = excessive_bits + self.quantization_config["quantization_bits"] - self.quantization_config[
                "int_bits_activation"]
            self.gp_model.addConstr(gp_output_var == (gp_value / (2 ** all_bit)))
            # self.gp_model.addConstr(gp_output_var <= (gp_value / (2 ** excessive_bits)) + 0.5)
            # self.gp_model.addConstr(gp_output_var >= (gp_value / (2 ** excessive_bits)) - 0.5 + 0.0001)
            return

        if signed_output:
            min_val, max_val = qu.int_get_min_max_integer(
                self.quantization_config["quantization_bits"],
                self.quantization_config["quantization_bits"]
                - self.quantization_config["int_bits_activation"],
            )
        else:
            min_val, max_val = qu.uint_get_min_max_integer(
                self.quantization_config["quantization_bits"],
                self.quantization_config["quantization_bits"]
                - self.quantization_config["int_bits_activation"],
            )

        clipped_lb = np.clip(lb, min_val, max_val)
        clipped_ub = np.clip(ub, min_val, max_val)

        if clipped_ub == clipped_lb:  # A or C
            self.gp_model.addConstr(gp_output_var == int(clipped_ub))
            self._stats["constant_neurons"] += 1
            return

        bigM = max(abs(lb), abs(ub)) + 1

        if lb >= min_val and ub <= max_val:  # B
            self.gp_model.addConstr(gp_output_var <= (gp_value / (2 ** excessive_bits)) + 0.5)
            self.gp_model.addConstr(gp_output_var >= (gp_value / (2 ** excessive_bits)) - 0.5 + 0.0001)
            return
        elif ub <= max_val:  # AB
            gp_var_new_int = self.gp_model.addVar(lb=lb, ub=ub, vtype=GRB.INTEGER)
            gp_var_bool = self.gp_model.addVar(vtype=GRB.BINARY)
            self.gp_model.update()
            self.gp_model.addConstr(gp_var_new_int <= gp_value / (2 ** excessive_bits) + 0.5)
            self.gp_model.addConstr(gp_var_new_int >= gp_value / (2 ** excessive_bits) - 0.5 + 0.0001)

            self.gp_model.addConstr(gp_output_var >= min_val)
            self.gp_model.addConstr(gp_output_var >= gp_var_new_int)
            self.gp_model.addConstr(gp_output_var <= min_val + bigM * gp_var_bool)
            self.gp_model.addConstr(gp_output_var <= gp_var_new_int + bigM * (1 - gp_var_bool))
        elif lb >= min_val:  # BC
            gp_var_new_int = self.gp_model.addVar(lb=lb, ub=ub, vtype=GRB.INTEGER)
            gp_var_bool = self.gp_model.addVar(vtype=GRB.BINARY)
            self.gp_model.update()
            self.gp_model.addConstr(gp_var_new_int <= gp_value / (2 ** excessive_bits) + 0.5)
            self.gp_model.addConstr(gp_var_new_int >= gp_value / (2 ** excessive_bits) - 0.5 + 0.0001)

            self.gp_model.addConstr(gp_output_var <= max_val)
            self.gp_model.addConstr(gp_output_var <= gp_var_new_int)
            self.gp_model.addConstr(gp_output_var >= max_val - bigM * gp_var_bool)
            self.gp_model.addConstr(gp_output_var >= gp_var_new_int - bigM * (1 - gp_var_bool))

        else:
            gp_var_new_int = self.gp_model.addVar(lb=lb, ub=ub, vtype=GRB.INTEGER)
            gp_output_var_min = self.gp_model.addVar(lb=lb, ub=ub, vtype=GRB.INTEGER)

            self.gp_model.update()
            self.gp_model.addConstr(gp_var_new_int <= gp_value / (2 ** excessive_bits) + 0.5)
            self.gp_model.addConstr(gp_var_new_int >= gp_value / (2 ** excessive_bits) - 0.5 + 0.0001)
            #
            gp_var_relu_bool = self.gp_model.addVar(vtype=GRB.BINARY)
            gp_var_relu_N_bool = self.gp_model.addVar(vtype=GRB.BINARY)

            self.gp_model.addConstr(gp_output_var_min >= min_val)
            self.gp_model.addConstr(gp_output_var_min >= gp_var_new_int)
            self.gp_model.addConstr(gp_output_var_min <= min_val + bigM * gp_var_relu_bool)
            self.gp_model.addConstr(gp_output_var_min <= gp_var_new_int + bigM * (1 - gp_var_relu_bool))
            #
            self.gp_model.addConstr(gp_output_var <= max_val)
            self.gp_model.addConstr(gp_output_var <= gp_output_var_min)
            self.gp_model.addConstr(gp_output_var >= max_val - bigM * gp_var_relu_N_bool)
            self.gp_model.addConstr(gp_output_var >= gp_output_var_min - bigM * (1 - gp_var_relu_N_bool))

    def gp_DNN_renormalize(self, gp_DNN_value, lb_cont, ub_cont,
                           gp_output_DNN_var, signed_output, if_last):

        assert lb_cont <= ub_cont

        if if_last:
            self.gp_model.addConstr(gp_output_DNN_var == gp_DNN_value)
            return

        if signed_output:
            min_val_cont, max_val_cont = qu.int_get_min_max(
                self.quantization_config["quantization_bits"],
                self.quantization_config["quantization_bits"]
                - self.quantization_config["int_bits_activation"],
            )
            exit(0)
        else:
            min_val_cont, max_val_cont = qu.uint_get_min_max(
                self.quantization_config["quantization_bits"],
                self.quantization_config["quantization_bits"]
                - self.quantization_config["int_bits_activation"],
            )

        if lb_cont >= min_val_cont:
            self.gp_model.addConstr(gp_output_DNN_var == gp_DNN_value)
        else:
            bigM = max(abs(lb_cont), abs(ub_cont)) + 1
            gp_var_relu_bool = self.gp_model.addVar(vtype=GRB.BINARY)
            self.gp_model.update()
            self.gp_model.addConstr(gp_output_DNN_var >= min_val_cont)
            self.gp_model.addConstr(gp_output_DNN_var >= gp_DNN_value)
            self.gp_model.addConstr(gp_output_DNN_var <= min_val_cont + bigM * gp_var_relu_bool)
            self.gp_model.addConstr(gp_output_DNN_var <= gp_DNN_value + bigM * (1 - gp_var_relu_bool))

    def assert_input_box(self, x_lb, x_ub, mode):
        low, high = x_lb, x_ub
        value_bits = self.quantization_config["input_bits"] - self.quantization_config["int_bits_input"]

        assert value_bits == 8

        low_cont, high_cont = np.float32(x_lb / 255), np.float32(x_ub / 255)

        input_size = len(self.input_gp_vars)

        low = np.array(low, dtype=np.int32) * np.ones(input_size, dtype=np.int32)
        high = np.array(high, dtype=np.int32) * np.ones(input_size, dtype=np.int32)

        low_cont = np.array(low_cont, dtype=np.float32) * np.ones(input_size, dtype=np.float32)
        high_cont = np.array(high_cont, dtype=np.float32) * np.ones(input_size, dtype=np.float32)

        saturation_min, saturation_max = qu.int_get_min_max_integer(
            self.quantization_config["input_bits"],
            self.quantization_config["input_bits"]
            - self.quantization_config["int_bits_input"],
        )

        saturation_min_fp, saturation_max_fp = qu.int_get_min_max(
            self.quantization_config["input_bits"],
            self.quantization_config["input_bits"]
            - self.quantization_config["int_bits_input"],
        )

        low = np.clip(low, saturation_min, saturation_max)
        high = np.clip(high, saturation_min, saturation_max)

        low_cont = np.float32(np.clip(low_cont, saturation_min_fp, saturation_max_fp))
        high_cont = np.float32(np.clip(high_cont, saturation_min_fp, saturation_max_fp))

        self.input_layer.set_bounds(low, high, is_input_layer=True)
        self.input_layer.set_bounds_cont(low_cont, high_cont, is_input_layer=True)
        self.input_layer.set_bounds_diff(low_cont, high_cont, is_input_layer=True)

        for i in range(input_size):
            self.gp_model.addConstr(self.input_layer.gp_vars[i] >= int(low[i]))
            self.gp_model.addConstr(self.input_layer.gp_DNN_vars[i] == self.input_layer.gp_vars[i] / 255)
            self.gp_model.addConstr(self.input_layer.gp_vars[i] <= int(high[i]))

        self.deepPolyNets_DNN.property_region = 1

        for i in range(self.deepPolyNets_DNN.layerSizes[0]):
            self.deepPolyNets_DNN.layers[0].neurons[i].concrete_lower = low_cont[i]
            self.deepPolyNets_DNN.layers[0].neurons[i].concrete_upper = high_cont[i]
            self.deepPolyNets_DNN.property_region *= (high_cont[i] - low_cont[i])
            self.deepPolyNets_DNN.layers[0].neurons[i].concrete_algebra_lower = np.array([low_cont[i]])
            self.deepPolyNets_DNN.layers[0].neurons[i].concrete_algebra_upper = np.array([high_cont[i]])
            self.deepPolyNets_DNN.layers[0].neurons[i].algebra_lower = np.array([low_cont[i]])
            self.deepPolyNets_DNN.layers[0].neurons[i].algebra_upper = np.array([high_cont[i]])

        if mode == "Sym":
            self.deepPolyNets_QNN.property_region = 1

            for i in range(self.deepPolyNets_QNN.layerSizes[0]):
                self.deepPolyNets_QNN.layers[0].neurons[i].concrete_lower = int(low[i])
                self.deepPolyNets_QNN.layers[0].neurons[i].concrete_upper = int(high[i])
                self.deepPolyNets_QNN.property_region *= (int(high[i]) - int(low[i]))
                self.deepPolyNets_QNN.layers[0].neurons[i].concrete_algebra_lower = np.array([int(low[i])])
                self.deepPolyNets_QNN.layers[0].neurons[i].concrete_algebra_upper = np.array([int(high[i])])
                self.deepPolyNets_QNN.layers[0].neurons[i].algebra_lower = np.array([int(low[i])])
                self.deepPolyNets_QNN.layers[0].neurons[i].algebra_upper = np.array([int(high[i])])

    def error_bound_gurobi_absoluateValue(self, err, ifDiff, max_index):
        val_bits = self.quantization_config["quantization_bits"] - self.quantization_config["int_bits_activation"]
        factor = 2 ** val_bits

        diff = self.output_gp_DNN_vars[max_index] - self.output_gp_vars[max_index]

        diff_bool = self.gp_model.addVar(vtype=GRB.BINARY)

        if ifDiff == 0:
            bigM_var = max(0, abs(
                self.dense_layers[-1].lb[max_index] / factor - self.dense_layers[-1].ub_cont[max_index]))
            bigM_var = max(bigM_var, abs(
                self.dense_layers[-1].ub[max_index] / factor - self.dense_layers[-1].lb_cont[max_index]))
            bigM = bigM_var + 10
        else:
            bigM_var = max(abs(self.dense_layers[-1].diff_concrete_lower[max_index]),
                           abs(self.dense_layers[-1].diff_concrete_upper[max_index]))
            bigM = bigM_var + 10

        diff_max = self.gp_model.addVar(lb=-bigM_var, ub=bigM_var, vtype=GRB.CONTINUOUS)
        self.gp_model.update()

        self.gp_model.addConstr(diff_max >= 0)
        self.gp_model.addConstr(diff_max >= diff)
        self.gp_model.addConstr(diff_max <= bigM * diff_bool)
        self.gp_model.addConstr(diff_max <= diff + bigM * (1 - diff_bool))

        self.gp_model.addConstr(2 * diff_max - diff >= err)

    def get_input_assignment(self):
        input_values = np.array(
            [qu.binary_str_to_uint(var.assignment) for var in self.input_layer.vars]
        )
        return np.array(input_values, dtype=np.int32)

    def get_input_assignment_gurobi(self):
        input_values_gp = np.array(
            [var.X for var in self.input_layer.gp_vars]
        )
        input_values_gp_DNN = np.array(
            [var.X for var in self.input_layer.gp_DNN_vars]
        )
        output_values_gp = np.array(
            [var.X for var in self.dense_layers[-1].gp_vars]
        )
        output_values_gp_DNN = np.array(
            [var.X for var in self.dense_layers[-1].gp_DNN_vars]
        )
        return np.array(input_values_gp, dtype=np.int32), np.array(input_values_gp_DNN, dtype=np.float32), np.array(
            output_values_gp, dtype=np.float32), np.array(output_values_gp_DNN, dtype=np.float32)


def check_robustness_gurobi(encoding, x_lb, x_ub, y, args, original_prediction):
    path = args.outputPath

    print(
        "\n============================== Begin to encode input region & output property ==============================\n")

    encoding.assert_input_box(x_lb, x_ub, args.mode)

    encoding.propogation(args.mode)

    target_diff_lb = encoding.dense_layers[-1].diff_concrete_lower[original_prediction]
    target_diff_ub = encoding.dense_layers[-1].diff_concrete_upper[original_prediction]

    print("The difference bound is [", target_diff_lb, ", ", target_diff_ub, "].")

    if abs(target_diff_lb) <= args.error and abs(target_diff_ub) <= args.error:
        print("We directly verified and the property hold!!!")
        fo = open(path + "/FP_" + str(args.property) + "_Net_" + str(args.AcasNet) + "_qu_" + str(
            args.qu_bit) + "_eps_" + str(args.eps) + "_err_" + str(args.error) + "_mode_" + str(
            args.mode) + "_ifDiff_" + str(args.ifDiff) + "_gp.txt", "w")
        fo.write("The difference bound is [" + str(target_diff_lb) + ", " + str(target_diff_ub) + "].\n")
        fo.write("Verification Result: True\n")
        fo.write("Property (Error bound is always smaller than r) is True.\n")
        fo.write("Input Propagation Time: " + str(encoding._stats["IP_time"]) + "\n")
        fo.write("Encoding Time: 0\n")
        fo.write("Solving Time: 0\n")
        fo.write("Total Time: " + str(encoding._stats["IP_time"]) + "\n")
        exit(0)
    else:
        print("We can not directly verified via DRA!!!")
        print("\n============================== Begin to encode ILP ==============================\n")

    encoding.error_bound_gurobi_absoluateValue(args.error, args.ifDiff, int(y))

    print(
        "\n============================== Begin to encode the QNN semantics ==============================\n")

    ifUNSat = encoding.sat(args, original_prediction)
    if not ifUNSat:  #
        attack_QNN, attack_DNN, output_QNN, output_DNN = encoding.get_input_assignment_gurobi()
        return (False, attack_QNN, attack_DNN, output_QNN, output_DNN)
    else:
        return (True, None, None, None, None)
