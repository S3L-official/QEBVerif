import argparse
from utils_AcasXu.quantization_util import *
from utils_AcasXu.quantized_layers import QuantizedModel
from gurobi_encoding.gurobi_encoding_DQ_AcasXu import *
from gurobipy import GRB
import sys

rootPath = sys.path[0]

bigM = GRB.MAXINT

in_min = [0.0, -3.141593, -3.141593, 100.0, 0.0, 0.0, -3.0]
in_max = [60760.0, 3.141593, 3.141593, 1200.0, 1200.0, 100.0, 3.0]
in_mean = [1.9791091e+04, 0.0, 0.0, 650.0, 600.0, 35.1111111, 0.0, 7.5188840201005975]
in_var = [60261.0, 6.28318530718, 6.28318530718, 1100.0, 1200.0, 100.0, 6.0, 373.94992]

parser = argparse.ArgumentParser()
parser.add_argument("--AcasNet", type=int, default=1)
parser.add_argument("--property", type=int, default=1)
parser.add_argument("--qu_bit", type=int, default=6)
parser.add_argument("--error", type=float, default=0.05)
parser.add_argument("--mode", default="Sym")
parser.add_argument("--ifDiff", type=int, default=0)
parser.add_argument("--relu", type=int, default=0)
parser.add_argument("--outputPath", default="")

args = parser.parse_args()

target_cls_map = {1: 0, 2: 0, 3: 0, 4: 0, 5: 4, 16: 0, 26: 0, 7: 4, 8: 1, 9: 3, 10: 0, 11: 4, 12: 0, 13: 0, 14: 4,
                  15: 3}

inputRegion_ub_map = {1: [60760, 3.141592, 3.141592, 1200, 60], 2: [60760, 3.141592, 3.141592, 1200, 60],
                      3: [1800, 0.06, 3.141592, 1200, 1200], 4: [1800, 0.06, 0, 1200, 800],
                      5: [400, 0.4, -3.141592 + 0.005, 400, 400], 16: [62000, -0.7, -3.141592 + 0.005, 200, 1200],
                      26: [62000, 3.141592, -3.141592 + 0.005, 200, 1200], 7: [60760, 3.141592, 3.141592, 1200, 1200],
                      8: [60760, -3.141592 * 0.75, 0.1, 1200, 1200], 9: [7000, 3.141592, -3.141592 + 0.01, 150, 150],
                      10: [60760, 3.141592, -3.141592 + 0.01, 1200, 1200], 11: [400, 0.4, -3.1415926 + 0.005, 400, 400],
                      12: [60760, 3.141592, 3.141592, 1200, 60], 13: [60760, 3.141592, 3.141592, 360, 360],
                      14: [400, 0.4, -3.141592 + 0.005, 400, 400], 15: [400, -0.2, -3.141592 + 0.005, 400, 400]}  #

inputRegion_lb_map = {1: [55947.691, -3.141592, -3.141592, 1145, 0], 2: [55947.691, -3.141592, -3.141592, 1145, 0],
                      3: [1500, -0.06, 3.10, 980, 960], 4: [1500, -0.06, 0, 1000, 700],
                      5: [250, 0.2, -3.141592, 100, 0], 16: [12000, -3.141592, -3.141592, 100, 0],
                      26: [12000, 0.7, -3.141592, 100, 0], 7: [0, -3.141592, -3.141592, 100, 0],
                      8: [0, -3.141592, -0.1, 600, 600], 9: [2000, 0.7, -3.141592, 100, 0],
                      10: [36000, 0.7, -3.141592, 900, 600], 11: [250, 0.2, -3.1415926, 100, 0],
                      12: [55947.691, -3.141592, -3.141592, 1145, 0], 13: [60000, -3.141592, -3.141592, 0, 0],
                      14: [250, 0.2, -3.141592, 100, 0], 15: [250, -0.4, -3.141592, 100, 0]}  #

input_min, input_max = int_get_min_max(8, 8)
input_min_int, input_max_int = int_get_min_max_integer(8, 8)

x_input_ub = inputRegion_ub_map[args.property]
x_input_lb = inputRegion_lb_map[args.property]

x_input_ub_real = [(x_input_ub[0] - in_mean[0]) / in_var[0], (x_input_ub[1] - in_mean[1]) / in_var[1],
                   (x_input_ub[2] - in_mean[2]) / in_var[2],
                   (x_input_ub[3] - in_mean[3]) / in_var[3],
                   (x_input_ub[4] - in_mean[4]) / in_var[4]]

x_input_lb_real = [(x_input_lb[0] - in_mean[0]) / in_var[0], (x_input_lb[1] - in_mean[1]) / in_var[1],
                   (x_input_lb[2] - in_mean[2]) / in_var[2],
                   (x_input_lb[3] - in_mean[3]) / in_var[3],
                   (x_input_lb[4] - in_mean[4]) / in_var[4]]

input_ub = [np.clip(real_round(x_i * 256), input_min_int, input_max_int) for x_i in x_input_ub_real]
input_lb = [np.clip(real_round(x_i * 256), input_min_int, input_max_int) for x_i in x_input_lb_real]

model = QuantizedModel(
    [50, 50, 50, 50, 50, 50, 5],
    input_bits=8,
    quantization_bits=args.qu_bit,
    last_layer_signed=True,
)

i_index = args.AcasNet // 9
j_index = args.AcasNet - i_index * 9

if j_index == 0:
    weight_path = "benchmark/acasxu/ACASXU_run2a_{}_{}_batch_2000.h5".format(i_index, 9)

else:
    weight_path = "benchmark/acasxu/ACASXU_run2a_{}_{}_batch_2000.h5".format(i_index + 1, j_index)

original_prediction = target_cls_map[args.property]

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.0001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

model.build((None, 1 * 5))
model.load_weights(weight_path)

ilp = QNNEncoding_gurobi(model, args.relu, args.mode)
output_factor = 2 ** (model.quantization_config["quantization_bits"] - model.quantization_config["int_bits_activation"])

if_SAT, counterexample_QNN, counterexample_DNN, output_QNN, output_DNN = check_robustness_gurobi(
    ilp, np.asarray(input_lb), np.asarray(input_ub), original_prediction, args, original_prediction)

if if_SAT == True:
    print("\nThe error bound property (f'-f<{}) holds for the property {:04d}!".format(args.error,
                                                                                       args.property))

else:
    print("\nThe error bound property (f'-f<{}) does not hold for the property {:04d}!".format(args.error,
                                                                                               args.property))

    f_DNN = forward_DNN(np.float32(counterexample_QNN / 255), ilp, args.relu)[:10]
    f_QNN = model.predict(np.expand_dims(counterexample_QNN, 0))[0][:10]

    print("\nWe find a counter-example: \n", counterexample_QNN)
    print("\nThe output of DNN:", f_DNN)
    print("\nThe output of QNN: ", f_QNN)

    diff = f_QNN - f_DNN
    print("\nf_QNN-f_DNN: ", diff)
    print("\nThe output difference of the predicted class is: ", diff[original_prediction])

print("\nNow we finish verifying ...")