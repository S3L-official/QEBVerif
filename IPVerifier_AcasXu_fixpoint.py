import argparse
from utils_AcasXu.quantization_util import *
from utils_AcasXu.quantized_layers import QuantizedModel
from gurobi_encoding.gurobi_encoding_IP_AcasXu_FP import *
from gurobipy import GRB
import sys

rootPath = sys.path[0]

bigM = GRB.MAXINT

in_min = [0.0, -3.141593, -3.141593, 100.0, 0.0, 0.0, -3.0]
in_max = [60760.0, 3.141593, 3.141593, 1200.0, 1200.0, 100.0, 3.0]
in_mean = [1.9791091e+04, 0.0, 0.0, 650.0, 600.0, 35.1111111, 0.0, 7.5188840201005975]
in_var = [60261.0, 6.28318530718, 6.28318530718, 1100.0, 1200.0, 100.0, 6.0, 373.94992]

in_ub = [(in_max[i] - in_mean[i]) / in_var[i] for i in range(5)]
in_lb = [(in_min[i] - in_mean[i]) / in_var[i] for i in range(5)]

parser = argparse.ArgumentParser()
parser.add_argument("--AcasNet", type=int, default=1)
parser.add_argument("--property", type=int, default=1)
parser.add_argument("--eps", type=float, default=0.01)
parser.add_argument("--qu_bit", type=int, default=6)
parser.add_argument("--error", type=float, default=0.05)
parser.add_argument("--mode", default="Base")
parser.add_argument("--relu", type=int, default=0)
parser.add_argument("--outputPath", default="")

args = parser.parse_args()

target_cls_map = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

input_min, input_max = int_get_min_max(8, 8)
input_min_int, input_max_int = int_get_min_max_integer(8, 8)

x_input_real_map = {1: [0, 0, 0, 0, 0], 2: [0.2, -0.1, 0, -0.3, 0.4], 3: [0.45, -0.23, -0.4, 0.12, 0.33],
                    4: [-0.2, -0.25, -0.5, -0.3, -0.44], 5: [0.61, 0.36, 0.0, 0.0, -0.24]}

x_input_real = x_input_real_map[args.property]

x_input_ub_real = [np.clip(x_input_real[i] + args.eps, in_lb[i], in_ub[i]) for i in range(5)]
x_input_lb_real = [np.clip(x_input_real[i] - args.eps, in_lb[i], in_ub[i]) for i in range(5)]

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

check_robustness_gurobi_DRA_only(ilp, np.asarray(input_lb), np.asarray(input_ub), args, original_prediction)
