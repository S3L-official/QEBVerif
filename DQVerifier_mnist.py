import argparse
from utils_mnist.quantization_util import *
from utils_mnist.quantized_layers import QuantizedModel
from gurobi_encoding.gurobi_encoding_DQ_mnist import *
from gurobipy import GRB

bigM = GRB.MAXINT

parser = argparse.ArgumentParser()
parser.add_argument("--sample_id", type=int, default=1)
parser.add_argument("--eps", type=int, default=2)
parser.add_argument("--arch", default="1blk_64")
parser.add_argument("--in_bit", type=int, default=8)
parser.add_argument("--relu", type=int, default=0)
parser.add_argument("--qu_bit", type=int, default=6)
parser.add_argument("--outputPath", default="")
parser.add_argument("--mode", default="Sym")
parser.add_argument("--ifDiff", type=int, default=0)  # 0: without Diff; >=1: with Diff
parser.add_argument("--error", type=float, default=1.0)

args = parser.parse_args()

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

y_train = y_train.flatten()
y_test = y_test.flatten()

x_train = x_train.reshape([-1, 28 * 28]).astype(np.float32)
x_test = x_test.reshape([-1, 28 * 28]).astype(np.float32)

arch = args.arch.split('_')
numBlk = arch[0][:-3]
blkset = list(map(int, arch[1:]))
blkset.append(10)

assert int(numBlk) == len(blkset) - 1

model = QuantizedModel(
    blkset,
    input_bits=args.in_bit,
    quantization_bits=args.qu_bit,
    last_layer_signed=True,
)

weight_path = "benchmark/mnist/PTQ_mnist_{}_in_8_relu_0.h5".format(args.arch)

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.0001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

model.build((None, 28 * 28))

model.load_weights(weight_path)

original_prediction = np.argmax(model.predict(np.expand_dims(x_test[args.sample_id], 0))[0])

if original_prediction == y_test[args.sample_id]:
    ilp = QNNEncoding_gurobi(model, args.relu, args.mode)

    if_SAT, counterexample_QNN, counterexample_DNN, output_QNN, output_DNN = check_robustness_gurobi(
        ilp, x_test[args.sample_id].flatten(), args, original_prediction)
    if if_SAT == True:
        print("\nThe error bound property (f'-f<{}) holds for the input sample {:04d}!".format(args.error,
                                                                                               args.sample_id))
    else:
        print("\nThe error bound property (f'-f<{}) does not hold for the input sample {:04d}!".format(args.error,
                                                                                                       args.sample_id))

        f_DNN = forward_DNN(np.float32(counterexample_QNN / 255), ilp, args.relu)[:10]
        f_QNN = model.predict(np.expand_dims(counterexample_QNN, 0))[0][:10]

        print("\nWe find a counter-example: \n", counterexample_QNN)
        print("\nThe output of DNN:", f_DNN)
        print("\nThe output of QNN: ", f_QNN)

        diff = f_QNN - f_DNN
        print("\nf_QNN-f_DNN: ", diff)
        print("\nThe output difference of the predicted class is: ", diff[original_prediction])
else:
    print("Sample {} is misclassified!".format(args.sample_id))

print("\nNow we finish verifying ...")