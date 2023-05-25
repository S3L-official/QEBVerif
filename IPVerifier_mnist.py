import argparse
from utils_mnist.quantization_util import *
from utils_mnist.quantized_layers import QuantizedModel
from gurobi_encoding.gurobi_encoding_IP_mnist import *

parser = argparse.ArgumentParser()
parser.add_argument("--sample_id", type=int, default=1)
parser.add_argument("--eps", type=int, default=2)
parser.add_argument("--arch", default="1blk_64")
parser.add_argument("--in_bit", type=int, default=8)
parser.add_argument("--relu", type=int, default=0)
parser.add_argument("--qu_bit", type=int, default=6)
parser.add_argument("--outputPath", default="")
parser.add_argument("--mode", default="Base")  # Base, Con & Sym
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
    check_robustness_gurobi_DRA_only(ilp, x_test[args.sample_id].flatten(), args, original_prediction)
else:
    print("Sample {} is misclassified!".format(args.sample_id))

print("\nNow we finish verifying ...")
