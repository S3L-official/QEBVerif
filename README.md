# QEBVerif
This is the official webpage for paper *QEBVerif: Quantization Error Bound Verification of Neural Networks*. In this paper, we make the following main contributions:
- We introduce the first sound, complete and reasonably efficient quantization error bound verification method QEBVerif for fully QNNs by cleverly combining novel DRA and MILP-based verification methods.
- We propose a novel DRA to compute sound and tight quantization error intervals accompanied by an abstract domain tailored to QNNs, which can significantly and soundly tighten the quantization error intervals.
- We implement QEBVerif as an end-to-end tool and conduct extensive evaluation on various verification tasks, demonstrating its effectiveness and efficiency.


## Setup
Please install gurobypy from PyPI:

```shell script
$ pip install gurobipy
```

For MILP-based solving usage, please install Gurobi on your machine.

## Usage

### Benchmarks in Sections 5.1 & 5.2 & 5.3:

The 30 correctly predicted samples from MNIST dataset by all the 25 networks (shown by IDs):

```
3236 5656 7219 7098 3825  521 9886 7389 3064 4544 
7773 1340 7894 8463 6930 2898 2190 1866  914 5574
3770 3016 6395 2897 8541 5082 3712  150 2486 3986
```
The 5 fix points used in Section 5 for ACAS Xu:

```
1: [0, 0, 0, 0, 0]
2: [0.2, -0.1, 0, -0.3, 0.4]
3: [0.45, -0.23, -0.4, 0.12, 0.33]
4: [-0.2, -0.25, -0.5, -0.3, -0.44]
5: [0.61, 0.36, 0.0, 0.0, -0.24]
```

### Running the DRA on the benchmarks
```shell script
# DRA Mode = Base or Con or Sym

# Running DRA on MNIST: 
# Network=P1, Q=10, Input=3236, Attack=3, ErrorBound=1, Mode=Sym, OutputFolder=./output
python IPVerifier_mnist.py --arch 1blk_100 --sample_id 3236 --eps 3 --qu_bit 10 --error 1 --mode Sym  --outputPath output

# Running DRA on ACAS Xu for fix-points-based input region: 
# Network=A15, Q=10, Fixpoint=4, Attack=0.01 (Floating-point counterpart 3/255=0.01 for the integer attack 3), ErrorBound=0.1, Mode=Sym, OutputFolder=./output
python IPVerifier_AcasXu_fixpoint.py --AcasNet 15 --property 4 --eps 0.01 --qu_bit 10 --error 0.1 --mode Sym --outputPath output

# Running DRA on ACAS Xu for properties-based input region: 
# Network=A15, Q=10, Property=4, ErrorBound=0.1, Mode=Sym, OutputFolder=./output
python IPVerifier_AcasXu_properties.py --AcasNet 15 --property 4 --qu_bit 10 --error 0.1 --mode Sym --outputPath output
```

### Running QEBVerif (DRA+MILP) on the benchmarks
```shell script
# Running DRA+MILP: --ifDiff 0
# Running DRA+MILP+Diff: --ifDiff 1
python DQVerifier_mnist.py --arch 1blk_100 --sample_id 3236 --eps 3 --qu_bit 10 --error 1 --mode Sym --ifDiff 0 --outputPath output
python DQVerifier_AcasXu_fixpoint.py --AcasNet 15 --property 4 --eps 0.01 --qu_bit 10 --error 0.1 --mode Sym --ifDiff 0 --outputPath output
python DQVerifier_AcasXu_properties.py --AcasNet 15 --property 4 --qu_bit 10 --error 0.1 --mode Sym --ifDiff 1 --outputPath output
```

## Publications

If you use QEBVerif, please kindly cite our papers:
- Zhang, Y., Song, F., Sun, J.: QEBVerif: Quantization Error Bound Verification of Neural Networks. In: Proceedings of the 35th International Conference on Computer Aided Verification, 2023.


Some related works (verification of Quantized Neural Networks) can also be found in our other papers:
- Zhang, Y., Zhao, Z., Chen, G., Song, F., Chen, T.: BDD4BNN: A BDD-based quantitative analysis framework for binarized neural networks. In: Proceedings of the 33rd International Conference on Computer Aided Verification, 2021.
- Zhang, Y., Zhao, Z., Chen, G., Song, F., Zhang, M., Chen, T., Sun, J.: QVIP: An ILP-based formal verification approach for quantized neural networks. In: Proceedings of the 37th IEEE/ACM International Conference on Automated Software Engineering, 2022.
- Zhang, Y., Zhao, Z., Chen, G., Song, F., Chen, T.: Precise quantitative analysis of binarized neural networks: A bdd-based approach. ACM Transactions on Software Engineering and Methodology, 2023.
