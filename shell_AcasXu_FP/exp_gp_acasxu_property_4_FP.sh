#!/bin/bash
cd ..
TO=3600
prop=4
net=1

for qu_bit in 4 6 8 10
do
  for error in 0.05 0.1 0.2 0.3 0.4
  do
    for eps in 0.01 0.025 0.05
    do
      timeout $TO python DQVerifier_AcasXu_fixpoint.py --AcasNet $net --eps $eps --qu_bit $qu_bit --error $error --property $prop --mode Sym --ifDiff 0 --relu 0 --outputPath output_AcasXu_FP/qu_$qu_bit"_bit"/ > output_AcasXu_FP/qu_$qu_bit"_bit"/out_"net_"$net"_eps_"$eps"_error_"$error"_P2_Sym_ifDiff_0".out
      timeout $TO python DQVerifier_AcasXu_fixpoint.py --AcasNet $net --eps $eps --qu_bit $qu_bit --error $error --property $prop --mode Sym --ifDiff 1 --relu 0 --outputPath output_AcasXu_FP/qu_$qu_bit"_bit"/ > output_AcasXu_FP/qu_$qu_bit"_bit"/out_"net_"$net"_eps_"$eps"_error_"$error"_P2_Sym_ifDiff_1".out
    done
  done
done