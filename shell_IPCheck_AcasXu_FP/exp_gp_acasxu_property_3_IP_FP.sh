#!/bin/bash
cd ..
TO=3600
error=0.05
prop=3
net=1

for qu_bit in 4 6 8 10
do
  for eps in 0.1 0.075 0.05 0.025 0.01
  do
    timeout $TO python IPVerifier_AcasXu_fixpoint.py --AcasNet $net --eps $eps --qu_bit $qu_bit --error $error --property $prop --mode Base --relu 0 --outputPath outputIPCheck_AcasXu_FP/qu_$qu_bit"_bit"/
    timeout $TO python IPVerifier_AcasXu_fixpoint.py --AcasNet $net --eps $eps --qu_bit $qu_bit --error $error --property $prop --mode Con --relu 0 --outputPath outputIPCheck_AcasXu_FP/qu_$qu_bit"_bit"/
    timeout $TO python IPVerifier_AcasXu_fixpoint.py --AcasNet $net --eps $eps --qu_bit $qu_bit --error $error --property $prop --mode Sym --relu 0 --outputPath outputIPCheck_AcasXu_FP/qu_$qu_bit"_bit"/
  done
done