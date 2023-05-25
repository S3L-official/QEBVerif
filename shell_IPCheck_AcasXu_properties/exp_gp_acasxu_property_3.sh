#!/bin/bash
cd ..
TO=3600
error=0.05
prop=3

for qu_bit in 4 6 8 10
do
  net=14
  timeout $TO python IPVerifier_AcasXu_properties.py --AcasNet $net --qu_bit $qu_bit --error $error --property $prop --mode Base --relu 0 --outputPath outputIPCheck_AcasXu/qu_$qu_bit"_bit"/
  timeout $TO python IPVerifier_AcasXu_properties.py --AcasNet $net --qu_bit $qu_bit --error $error --property $prop --mode Con --relu 0 --outputPath outputIPCheck_AcasXu/qu_$qu_bit"_bit"/
  timeout $TO python IPVerifier_AcasXu_properties.py --AcasNet $net --qu_bit $qu_bit --error $error --property $prop --mode Sym --relu 0 --outputPath outputIPCheck_AcasXu/qu_$qu_bit"_bit"/

done
