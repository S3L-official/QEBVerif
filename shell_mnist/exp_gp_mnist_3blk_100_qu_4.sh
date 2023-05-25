#!/bin/bash
cd ..
arch=3blk_100_100_100
qu_bit=4
dataset=mnist
TO=3600
epsl=3
for i in 3236 5656 7219 7098 3825 521 9886 7389 3064 4544 7773 1340 7894 8463 6930 2898 2190 1866 914 5574 3770 3016 6395 2897 8541 5082 3712 150 2486 3986
do

  error=1
  timeout $TO python DQVerifier_mnist.py --sample_id $i --eps $epsl --arch $arch --qu_bit $qu_bit --mode Sym --error $error --ifDiff 0 --outputPath output/$arch/qu_$qu_bit"_bit"/ > output/$arch/qu_$qu_bit"_bit"/out_"ep_"$epsl"_ins_"$i"_error_"$error"_01_IP_2_mode_0".out
  timeout $TO python DQVerifier_mnist.py --sample_id $i --eps $epsl --arch $arch --qu_bit $qu_bit --mode Sym --error $error --ifDiff 1 --outputPath output/$arch/qu_$qu_bit"_bit"/ > output/$arch/qu_$qu_bit"_bit"/out_"ep_"$epsl"_ins_"$i"_error_"$error"_01_IP_2_mode_1".out

  error=2
  timeout $TO python DQVerifier_mnist.py --sample_id $i --eps $epsl --arch $arch --qu_bit $qu_bit --mode Sym --error $error --ifDiff 0 --outputPath output/$arch/qu_$qu_bit"_bit"/ > output/$arch/qu_$qu_bit"_bit"/out_"ep_"$epsl"_ins_"$i"_error_"$error"_01_IP_2_mode_0".out
  timeout $TO python DQVerifier_mnist.py --sample_id $i --eps $epsl --arch $arch --qu_bit $qu_bit --mode Sym --error $error --ifDiff 1 --outputPath output/$arch/qu_$qu_bit"_bit"/ > output/$arch/qu_$qu_bit"_bit"/out_"ep_"$epsl"_ins_"$i"_error_"$error"_01_IP_2_mode_1".out

  error=4
  timeout $TO python DQVerifier_mnist.py --sample_id $i --eps $epsl --arch $arch --qu_bit $qu_bit --mode Sym --error $error --ifDiff 0 --outputPath output/$arch/qu_$qu_bit"_bit"/ > output/$arch/qu_$qu_bit"_bit"/out_"ep_"$epsl"_ins_"$i"_error_"$error"_01_IP_2_mode_0".out
  timeout $TO python DQVerifier_mnist.py --sample_id $i --eps $epsl --arch $arch --qu_bit $qu_bit --mode Sym --error $error --ifDiff 1 --outputPath output/$arch/qu_$qu_bit"_bit"/ > output/$arch/qu_$qu_bit"_bit"/out_"ep_"$epsl"_ins_"$i"_error_"$error"_01_IP_2_mode_1".out

  error=6
  timeout $TO python DQVerifier_mnist.py --sample_id $i --eps $epsl --arch $arch --qu_bit $qu_bit --mode Sym --error $error --ifDiff 0 --outputPath output/$arch/qu_$qu_bit"_bit"/ > output/$arch/qu_$qu_bit"_bit"/out_"ep_"$epsl"_ins_"$i"_error_"$error"_01_IP_2_mode_0".out
  timeout $TO python DQVerifier_mnist.py --sample_id $i --eps $epsl --arch $arch --qu_bit $qu_bit --mode Sym --error $error --ifDiff 1 --outputPath output/$arch/qu_$qu_bit"_bit"/ > output/$arch/qu_$qu_bit"_bit"/out_"ep_"$epsl"_ins_"$i"_error_"$error"_01_IP_2_mode_1".out

  error=8
  timeout $TO python DQVerifier_mnist.py --sample_id $i --eps $epsl --arch $arch --qu_bit $qu_bit --mode Sym --error $error --ifDiff 0 --outputPath output/$arch/qu_$qu_bit"_bit"/ > output/$arch/qu_$qu_bit"_bit"/out_"ep_"$epsl"_ins_"$i"_error_"$error"_01_IP_2_mode_0".out
  timeout $TO python DQVerifier_mnist.py --sample_id $i --eps $epsl --arch $arch --qu_bit $qu_bit --mode Sym --error $error --ifDiff 1 --outputPath output/$arch/qu_$qu_bit"_bit"/ > output/$arch/qu_$qu_bit"_bit"/out_"ep_"$epsl"_ins_"$i"_error_"$error"_01_IP_2_mode_1".out

done