#!/bin/bash
cd ..
TO=3600
error=0.05
prop=11

for qu_bit in 4 6 8 10
do
  net=1

  error=0.05
  timeout $TO python DQVerifier_AcasXu_properties.py --AcasNet $net --qu_bit $qu_bit --error $error --property $prop --mode Sym --ifDiff 0 --relu 0 --outputPath output_AcasXu/qu_$qu_bit"_bit"/ > output_AcasXu/qu_$qu_bit"_bit"/out_"net_"$net"_error_"$error"_P11_IP_2_mode_0".out
  timeout $TO python DQVerifier_AcasXu_properties.py --AcasNet $net --qu_bit $qu_bit --error $error --property $prop --mode Sym --ifDiff 1 --relu 0 --outputPath output_AcasXu/qu_$qu_bit"_bit"/ > output_AcasXu/qu_$qu_bit"_bit"/out_"net_"$net"_error_"$error"_P11_IP_2_mode_1".out

  error=0.1
  timeout $TO python DQVerifier_AcasXu_properties.py --AcasNet $net --qu_bit $qu_bit --error $error --property $prop --mode Sym --ifDiff 0 --relu 0 --outputPath output_AcasXu/qu_$qu_bit"_bit"/ > output_AcasXu/qu_$qu_bit"_bit"/out_"net_"$net"_error_"$error"_P11_IP_2_mode_0".out
  timeout $TO python DQVerifier_AcasXu_properties.py --AcasNet $net --qu_bit $qu_bit --error $error --property $prop --mode Sym --ifDiff 1 --relu 0 --outputPath output_AcasXu/qu_$qu_bit"_bit"/ > output_AcasXu/qu_$qu_bit"_bit"/out_"net_"$net"_error_"$error"_P11_IP_2_mode_1".out

  error=0.2
  timeout $TO python DQVerifier_AcasXu_properties.py --AcasNet $net --qu_bit $qu_bit --error $error --property $prop --mode Sym --ifDiff 0 --relu 0 --outputPath output_AcasXu/qu_$qu_bit"_bit"/ > output_AcasXu/qu_$qu_bit"_bit"/out_"net_"$net"_error_"$error"_P11_IP_2_mode_0".out
  timeout $TO python DQVerifier_AcasXu_properties.py --AcasNet $net --qu_bit $qu_bit --error $error --property $prop --mode Sym --ifDiff 1 --relu 0 --outputPath output_AcasXu/qu_$qu_bit"_bit"/ > output_AcasXu/qu_$qu_bit"_bit"/out_"net_"$net"_error_"$error"_P11_IP_2_mode_1".out

  error=0.3
  timeout $TO python DQVerifier_AcasXu_properties.py --AcasNet $net --qu_bit $qu_bit --error $error --property $prop --mode Sym --ifDiff 0 --relu 0 --outputPath output_AcasXu/qu_$qu_bit"_bit"/ > output_AcasXu/qu_$qu_bit"_bit"/out_"net_"$net"_error_"$error"_P11_IP_2_mode_0".out
  timeout $TO python DQVerifier_AcasXu_properties.py --AcasNet $net --qu_bit $qu_bit --error $error --property $prop --mode Sym --ifDiff 1 --relu 0 --outputPath output_AcasXu/qu_$qu_bit"_bit"/ > output_AcasXu/qu_$qu_bit"_bit"/out_"net_"$net"_error_"$error"_P11_IP_2_mode_1".out

  error=0.4
  timeout $TO python DQVerifier_AcasXu_properties.py --AcasNet $net --qu_bit $qu_bit --error $error --property $prop --mode Sym --ifDiff 0 --relu 0 --outputPath output_AcasXu/qu_$qu_bit"_bit"/ > output_AcasXu/qu_$qu_bit"_bit"/out_"net_"$net"_error_"$error"_P11_IP_2_mode_0".out
  timeout $TO python DQVerifier_AcasXu_properties.py --AcasNet $net --qu_bit $qu_bit --error $error --property $prop --mode Sym --ifDiff 1 --relu 0 --outputPath output_AcasXu/qu_$qu_bit"_bit"/ > output_AcasXu/qu_$qu_bit"_bit"/out_"net_"$net"_error_"$error"_P11_IP_2_mode_1".out

done
