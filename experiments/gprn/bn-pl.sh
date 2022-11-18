#!/bin/bash

for (( approx=0; approx<5; approx++ ))
do
for (( fold=0; fold<4; fold++ ))
do
printf "$approx $fold \n"
python3 bn_gprn.py 3 "$approx" "$fold"
printf "\n"
done
done
printf "\n"
