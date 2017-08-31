#!/usr/bin/env sh

SPEAKERS="6 7 4 22 21 35 2 19 12 38 24 17 20 5 23 33 11 39 40 9 27 18 34 30 26 29 1 16 37 10"
#SPEAKERS="12 38 24 17 20 5 23 33 11 39 40 9 27 18 34 30 26 29 1 16 37 10"
#SPEAKERS="4 22 21 35"
for i in $SPEAKERS;
do
    echo "$i"
    #python ./authenbase_40_vsa.py finetune $i
    python ./authenbase_40_vsa.py test $i
done

#SPEAKERS="6 2 19 12 38 24 17 20 5 23 33 11 39 40 9 27 18 34 30 26 29 1 16 37 10"
#for i in $SPEAKERS;
#do
    #echo "$i"
    #python ./authenbase_40_vsa.py finetune $i
    #python ./authenbase_40_vsa.py test $i
#done
