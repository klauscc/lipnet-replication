#!/usr/bin/env sh

SPEAKERS="22 29"
for i in $SPEAKERS;
do
    for idx in 25 50 100 200
    do
        echo $i
        for j in 1 2 3 4 5 6 7 8 9 10
        do
            rm -f ./data/checkpoints/grid_vsa_speaker_*
            python ./visual_speaker_authentication.py train $i $idx
            python ./visual_speaker_authentication.py test $i "${idx}_${j}"
        done
    done
done
