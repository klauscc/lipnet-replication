#!/usr/bin/env sh

#SPEAKERS="22 23 24 25 26 27 28 29 30 31 32 33 34"
SPEAKERS="23 24 25 26 27 28 29 30 31 32 33 34"

for i in $SPEAKERS;
do
    echo $i
    python ./visual_speaker_authentication.py train $i 25
    #python ./visual_speaker_authentication.py test $i 0
done
