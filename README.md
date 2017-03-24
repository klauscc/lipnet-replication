## lipnet-replication
A replication of Google DeepMind's paper:LipNet: End-to-End Sentence-level Lipreading

using keras framework and tensorflow backend.

###1. preprocess

preprocess the videos to frames and crop it to lips.(preprocessing codes haven't uploaded yet)
The dir structure looks like:

./data/GRID/
    lip/
        s1/
            bbaf2n/
                0.jpg
                1.jpg
                2.jpg
                ...  
                74.jpg
            ...
        s2/
            ...
        ...
    
    alignments/
        s1/align/
                bbaf2n.align
                ...
        s2/align/
                ...
        ...
    
 
Dir `./data/GRID/lip/s{i}/{name}` saves the lip sequence pictures of the video with {name} of person s{i}, and dir `./data/GRID/alignments/s{i}/align/{name}.align` is the alignment file correspondingly.

**data augmention and normalization havent implemented yet. I may finish this part later.**

###2. train

```
python train.py
```

the dataset splition and the network is the same as the Deepmind's paper.
file `gridDataset.py` is the GRID data generator and `model/lipnet.py` is the model.

**the loss is ctc loss, but ctc_decode havent yet implemented. Therefore, when training only loss was presented.**

