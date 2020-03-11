# LSTM

For a simple start to this project before building a generative network, we first train an LSTM on MIDI files from one composer

The LSTM is given sequences of 100 notes and attempts to predict the next note in the sequence

Data are partitioned into training and validation sets according to song so all of the sequences in the validation set are from songs that are not used in the training set at all

