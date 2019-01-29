### chromNN
We develop a bimodal neural network architecture to model induced in vivo TF binding based on TF's sequence preferences and the preexisting chromatin landscape.The sequence and chromatin sub-networks are weighted by a single dense node to predict genome-wide TF binding.

#### Citation (Update with BioRxiv submission)
Characterizing the sequence and chromatin pre-determinants of induced TF binding with deep bimodal neural networks.

#### Requirements
1. Python2.7
   + All python dependencies can be installed with: pip install -r requirements.txt
2. Keras (Recommended version >= 2.02)
3. Tensorflow (Recommended verion >= v1.8)
4. Seaborn(0.9.0) for plotting (https://seaborn.pydata.org/installing.html)

#### Input Datasets
The model requires three files for training:
1. Training labels (1 training instance per line) [ Filename = Testfile.label]
2. DNA sequence at training windows [ Filename=Testfile.seq ]
3. Tab delimited tag counts for each input chromatin chromatin data track at training windows. [Filename = Testfile.chromtracks] 

#### Usage 
1. Run the sequence network (the bimodal network uses this as a basemodel)
To view help:
python sequence_network.py -h

Required arguments: 

datapath: Prefix to the datafiles (For example: datapath=FILENAME, if the training files are FILENAME.seq, FILENAME.chromtracks and FILENAME.labels) 
datapath_val: Prefix to the validation files
outfile: Outfile, file to print validation metrics

Optional arguments
--batchsize (default=512)
--seqlen (length of input training sequences, default=500)

 

