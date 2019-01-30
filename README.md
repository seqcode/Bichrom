### iTF: Predicting induced TF binding given DNA sequence and preexisting chromatin data
A bimodal neural network architecture to model induced in vivo TF binding based on TF's sequence preferences and the preexisting chromatin landscape.

### Citation (Update with BioRxiv submission)
Characterizing the sequence and chromatin pre-determinants of induced TF binding with deep bimodal neural networks.

### Requirements
1. Python2.7
   + All python dependencies can be installed with: pip install -r requirements.txt
2. Keras (Recommended version >= 2.02)
3. Tensorflow (Recommended verion >= v1.8)
4. Seaborn(0.9.0) for plotting (https://seaborn.pydata.org/installing.html)

### Input Datasets
The model requires three files for training:
1. Training labels (1 training instance per line) (Format: FILENAME.label)
   - **Description**: Each line in this file will be either 0 or 1.  
2. DNA sequence at training windows (Format: FILENAME.seq)
   - **Description**: Each line should contain the DNA sequence for one window. 
3. Tab delimited tag counts for each input chromatin chromatin data track at training windows.  
(Format: FILENAME.chromtracks)
   - **Description**: Tag counts for each chromatin data track are appended. For example, if we have tag counts over 10 binned windows and 12 chromatin tracks, each line in FILENAME.chromtracks will contain 10 * 12 columns.  

### Usage: Training networks to predict TF binding 
#### Step 1: Run the sequence-only network
To view help:
cd trainNN
python sequence_network.py -h

Required arguments: 
- datapath: Prefix to the datafiles (For example: datapath=FILENAME, if the training files are FILENAME.seq, FILENAME.chromtracks and FILENAME.labels)  
- datapath_val: Prefix to the validation files  
- outfile: Outfile, file to print validation metrics  


Optional arguments
- --batchsize (default=512)  
- --seqlen (length of input training sequences, default=500) 
- --chromsize (The number of input chromatin datasets used as features, default=12)

For example: 
python sequence_network.py --batchsize 400 training_files validation_files outfile

#### Step 2: Run the sequence and chromatin bimodal network  
To view help:  
cd trainNN  
python bimodal_network.py -h

Required (positional) arguments: 
- datapath: Prefix to the datafiles (For example: datapath=FILENAME, if the training files are FILENAME.seq, FILENAME.chromtracks and FILENAME.labels)  
- datapath_val: Prefix to the validation files  
- outfile: Outfile, file to print validation metrics  
- basemodel: The trained sequence model from Step 1.

Optional arguments
- --batchsize (default=400)  
- --seqlen (length of input training sequences, default=500) 
- --chromsize (The number of input chromatin datasets used as features, default=12)

For example: 
python bimodal_network.py --batchsize 400 training_files validation_files outfile basemodel 

### Usage: Interpreting networks that predict TF binding
Interpret the sequence and chromatin contributions at individual loci, as well as the sequence and chromatin drivers of induced TF binding
To view help:  
cd interpretNN  
python interpretNN.py  

Required (positional) arguments:  
- model: A trained model to be interpreted
- datapath: Prefix to the datafiles

Optional arguments:   
- --joint: Extract the joint sequence and chromatin sub-network embeddings
- --sequence: Interpret the sequence features driving network variation

The required input datasets contain FILENAME.seq, FILENAME.chromtracks, FILENAME.labels and FILENAME.bed  
For example:  
python interpretNN.py --joint MODEL FILENAME
