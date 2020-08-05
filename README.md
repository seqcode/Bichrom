### Bichrom: A bimodal neural network to predict TF binding using sequence and pre-existing chromatin track data
Transcription factor (TF) binding specificity is determined via a complex interplay between the TF’s DNA binding preference and cell type-specific chromatin environments. The chromatin features that correlate with TF binding in a given cell type have been well characterized. For instance, the binding sites for a majority of TFs display concurrent chromatin accessibility. However, concurrent chromatin features reflect the binding activities of the TF itself, and thus provide limited insight into how genome-wide TF-DNA binding patterns became established in the first place. To understand the determinants of TF binding specificity, we therefore need to examine how newly activated TFs interact with sequence and preexisting chromatin landscapes.

Here, we investigate the sequence and preexisting chromatin predictors of TF-DNA binding by examining the genome-wide occupancy of TFs that have been induced in well-characterized chromatin environments. We develop Bichrom, a bimodal neural network that jointly models sequence and preexisting chromatin data to interpret the genome-wide binding patterns of induced TFs. We find that the preexisting chromatin landscape is a differential global predictor of TF-DNA binding; incorporating preexisting chromatin features improves our ability to explain the binding specificity of some TFs substantially, but not others. Furthermore, by analyzing site-level predictors, we show that TF binding in previously inaccessible chromatin tends to correspond to the presence of more favorable cognate DNA sequences. Bichrom thus provides a framework for modeling, interpreting, and visualizing the joint sequence and chromatin landscapes that determine TF-DNA binding dynamics.

#### bioRxiv ID
Srivastava, D., Aydin, B., Mazzoni, E.O. and Mahony, S., 2020. An interpretable bimodal neural network characterizes the sequence and preexisting chromatin predictors of induced TF binding. bioRxiv, p.672790.

#### Download and train Bichrom
cd trainNN  
To view help:   
python train.py --help  
  
**Required arguments**:  
**train_path**: Directory + prefix for the training data
The training data contains three files: 'FILE_PREFIX.seq', 'FILE_PREFIX.chromtracks', 'FILE_PREFIX + '.labels'
These file types are discussed in detail in the section on input data.  
**val_path**  : Directory + prefix for the validation data.  
**test_path** : Directory + prefix for the validation data.  
**no_of_chrom_tracks** : The number of chromatin experiments used.  
(Note: this does not include replicates; for each experiment; eg. ATAC-seq, the replicates must be merged prior to Bichrom application, more detains in the section Input Data).   
**out** : An output directory where Bichrom results will be stored.   

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
