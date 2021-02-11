## Bichrom
Bichrom provides a framework for modeling, interpreting, and visualizing the joint sequence and chromatin landscapes that determine TF-DNA binding dynamics.

## Citation
Srivastava, D., Aydin, B., Mazzoni, E.O. et al. An interpretable bimodal neural network characterizes the sequence and preexisting chromatin predictors of induced transcription factor binding. Genome Biol 22, 20 (2021). 
https://doi.org/10.1186/s13059-020-02218-6


## Installation and Requirements 

python >= 3.5  
We suggest using anaconda to create a virtual environment using the provided YAML configuration file:
`conda env create -f bichrom.yml`  

Alternatively, to install requirements using pip: 
`pip install -r requirements.txt`

**Note**: Bichrom uses Pybedtools to construct genome-wide training, test and validation datasets. In order to use this functionality, you must have bedtools installed. To install bedtools, follow instructions here: https://bedtools.readthedocs.io/en/latest/content/installation.html 

**Note**: For GPU compatibility, tensorflow 2.2.1 requires CUDA 10.1 and cuDNN >= 7.

## Usage I: Construct Datasets for Training, Testing and Validation

Clone and navigate to the Bichrom repository. 
```
cd construct data
usage: construct_data.py [-h] -info INFO -fa FA -blacklist BLACKLIST -len LEN
                         -acc_domains ACC_DOMAINS -chromtracks CHROMTRACKS
                         [CHROMTRACKS ...] -peaks PEAKS -o OUTDIR

Construct Training Data For Bichrom

optional arguments:
  -h, --help            show this help message and exit
  -info INFO            Genome sizes file
  -fa FA                The fasta file for the genome of interest
  -blacklist BLACKLIST  Blacklist file for the genome of interest
  -len LEN              Size of training, test and validation windows
  -acc_domains ACC_DOMAINS
                        Bed file with accessible domains
  -chromtracks CHROMTRACKS [CHROMTRACKS ...]
                        A list of BigWig files for all input chromatin
                        experiments
  -peaks PEAKS          A ChIP-seq or ChIP-exo peak file in multiGPS file
                        format
  -o OUTDIR, --outdir OUTDIR
                        Output directory for storing train, test data

```

### Required Arguments

**info** : This is a standard genome sizes file, recording the size of each chromosome. It contains 2 tab-separated columns containing the chromosome name and chromosome size. For an example, please see: sample_data/mm10.info. Genome sizes files are typically available from UCSC Genome Browser.

**fa**: This is a fasta file from which train, test and validation data should be constructed. 

**len**: Length of training, test and validation windows. (Recommended=500)

**acc_domains**: A BED file containing accessible domains. This will be used for sampling from accessible chromatin while constructing the training data. 

**chromtracks**: One or more BigWig files representing the chromatin datasets to be used as predictors of TF binding. 

**peaks**: A ChIP-seq or ChIP-exo peaks file in the multiGPS file format. Each peak (line in file) is represented as chromosome:midpoint. For an example, please see: sample_data/Ascl1.events 

**o**: Output directory for storing output train, test and validation datasets. 

### Optional Arguments

**blacklist**: A blacklist BED file, with artifactual regions to be excluded from the training. For an example, please see: sample_data/mm10_blacklist.bed

## Usage 2: Train and compare a sequence-only CNN-LSTM to Bichrom

```
cd trainNN  
To view help:   
python run_bichrom.py -h
usage: run_bichrom.py [-h] -training_schema_yaml TRAINING_SCHEMA_YAML -len LEN
                      -outdir OUTDIR

Train and compare Bichrom-SEQ and Bichrom

optional arguments:
  -h, --help            show this help message and exit
  -training_schema_yaml TRAINING_SCHEMA_YAML
                        YAML file with paths to train, test and val data
  -len LEN              Size of genomic windows
  -outdir OUTDIR        Output directory
```
  
### Required arguments**: 

* **training_schema_yaml**:
This is a YAML file containing containing paths to the training data (sequence, preexisting chromatin and labels), validation data and test data. A sample YAML file can be found in trainNN/sample.yaml. The structure of the training_schema_yaml file should be as follows:  

  <pre>
  train:  
    seq: data_train.seq    
    labels: data_train.labels  
    chromatin_tracks: ['data_train.atacseq.chromatin', ..., 'data_train.h3k27ac.txt']  

  val: 
    seq: data_val.seq 
    labels: data_val.labels 
    chromatin_tracks: ['data_val.atacseq.chromatin', ..., 'data_val.h3k27ac.txt']  

  test: 
    seq: data_test.seq
    labels: data_test.labels  
    chromatin_tracks: ['data_test.atacseq.chromatin', ..., 'data_test.h3k27ac.txt']
  </pre>

**Note**: These datasets should be pre-constructed using construct_data.py (See part 1). 

* **len**: The size of genomic windows used for training, validation and testing. (Recommended: 500)
* **outdir**: Output directory. Bichrom creates the following outputs files and sub-directories: 
  * seqnet: 
    * records the validation loss and auPRC for each epoch the sequence-only network (Bichrom-SEQ).
    * stores models (tf.Keras Models) checkpointed after each epoch. 
    * stores Bichrom-SEQ ouput probabilities over the testing data. 
  * bichrom: 
    * records the validation loss and auPRC for each epoch the Bichrom. 
    * stores models (tf.Keras Models) checkpointed after each epoch. 
    * stores the Bichrom ouput probabilities over the testing data. 
  * metrics.txt: stores the test auROC and the auPRC for both the sequence-only network (Bichrom-SEQ) and for Bichrom. 
  * best_model.hdf5: A tensorflow.Keras Model (with the highest validation set auPRC)
  * precision-recall curves for Bichrom-SEQ and Bichrom.

### 2-D Bichrom embeddings
For 2-D latenet embeddings, please refer to the README in the ```Bichrom/latent_embeddings directory```
