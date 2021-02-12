## Bichrom
Bichrom provides a framework for modeling, interpreting, and visualizing the joint sequence and chromatin landscapes that determine TF-DNA binding dynamics.

## Citation
Srivastava, D., Aydin, B., Mazzoni, E.O. et al. An interpretable bimodal neural network characterizes the sequence and preexisting chromatin predictors of induced transcription factor binding. Genome Biol 22, 20 (2021). 
https://doi.org/10.1186/s13059-020-02218-6


## Installation and Requirements 

**Please Note**: This repository has been updated as of **02/11/2021**. Input file formats have been modified to increase readability.   
python >= 3.5  
We suggest using anaconda to create a virtual environment using the provided YAML configuration file:
`conda env create -f bichrom.yml`  

Alternatively, to install requirements using pip: 
`pip install -r requirements.txt`

**Note**: Bichrom uses Pybedtools to construct genome-wide training, test and validation datasets. In order to use this functionality, you must have bedtools installed. To install bedtools, follow instructions here: https://bedtools.readthedocs.io/en/latest/content/installation.html 

**Note**: For GPU compatibility, tensorflow 2.2.1 requires CUDA 10.1 and cuDNN >= 7.

## Usage


### Step 1 - Construct Bichrom Input Data

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

**Required Arguments**

**info**:   
This is a standard genome sizes file, recording the size of each chromosome. It contains 2 tab-separated columns containing the chromosome name and chromosome size.  
For an example file, please see: `sample_data/mm10.info`.  
Genome sizes files are typically available from the UCSC Genome Browser (https://genome.ucsc.edu)

**fa**:  
This is a fasta file from which train, test and validation data should be constructed. 

**len**:  
Length of training, test and validation windows. (**Recommended=500**)

**acc_domains**:   
A BED file containing accessibility domains in the cell-type of interest. This will be used for sampling regions from accessible chromatin when constructing the Bichrom training data.  
For an example file, please see: `sample_data/mES_atacseq_domains.bed`.

**chromtracks**:   
One or more BigWig files containing histone ChIP-seq or ATAC-seq data. 

**peaks**:  
ChIP-seq or ChIP-exo TF peaks in the multiGPS file format. Each peak is represented as **chromosome:midpoint**.  
For an example file, please see: `sample_data/Ascl1.events`.

**nbins**:  
The number of bins to use for binning the chromatin data. (**Recommended=10-20**. Note that with an increase in resolution and **nbins** (or decrease in bin size), the memory requirements will increase.)

**o**:   
Output directory for storing output train, test and validation datasets. 

**blacklist** (optional):   
A blacklist BED file, with artifactual regions to be excluded from the training.  
For an example file, please see: `sample_data/mm10_blacklist.bed`.

### Step 1 - Output 
construct_data.py will produce train, test and validation datasets in the specified output directory.
This function will also produce a configuration file called **bichrom.yaml**, which can be used as input to run Bichrom. This configuration file stores the paths to the created train, test and validation datasets. 


### Step 2 - Train Bichrom

```
cd trainNN  
To view help:   
python run_bichrom.py -h
usage: run_bichrom.py [-h] -training_schema_yaml TRAINING_SCHEMA_YAML -len LEN
                      -outdir OUTDIR -nbins NBINS

Train and Evaluate Bichrom

optional arguments:
  -h, --help            show this help message and exit
  -training_schema_yaml TRAINING_SCHEMA_YAML
                        YAML file with paths to train, test and val data
  -len LEN              Size of genomic windows
  -outdir OUTDIR        Output directory
  -nbins NBINS          Number of bins

```
  
**Required arguments**: 

**training_schema_yaml**:  
This configuration files contains paths to the formatted train, test and validation data. This file will be automatically generated using construct_data.py (**see above - construct_data.py will output bichrom.yaml**).

In order to construct the training data, we implement several sampling strategies including over-sampling the negative training regions from accessible chromatin and from genomic regions flanking the TF binding sites (detailed in the paper). However, if you would like to construct training data using your own strategy, please input a custom configuration file here. More details for custom configuration files can be found at the bottom of the README.  

**len**:  
The size of genomic windows used for training, validation and testing. (Recommended: 500).   
**nbins**:  
The number of bins to use for binning the chromatin data.   
**outdir**:   
Bichrom's output directory.  


### Step 2 - Description of Bichrom's Output
Bichrom output directory. 
  * seqnet: 
    * records the validation loss and auPRC for each epoch the sequence-only network (Bichrom-SEQ).
    * stores models (tf.Keras Models) checkpointed after each epoch. 
    * stores ouput probabilities over the test data for a sequence-only network. 
  * bichrom: 
    * records the validation loss and auPRC for each epoch the Bichrom. 
    * stores models (tf.Keras Models) checkpointed after each epoch. 
    * stores the Bichrom ouput probabilities over testing data. 
  * metrics.txt: stores the test auROC and the auPRC for both a sequence-only network and for Bichrom. 
  * best_model.hdf5: A Bichrom tensorflow.Keras Model (with the highest validation set auPRC)
  * precision-recall curves for the sequence-only network and Bichrom.
  
### Optional: Custom Training Sets and YAML files
If generating custom training data, please specify a custom YAML file for training Bichrom. Bichrom requires the following files: **1)** Training files, **2)** Validation files, **3)** Test Files. 

Within each category, Bichrom expects **3 file types**: 
* Sequence File: This file contains sequence data (one training sequence of lenght L/line). Acceptable nucleotides: A, T, G, C, N. 
* Chromatin Files: 1 file per chromatin experiment. Each input chromatin file contains chromatin signal (binned at any resolution) over the input genomic windows.
* Label File: This file contains binary labels associated with TF binding over the input genomic windows. 

File paths to these files should be summarized in a configuration YAML file. For the structure of the YAML file, please see:   `sample_data/sample_custom_config.yaml`


### 2-D Bichrom embeddings
For 2-D latenet embeddings, please refer to the README in the ```Bichrom/latent_embeddings directory```
