## Bichrom: A bimodal neural network to predict TF binding using sequence and pre-existing chromatin track data
Bichrom provides a framework for modeling, interpreting, and visualizing the joint sequence and chromatin landscapes that determine TF-DNA binding dynamics.

### Citation
Srivastava, D., Aydin, B., Mazzoni, E.O. and Mahony, S., 2020. An interpretable bimodal neural network characterizes the sequence and preexisting chromatin predictors of induced TF binding. bioRxiv, p.672790.

### Installation
**Requirements**:  

python >= 3.5  
We suggest using anaconda to create a virtual environment using the provided YAML configuration file:
`conda env create -f bichrom.yml`  
Alternatively, to install requirements using pip: 
`pip install -r requirements.txt`

### Usage
```
# Clone and navigate to the iTF repository. 
cd trainNN  
To view help:   
python run_bichrom.py --help
usage: run_bichrom.py [-h] training_schema_yaml window_size bin_size outdir

Train and compare BichromSEQ and Bichrom

positional arguments:
  training_schema_yaml  YAML file with paths to train, test and val data
  window_size           Size of genomic windows
  bin_size              Size of bins for chromatin data
  outdir                Output directory

optional arguments:
  -h, --help            show this help message and exit

```
  
### Input Files: Description  

**Required arguments**: 

* **training_schema_yaml**:
This is a YAML file containing containing paths to the training data (sequence, preexisting chromatin and labels), validation data and test data. A sample YAML file can be found in trainNN/sample.yaml. The structure of the training_schema_yaml file should be as follows:  

  <pre>
  train:  
    seq: '/path/to/train/seq.txt'    
    labels: '/path/to/train/labels.txt'  
    chromatin_tracks: ['/path/to/train/atacseq.txt', ..., '/path/to/train/h3k27ac.txt']  

  val: 
    seq: '/path/to/val/seq.txt'  
    labels: '/path/to/val/labels.txt'  
    chromatin_tracks: ['/path/to/val/atacseq.txt', ..., '/path/to/val/h3k27ac.txt'] 

  test: 
    seq: '/path/to/test/seq.txt'  
    labels: '/path/to/test/labels.txt'  
    chromatin_tracks: ['/path/to/val/atacseq.txt', ..., '/path/to/test/h3k27ac.txt'] 
  </pre>

  **Description for the input files provided in the YAML configuration**: 
  Each input data point (train, test or validation) corresponds to a 500 base pair window on the genome. The "seq", "labels" and "chromatin_tracks" files contain genomic features associated with these input 500 base pair windows. 

  - **seq**: The sequence input file contains one sequence per line. For example, if your training set has 25,100 genomic windows, the seq file will contain 25,100 lines. (Permitted bases: A, T, G, C, N). 

  - **labels**: This file contains a binary label that has been assigned to each training, validation and test input data point. (Must be 0/1).  
  
  - **chromatin_tracks**: Multiple chromatin files can be passed to Bichrom through the YAML file. (The YAML field chromatin_tracks accepts a list of file locations.) Each line in a chromatin track file contains tab separated binned chromatin data. The data can be binned at any resolution.   For example, if the genomic windows used to train Bichrom are 500 base pairs, then: 
    * If bin_size=50 base pairs, then each line in the chromatin file must contain 10 (500/50) tab separated values. 
    * If bin_size=1 base pair, then each line in the chromatin file must contain 500 values. Note that all chromatin feature files that are passed to Bichrom must be binned at the same resolution.  

Other required arguments: 

* **window_size**: The size of genomic windows used for training, validation and testing. (For example: 500)
* **bin_size**: Binning applied to the chromatin data. (For example, if window_size=500 and bin_size=10, each line in a chromatin_track file must contain 500/10 tab separated values)
* **outdir**: Output directory. Bichrom creates the following outputs files and sub-directories: 
  * mseq: 
    * records the validation loss and auPRC for each epoch the sequence-only network (Bichrom-SEQ).
    * stores models (tf.Keras Models) checkpointed after each epoch. 
    * stores Bichrom-SEQ ouput probabilities over the testing data. 
  * msc: 
    * records the validation loss and auPRC for each epoch the Bichrom. 
    * stores models (tf.Keras Models) checkpointed after each epoch. 
    * stores the Bichrom ouput probabilities over the testing data. 
  * metrics.txt: stores the test auROC and the auPRC for both the sequence-only network (Bichrom-SEQ) and for Bichrom. 
  * best_model.hdf5: A tensorflow.Keras Model (with the highest validation set auPRC)
  * precision-recall curves for Bichrom-SEQ and Bichrom.


