## Bichrom: A bimodal neural network to predict TF binding using sequence and pre-existing chromatin track data
Transcription factor (TF) binding specificity is determined via a complex interplay between the TF’s DNA binding preference and cell type-specific chromatin environments. The chromatin features that correlate with TF binding in a given cell type have been well characterized. For instance, the binding sites for a majority of TFs display concurrent chromatin accessibility. However, concurrent chromatin features reflect the binding activities of the TF itself, and thus provide limited insight into how genome-wide TF-DNA binding patterns became established in the first place. To understand the determinants of TF binding specificity, we therefore need to examine how newly activated TFs interact with sequence and preexisting chromatin landscapes.

Here, we investigate the sequence and preexisting chromatin predictors of TF-DNA binding by examining the genome-wide occupancy of TFs that have been induced in well-characterized chromatin environments. We develop Bichrom, a bimodal neural network that jointly models sequence and preexisting chromatin data to interpret the genome-wide binding patterns of induced TFs. We find that the preexisting chromatin landscape is a differential global predictor of TF-DNA binding; incorporating preexisting chromatin features improves our ability to explain the binding specificity of some TFs substantially, but not others. Furthermore, by analyzing site-level predictors, we show that TF binding in previously inaccessible chromatin tends to correspond to the presence of more favorable cognate DNA sequences. Bichrom thus provides a framework for modeling, interpreting, and visualizing the joint sequence and chromatin landscapes that determine TF-DNA binding dynamics.

## Citation
Srivastava, D., Aydin, B., Mazzoni, E.O. and Mahony, S., 2020. An interpretable bimodal neural network characterizes the sequence and preexisting chromatin predictors of induced TF binding. bioRxiv, p.672790.

## Installation
**Requirements**:  

python >= 3.5  

To install python dependencies:
`pip install -r requirements.txt`

Alternatively, use anaconda to create a virtual environment using the provided YAML configuration file:
`conda create -f bichrom.yml`

## Usage
```
# Clone and navigate to the iTF repository. 
cd trainNN  
To view help:   
python train.py --help
```
  
## Input files & usage:  
iTF trains and evaluates two models: 
* A sequence based classifier for TF binding prediction (Bichrom<sub>SEQ</sub>)
* A sequence + pre-existing chromatin based classifier for TF binding prediction (Bichrom)

**Input**  

Required arguments: 
* training_schema_yaml: This is a YAML file containing containing paths to the training data (sequence, preexisting chromatin and labels), validation data and test data. A sample YAML file can be found in trainNN/sample.yaml. The structure of the training_schema_yaml file should be as follows:  

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


* outdir: This is the output directory, where all Bichrom output files will be placed. 

**Output**
iTF outputs the validation and test metrics (auROC and auPRC) for both a sequence-only network (Bichrom<sub>SEQ</sub>) and a complete sequence + preexisting chromatin bimodal network (Bichrom). It additionally plots the test Precision Recall curves for both models; as well as test recall at a false positive rate=0.01. 
   


