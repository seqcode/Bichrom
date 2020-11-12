### Bichrom Embeddings
This module will report latent (2-D) co-ordinates for any DNA sequence and chromatin feature combination, given a trained Bichrom model. It will also plot the latent embeddings. 

**(see trainNN README.md for details on how to train a Bichrom model)**

### Usage

To view help: ```python embed.py --help```
```
usage: embed.py [-h] -model MODEL -seq SEQ -chrom CHROM [CHROM ...] -length
                LENGTH -outdir OUTDIR

Derives and plots 2-D embeddings

optional arguments:
  -h, --help            show this help message and exit
  -model MODEL          Trained bichrom model
  -seq SEQ              Sequence file, each line contains sequences associated
                        with one genomic window
  -chrom CHROM [CHROM ...]
                        List of files, each containing chromatin features,
                        associated with one genomic window per line
  -length LENGTH        Length of training windows
  -outdir OUTDIR        Output directory
```

### Input Data Requirements and Details

* The **sequence file** contains: Sequences of length L. (Permitted bases: A, T, G, C, N)
* The **chromatin files** contains: A binned vector of chromatin signal. 

  (Example: If **L=500** and binsize **B=10**, then chromatin signal vector must contain **L/B** values)
    
  Note: Multiple chromatin track files can be provided with the -chrom argument (For example: ATAC-seq, H3K27ac, ... etc.).
  
