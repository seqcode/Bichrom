import pandas as pd
import scipy.stats


def get_significance_per_tf(input_datapath):
    """
    Run the sequence-network and Bichrom over N held-out test sets.
    Check whether the gain in performance upon incorporation of prior chromatin
    data with Bichrom is significant.

    Parameters:
        input_datapath: Input file storing the summary statistics for each
        TF. For example, the summary file for GATA ChIP-seq in BJ cells is
        as follows:
        ...
        bichrom, GATA, 0.49097278959035834
        bichrom, GATA, 0.515491844830841
        bichrom, GATA, 0.572293273059536
        bichrom, GATA, 0.4909197931794813
        bichrom, GATA, 0.519433898153947
        seq, GATA, 0.40140515853838615
        seq, GATA, 0.4071458624248806
        seq, GATA, 0.4944029049796368
        seq, GATA, 0.3942885914448734
        seq, GATA, 0.4207938581419808
        ...
    Return: Output & p-values from the Wilcoxon signed rank test.
    """
    dat = pd.read_csv(input_datapath, sep = ',', header=None,
                      names=['condition', 'tf', 'auprc'])

    sequence_network_prc = dat['auprc'][dat['condition'] == 'seq']
    bichrom_prc = dat['auprc'][dat['condition'] == 'bichrom']
    return scipy.stats.wilcoxon(x=sequence_network_prc, y=bichrom_prc)