import numpy as np
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_atac(datapath):
    """
    plotting the differential atac-seq at iA and iN neurons
    Parameters:
     datapath: path to an DESeq2 output file:
     ATAC-seq at iAscl1 vs. iNeurog2 neurons  atac-seq file with normalized \
    Returns:
        None
    """
    dat = pd.read_csv(datapath, delimiter='\t')
    fig, ax = plt.subplots()
    plt.scatter(dat['log2FoldChange'], -1 * np.log10(dat['pvalue']),
                color='#D98880', s=4, alpha=0.8)
    plt.ylim(0, 20)
    plt.xlim(-8, 8)
    # Plotting y-lines at LFC = +-1
    plt.axvline(x=-1, color='grey', ls='--')
    plt.axvline(x=1, color='grey', ls='--')
    # Plotting x-lines at p-value = 0.05
    neglog_pval = -1 * np.log10(0.05)
    plt.axhline(y=neglog_pval, color='grey', ls='--')
    fig.set_size_inches(4, 4)
    plt.savefig(datapath + '.volcano.png', dpi=960)


if __name__ == "__main__":

    datapath = sys.argv[1]
    sns.set_style('whitegrid')
    plot_atac(datapath)