import sys
import numpy as np
import pandas as pd
import scipy.stats


def get_significance(tf):
    dat = pd.read_csv(datapath + tf + '.iA.summ', sep=',', header=None,
                      names=['condition','auprc'])

    # get the median auprcs
    seq_prcs = dat['auprc'][dat['condition'] == 'iA_' + tf]
    chrom_prcs = dat['auprc'][dat['condition'] == 'iA_chrom_' + tf]

    # test for significance using wilcoxin signed rank test:
    print scipy.stats.wilcoxon(x=seq_prcs, y=chrom_prcs)


# path to summary files for Brn2, Ebf2 and Onecut2.
datapath = sys.argv[1]
get_significance('Brn2')
get_significance('Ebf2')
get_significance('Onecut2')