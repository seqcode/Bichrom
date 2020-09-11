import sys
import numpy as np
import pandas as pd
import scipy.stats


def get_significane_nih3t3(tf):
    # the data is in a slightly different format here:
    dat = pd.read_csv(datapath, sep='\t', header=None,
                      names=['tf', 'condition', 'auprc'])

    subdat = dat[dat['tf'] == tf]
    seq_prcs = subdat[subdat['condition'] == 'MSEQ']['auprc']
    bimodal_prcs = subdat[subdat['condition'] == 'MSC']['auprc']
    print scipy.stats.wilcoxon(x=seq_prcs.values, y=bimodal_prcs.values)


def get_significance(tf):
    dat = pd.read_csv(datapath + tf + '.iA.summ', sep=',', header=None,
                      names=['condition', 'auprc'])

    # get the median auprcs
    seq_prcs = dat['auprc'][dat['condition'] == 'iA_' + tf]
    chrom_prcs = dat['auprc'][dat['condition'] == 'iA_chrom_' + tf]

    print np.median(seq_prcs)
    print np.median(chrom_prcs)

    # test for significance using wilcoxin signed rank test:
    print scipy.stats.wilcoxon(x=seq_prcs, y=chrom_prcs)


def get_significance_pmns(datapath, tf):
    dat = pd.read_csv(datapath + tf + '.summ', sep=',', header=None,
                      names=['condition', 'auprc'])

    # get the median auprcs
    seq_prcs = dat['auprc'][dat['condition'] == 'pMNchrom_' + tf]
    chrom_prcs = dat['auprc'][dat['condition'] == 'pMNchrom_chrom_' + tf]

    # test for significance using wilcoxin signed rank test:
    print scipy.stats.wilcoxon(x=seq_prcs, y=chrom_prcs)
    print scipy.stats.mannwhitneyu(seq_prcs, chrom_prcs)



def get_significance_bj(datapath, tf):
    dat = pd.read_csv(datapath + tf + '.summ', sep=',', header=None,
                      names=['condition', 'tf', 'auprc'])

    print(dat)

    seq_prcs = dat['auprc'][dat['condition'] == 'BJ']
    chrom_prcs = dat['auprc'][dat['condition'] == 'BJ_chrom']
    print seq_prcs
    print chrom_prcs
    print scipy.stats.wilcoxon(x=seq_prcs, y=chrom_prcs)
    print scipy.stats.mannwhitneyu(seq_prcs, chrom_prcs)


# path to summary files for Brn2, Ebf2 and Onecut2.
datapath = sys.argv[1]
nih_datapath = sys.argv[2]
bj_datapath = sys.argv[3]
mn_datapath = sys.argv[4]
# get_significance('Brn2')
# get_significance('Ebf2')
# get_significance('Onecut2')

# get_significance_bj(bj_datapath, 'FoxA2')
# get_significance_bj(bj_datapath, 'Gata')
# get_significance_bj(bj_datapath, 'Oct4')

# for tf in ['CDX2', 'BHLHB8', 'FOXA1', 'DLX6', 'DUXBL', 'SOX2', 'SOX15', 'SIX6',
#            'HLF', 'RHOX11']:
#     print tf
#     get_significane_nih3t3(tf)

for tf in ['Hoxc6', 'Hoxc8', 'Hoxc9', 'Hoxc10', 'Hoxc13', 'Hoxa9', 'Hoxd9']:
    print tf
    get_significance_pmns(mn_datapath, tf)