import numpy as np
import sys
import pandas as pd
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def plot_boxplots(metrics_path):
    sns.set_style('darkgrid')
    fig, ax = plt.subplots()
    for idx, tf in enumerate(['GATA', 'FoxA2', 'Oct4']):
        dat = pd.read_csv(metrics_path + tf + '.summ', sep=',', header=None,
                          names=['condition', 'tf', 'auprc'])
        print dat
        plt.subplot(3, 1, idx+1)
        sns.violinplot(x=dat['condition'], y=dat['auprc'],
                       palette=('#ecce6d', '#5b8c85'),
                       order=['BJ', 'BJ_chrom'], cut=0)
        plt.ylim(0, 1)
        plt.xlabel("")
        plt.ylabel("")
    fig.set_size_inches(2, 7)
    fig.tight_layout(rect=[0.05, 0.03, 0.98, 0.98])
    plt.subplots_adjust(hspace=0.36)
    plt.savefig(metrics_path + 'violinplots.pdf')


metrics_path = sys.argv[1]
plot_boxplots(metrics_path)


