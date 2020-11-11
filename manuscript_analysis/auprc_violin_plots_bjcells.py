import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_violin_plots(metrics_path):
    """
    Plot violin plots (manuscript figure 7) for the BJ TFs.
    Parameters:
        metrics_path: Path the directory which contains TF.summary files
        For example, the GATA summary file looks as follows:
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
        Note: seq refers to a sequence-only model.
    Returns:
        None
    """
    sns.set_style('darkgrid')
    fig, ax = plt.subplots()
    for idx, tf in enumerate(['GATA', 'FoxA2', 'Oct4']):
        dat = pd.read_csv(metrics_path + tf + '.summ', sep=',', header=None,
                          names=['condition', 'tf', 'auprc'])
        print(dat)
        plt.subplot(3, 1, idx+1)
        sns.violinplot(x=dat['condition'], y=dat['auprc'],
                       palette=('#ecce6d', '#5b8c85'),
                       order=['seq', 'bichrom'], cut=0)
        plt.ylim(0, 1)
        plt.xlabel("")
        plt.ylabel("")
    fig.set_size_inches(2, 7)
    fig.tight_layout(rect=[0.05, 0.03, 0.98, 0.98])
    plt.subplots_adjust(hspace=0.36)
    plt.savefig(metrics_path + 'violinplots.pdf')


if __name__ == "__main__":
    metrics_path = sys.argv[1]
    plot_violin_plots(metrics_path)


