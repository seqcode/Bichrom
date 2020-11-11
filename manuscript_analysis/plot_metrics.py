import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import precision_recall_curve


def plot_boxplot_ascl1(summary_file):
    # manuscript figure 2A
    # set style
    sns.set_style('ticks', {'axes.grid': True})
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=.20, bottom=.10, right=.99, top=.90)

    # managing the data:
    dat = pd.read_csv(summary_file, sep=',', header=None,
                      names=['condition', 'prc'])
    print dat
    # Plot 1: Using only the following four runs here: (not the ATAC)
    # 'ES_chrom', 'ES_seq', 'ES', 'input'
    dat = dat[(dat['condition'] == 'ES_chrom') | (dat['condition'] == 'ES_seq') |
              (dat['condition'] == 'ES') | (dat['condition'] == 'input')]
    # choosing the color palette
    my_pal = {'ES_chrom': '#b0a160', 'ES_seq': '#ecce6d', 'ES': '#5b8c85',
              'input': '#434e52'}
    # box plot for gain in performance:
    sns.boxplot(dat['condition'], dat['prc'],
                boxprops=dict(alpha=1), width=0.8, linewidth=1.5,
                order=['ES_chrom', 'ES_seq', 'input', 'ES'],
                palette=my_pal, fliersize=1.5)
    plt.ylim(0, 1)
    plt.yticks(fontsize=14)
    # setting figure size and saving figure.
    fig.set_size_inches(3.5, 5)
    plt.savefig(summary_file + '.pdf')


def make_pr_curves(labels_file, probas_dir, bedfile, outpath):
    # manuscript figure 2B
    sns.set_style('ticks', {'axes.grid': False})

    def plot_pr_curves(y_true, y_pred, color, lw, alpha):
        precision, recall, _ = precision_recall_curve(
            y_true=y_true,
            probas_pred=y_pred)
        plt.plot(recall, precision, c=color, lw=lw, alpha=alpha)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

    # loading files:
    bed_locations = pd.read_csv(bedfile, sep='\t', header=None,
                                names=['chr', 'start', 'stop'])

    fig, ax = plt.subplots()
    for chromos in ['chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15',
                    'chr16', 'chr18', 'chr19']:
        seq_probas_file = probas_dir + 'Ascl1.' + chromos + '.seq.probs'
        chrom_probas_file = probas_dir + 'Ascl1.' + chromos + '.chrom.probs'
        seq_probas = pd.read_csv(seq_probas_file, header=None)
        chrom_probas = pd.read_csv(chrom_probas_file, header=None)

        # subset labels file
        # Doing this because I only have access to the labels file over the
        # whole genome.
        labels = pd.read_csv(labels_file, header=None, names=['labels'])
        dat = pd.concat([bed_locations, labels], axis=1)
        bool_array = dat['chr'] == chromos
        subsetted_data = dat[bool_array]

        # for chromosome 10, plotting in bold, else alpha=0.2
        if chromos == 'chr10':

            plot_pr_curves(y_true=subsetted_data['labels'], y_pred=seq_probas,
                           color='#ecce6d', lw=1.5, alpha=1)
            plot_pr_curves(y_true=subsetted_data['labels'], y_pred=chrom_probas,
                           color='#5b8c85', lw=1.5, alpha=1)
        else:
            plot_pr_curves(y_true=subsetted_data['labels'], y_pred=seq_probas,
                           color='#ecce6d', lw=1, alpha=0.25)
            plot_pr_curves(y_true=subsetted_data['labels'], y_pred=chrom_probas,
                           color='#5b8c85', lw=1, alpha=0.25)
    plt.ylim(0, 1)
    plt.yticks(fontsize=10)
    plt.xticks(fontsize=10)
    # setting figure size and saving figure.
    fig.set_size_inches(3.5, 3.5)
    plt.savefig(outpath + '.prcurve.pdf')


def main():
    # This is the summary file containing a summary of runs for Ascl1:
    infile = sys.argv[1]
    # This is the whole-genome labels file:
    labels_file = sys.argv[2]
    # Directory with probabilities over chr10-chr19
    run_dir = sys.argv[3]
    # This is the whole-genome bed file
    bedfile = sys.argv[4]

    # Plot figure 2A
    plot_boxplot_ascl1(infile)

    # Plot figure 2B
    make_pr_curves(labels_file=labels_file, probas_dir=run_dir, bedfile=bedfile,
                   outpath=infile)

