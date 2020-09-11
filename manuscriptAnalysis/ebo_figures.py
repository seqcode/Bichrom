from __future__ import division

import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns


# figure a
def plot_conservation(out_path):

    # Defining the dataframes using multiGPS and edgeR data from ACI
    brn2 = pd.DataFrame([['shared', 6776], ['iA>iN', 2432], ['iN>iA', 1242]],
                        columns=['category', '#'])
    # getting fraction
    brn2['#'] = brn2['#']/np.sum(brn2['#'])
    # Ebf2
    ebf2 = pd.DataFrame([['shared', 23331], ['iA>iN', 10687], ['iN>iA', 7921]],
                        columns=['category', '#'])
    ebf2['#'] = ebf2['#']/np.sum(ebf2['#'])
    # Onecut2
    onecut2 = pd.DataFrame([['shared', 45416], ['iA>iN', 4622], ['iN>iA', 2965]],
                           columns=['category', '#'])
    onecut2['#'] = onecut2['#']/np.sum(onecut2['#'])

    # plot data
    sns.set_style('ticks')
    fig, ax = plt.subplots()

    plt.subplot(1, 3, 1)
    plt.bar([0, 1, 2], onecut2['#'], width=0.5, color='#687466')
    plt.yticks(fontsize=12)
    plt.ylim(0, 1)

    plt.subplot(1, 3, 2)
    plt.bar([0, 1, 2], brn2['#'], width=0.5, color='#cd8d7b')
    plt.yticks(fontsize=12)
    plt.ylim(0, 1)

    plt.subplot(1, 3, 3)
    plt.bar([0, 1, 2], ebf2['#'], width=0.5, color='#fbc490')
    plt.yticks(fontsize=12)
    plt.ylim(0, 1)

    sns.despine()
    fig.tight_layout()
    fig.set_size_inches(6, 4)
    plt.savefig(out_path + 'fig_a.pdf')
    plt.show()


def plot_embeddings(data_path):

    tfs = ['Brn2', 'Ebf2', 'Onecut2']
    for tf in tfs:
        dat = np.loadtxt(data_path + tf + '.embedding.txt')
        print dat
        plt.scatter(dat[:, 0], dat[:, 1], s=3, alpha=0.3)
        plt.show()


def plot_correlation(data_path):

    sns.set_style('whitegrid')
    fig, axs = plt.subplots()

    for idx, tf in enumerate(['Onecut2', 'Brn2', 'Ebf2']):
        # load chromatin data
        chrom_data = np.load(data_path + tf + '.bound.chromtracks.npy')
        chrom_sum = np.sum(chrom_data, axis=1)
        # load scores
        embedding = np.loadtxt(data_path + tf + '.embedding.txt')
        chrom_score = embedding[:, 1]
        plt.subplot(1, 3, idx+1)
        plt.scatter(chrom_sum, chrom_score, color='#084177', s=1,
                    alpha=0.05)
    fig.set_size_inches(6, 2)
    plt.subplots_adjust(left=0.1, bottom=0.2, right=0.95, top=0.95)
    plt.savefig(out_path + 'fig_b.png', dpi=960, layout='tight')


def plot_hists(data_path):

    sns.set_style('whitegrid')
    fig, ax = plt.subplots()

    tf='Brn2'
    embedding = np.loadtxt(data_path + tf + '.embedding.txt')

    seq_median = np.median(embedding[:, 0])
    chromatin_median = np.median(embedding[:, 1])

    c_pred_bool = (embedding[:, 0] < int(seq_median)) & (embedding[:, 1] > int(chromatin_median))
    s_pred_bool = (embedding[:, 0] > int(seq_median)) & (embedding[:, 1] < int(chromatin_median))
    rest = np.logical_not(c_pred_bool | s_pred_bool)


def plot_motif_heatmaps(out_path):

    # run the numbers again if interested
    fig, ax = plt.subplots()
    brns =np.array([[919.0, 320], [999, 305], [318, 717], [142, 1769], [72, 612]])
    brns[:, 0] = brns[:, 0]/933.0  # Total # of sites: 933
    brns[:, 1] = brns[:, 1]/1055.0  # Total # of sites: 1055
    print brns
    sns.heatmap(brns, cmap='bone_r', cbar_kws={'shrink': 0.5}, vmax=1.5,
                linewidths=5.3, linecolor='white')
    plt.subplots_adjust(left=0.15, bottom=0.1, right=0.85, top=0.95)
    fig.set_size_inches(2, 3)
    plt.savefig(out_path + 'fig_c1.pdf')


    fig, ax = plt.subplots()
    ebf = np.array(
        [[3146.0, 700], [2922, 1864], [3544, 1228], [1865, 6496],[2882, 2124], [104, 1214]])
    ebf[:, 0] = ebf[:, 0] / 4146.0  # Total # of sites: 4146
    ebf[:, 1] = ebf[:, 1] / 3469.0  # Total # of sites: 3469
    print ebf
    sns.heatmap(ebf, cmap='bone_r', cbar_kws={'shrink': 0.5}, vmax=1.5,
                linewidths=5.3, linecolor='white')
    plt.subplots_adjust(left=0.15, bottom=0.1, right=0.85, top=0.95)
    fig.set_size_inches(2, 3)
    plt.savefig(out_path + 'fig_c2.pdf')

    fig, ax = plt.subplots()
    oc =np.array([[1055.0, 6234], [3637, 542], [5227, 1245], [1282, 10372],
                  [1266, 10067]])
    oc[:, 0] = oc[:, 0]/5771.0  # Total # of sites: 5771
    oc[:, 1] = oc[:, 1]/4627.0  # Total # of sites: 4627
    print oc
    sns.heatmap(oc, cmap='bone_r', cbar_kws={'shrink': 0.5}, vmax=1.5,
                linewidths=5.3, linecolor='white')
    plt.subplots_adjust(left=0.15, bottom=0.1, right=0.85, top=0.95)
    fig.set_size_inches(2, 3)
    plt.savefig(out_path + 'fig_c3.pdf')


def plot_ebo_boxplots(metrics_path):
    sns.set_style('darkgrid')
    fig, ax = plt.subplots()
    for idx, tf in enumerate(['Brn2', 'Ebf2', 'Onecut2']):
        dat = pd.read_csv(metrics_path + tf + '.iA.summ', sep=',', header=None,
                          names=['condition', 'tf', 'auprc'])
        print dat
        plt.subplot(1, 3, idx+1)
        sns.violinplot(x=dat['condition'], y=dat['auprc'],
                       palette=('#ecce6d', '#5b8c85'),
                       order=['iA', 'iA_chrom'], cut=0)
        plt.ylim(0, 1)
        plt.xlabel("")
        plt.ylabel("")
    fig.set_size_inches(6, 3)
    plt.savefig(metrics_path + 'violinplots.pdf')



out_path = sys.argv[1]
data_path = sys.argv[2]  # ~/Desktop/March-iTF/iTF_revision/localEBO/
# figure 1
# plot_conservation(out_path)
# figure 2: metrics/posterior_accuracies.py
# figure 3: conservation!
# plot_embeddings(data_path=data_path)
# figure : correlation between score and accessibility:
# plot_correlation(data_path=data_path)
# figure: plot seq-driven & chromatin-driven
# plot_hists(data_path=data_path)
# plot_motif_heatmaps(out_path=out_path)
metrics_path = sys.argv[3]
plot_ebo_boxplots(metrics_path)