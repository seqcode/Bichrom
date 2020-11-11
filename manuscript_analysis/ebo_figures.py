import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns


def plot_conservation(out_path):
    """
    Plotting the fraction of conserved binding sites for Brn2, Ebf2 and
    Onecut2, based on multiGPS and edgeR results from Aydin et al., 2019
    (Nature Neurosciece: PMID 31086315)

    Parameters:
        out_path: Filepath prefix for output bar plots (Manuscript Fig. 6A)
    Returns: None
    """
    # Defining the dataFrames using multiGPS and edgeR results \
    # from Aydin et al., (2019) Nat. Neuroscience.
    # Brn2
    brn2 = pd.DataFrame([['shared', 6776], ['iA>iN', 2432], ['iN>iA', 1242]],
                        columns=['category', '#'])
    brn2['#'] = brn2['#']/np.sum(brn2['#'])

    # Ebf2
    ebf2 = pd.DataFrame([['shared', 23331], ['iA>iN', 10687], ['iN>iA', 7921]],
                        columns=['category', '#'])
    ebf2['#'] = ebf2['#']/np.sum(ebf2['#'])

    # Onecut2
    onecut2 = pd.DataFrame([['shared', 45416], ['iA>iN', 4622], ['iN>iA', 2965]],
                           columns=['category', '#'])
    onecut2['#'] = onecut2['#']/np.sum(onecut2['#'])

    # plot bar plots
    sns.set_style('ticks')
    fig, ax = plt.subplots()
    plt.subplot(1, 3, 1)
    plt.bar([0, 1, 2], onecut2['#'], width=0.5, color='#687466')
    plt.yticks(fontsize=12)
    plt.ylim(0, 1)
    #
    plt.subplot(1, 3, 2)
    plt.bar([0, 1, 2], brn2['#'], width=0.5, color='#cd8d7b')
    plt.yticks(fontsize=12)
    plt.ylim(0, 1)
    #
    plt.subplot(1, 3, 3)
    plt.bar([0, 1, 2], ebf2['#'], width=0.5, color='#fbc490')
    plt.yticks(fontsize=12)
    plt.ylim(0, 1)
    #
    sns.despine()
    fig.tight_layout()
    fig.set_size_inches(6, 4)
    plt.savefig(out_path + 'Fig_6a.pdf')


def plot_embeddings(data_path, outpath):
    """
    Plot 2-D latent embeddings for Brn2, Ebf2 and Onecut2.

    Parameters:
        data_path: Input file paths (N rows * 2 columns) storing the 2-D co-ordinates
        for each binding site in the latent space. The embeddings must be derived
        using latent_embeddings/get_latent_embeddings.py
        Note: This function assumes that the files are saved with an \
        ".embedding.txt" extension. Provide only the prefix as an argument.
        For example, if the 2-D embedding is stored in "~/Prefix/Oct4.embedding.txt",
        call function as: plot_embeddings("~/Prefix/Oct4")
        outpath: Output file path.
    Returns: None
    """
    transcription_factors = ['Brn2', 'Ebf2', 'Onecut2']
    for tf in transcription_factors:
        dat = np.loadtxt(data_path + tf + '.embedding.txt')
        plt.scatter(dat[:, 0], dat[:, 1], s=3, alpha=0.3)
        plt.savefig(outpath)


def plot_correlation(data_path, outpath):
    """
    Plotting the correlation between ATAC-seq data at individual sites and the
    associated chromatin sub-network (Bichrom-CHR) scores.
    Parameters:
        data_path: Prefix for the ".bound.chromtracks.npy" file. This file stores the
        chromatin data at each binding site.
        outpath: Output file path.
    Returns: None
    """
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
    plt.savefig(outpath + 'fig_b.png', dpi=960, layout='tight')


def plot_motif_heatmaps(out_path):
    """
    Run MEME-ChIP & FIMO to get the number of motifs enriched at \
    chromatin predicted (CP) and sequence predicted (SP) sites.
    Parameters:
        out_path: Output file path
    """
    # Brn2
    fig, ax = plt.subplots()
    brn2 =np.array([[919.0, 320], [999, 305], [318, 717], [142, 1769], [72, 612]])
    brn2[:, 0] = brn2[:, 0]/933.0   # Total # of sites: 933
    brn2[:, 1] = brn2[:, 1]/1055.0  # Total # of sites: 1055
    sns.heatmap(brn2, cmap='bone_r', cbar_kws={'shrink': 0.5}, vmax=1.5,
                linewidths=5.3, linecolor='white')
    plt.subplots_adjust(left=0.15, bottom=0.1, right=0.85, top=0.95)
    fig.set_size_inches(2, 3)
    plt.savefig(out_path + 'fig_c1.pdf')

    # Ebf2
    fig, ax = plt.subplots()
    ebf2 = np.array([[3146.0, 700], [2922, 1864], [3544, 1228], [1865, 6496],
                    [2882, 2124], [104, 1214]])
    ebf2[:, 0] = ebf2[:, 0] / 4146.0  # Total # of sites: 4146
    ebf2[:, 1] = ebf2[:, 1] / 3469.0  # Total # of sites: 3469
    sns.heatmap(ebf2, cmap='bone_r', cbar_kws={'shrink': 0.5}, vmax=1.5,
                linewidths=5.3, linecolor='white')
    plt.subplots_adjust(left=0.15, bottom=0.1, right=0.85, top=0.95)
    fig.set_size_inches(2, 3)
    plt.savefig(out_path + 'fig_c2.pdf')

    # Onecut2
    fig, ax = plt.subplots()
    oc2 =np.array([[1055.0, 6234], [3637, 542], [5227, 1245], [1282, 10372],
                  [1266, 10067]])
    oc2[:, 0] = oc2[:, 0]/5771.0  # Total # of sites: 5771
    oc2[:, 1] = oc2[:, 1]/4627.0  # Total # of sites: 4627
    sns.heatmap(oc2, cmap='bone_r', cbar_kws={'shrink': 0.5}, vmax=1.5,
                linewidths=5.3, linecolor='white')
    plt.subplots_adjust(left=0.15, bottom=0.1, right=0.85, top=0.95)
    fig.set_size_inches(2, 3)
    plt.savefig(out_path + 'fig_c3.pdf')


def plot_ebo_boxplots(data_path, outpath):
    """
    Plot violin plots (manuscript figure 6) for the iAscl1 TFs.
    Parameters:
        metrics_path: Path the directory which contains TF.iA.summary files
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
        Note that seq refers to a sequence-only model.
        outpath: Output file path.
    Returns:
        None
    """
    sns.set_style('darkgrid')
    fig, ax = plt.subplots()
    for idx, tf in enumerate(['Brn2', 'Ebf2', 'Onecut2']):
        dat = pd.read_csv(data_path + tf + '.iA.summary', sep=',', header=None,
                          names=['condition', 'tf', 'auprc'])
        plt.subplot(1, 3, idx+1)
        sns.violinplot(x=dat['condition'], y=dat['auprc'],
                       palette=('#ecce6d', '#5b8c85'),
                       order=['seq', 'bichrom'], cut=0)
        plt.ylim(0, 1)
        plt.xlabel("")
        plt.ylabel("")
    fig.set_size_inches(6, 3)
    plt.savefig(data_path + 'violinplots.pdf')


if __name__ == "__main__":
    out_path = sys.argv[1]
    data_path = sys.argv[2]

    plot_conservation(out_path)
    plot_embeddings(data_path=data_path, outpath=out_path)
    plot_correlation(data_path=data_path, outpath=out_path)
    plot_motif_heatmaps(out_path=out_path)
    plot_ebo_boxplots(data_path=data_path, outpath=out_path)
