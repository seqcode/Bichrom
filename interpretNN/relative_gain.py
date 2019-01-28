import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def metrics_boxplots(datapath):
    sns.set_style('ticks', {'axes.edgecolor': 'black'})
    print sns.axes_style()
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=.20, bottom=.10, right=.99, top=.90)
    # sns.set_style('ticks')
    dat = np.loadtxt(datapath + '.txt', delimiter=", ", dtype=str)
    # Plot 1: Using only the following four runs here: (not the ATAC)
    c0 = (dat[:, 0] == 'ES_chrom')
    c1 = (dat[:, 0] == 'ES_seq')
    c2 = (dat[:, 0] == 'ES')
    c3 = (dat[:, 0] == 'input')
    # Let's use these conditions to subset are data
    conditions = np.transpose(np.vstack((c0, c1, c2, c3)))
    dat = dat[np.any(conditions, axis=1)]
    width = 4
    height = 4.5
    # Box plot for gain in performance:
    sns.boxplot(dat[:, 0], dat[:, 1].astype(float), palette="Set2")
    plt.xticks(range(4), [r'$M_c$', r'$M_s$', r'$M_s$'+r'$_i$', r'$M_s$' + r'$_c$'], fontsize=12)
    plt.ylim(0, 1)
    plt.yticks(fontsize=12)
    plt.ylabel("AUC (Precision-Recall Curve)", fontsize=12)
    plt.title('Ascl1', fontsize=14)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.5)
    # Setting figure size and saving figure.
    fig.set_size_inches(width, height)
    sns.despine()
    plt.savefig(datapath + '.results.pdf')
    plt.show()

