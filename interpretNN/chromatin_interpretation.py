import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joint_embeddings import get_embeddings_low_mem


def get_bed_quartiles(datapath, embedding, out_path):
    # Order the data based on chromatin scores
    order = np.argsort(embedding[:, 1])
    # Splitting the bedfile into quartiles based on chromatin scores.
    # This will be used eventually by the composite plotter.
    bedfile = np.loadtxt(datapath + '.bound.bed', dtype=str)
    bedfile = bedfile[order][::-1]
    file_len = len(bedfile)
    # saving the quartile split bedfiles
    np.savetxt(out_path + 'Q1.bound.bed', bedfile[:file_len/4], fmt='%s', delimiter='\t')
    np.savetxt(out_path + 'Q2.bound.bed', bedfile[file_len/4:file_len/2], fmt='%s', delimiter='\t')
    np.savetxt(out_path + 'Q3.bound.bed', bedfile[file_len/2:3*file_len/4], fmt='%s', delimiter='\t')
    np.savetxt(out_path + 'Q4.bound.bed', bedfile[3*file_len/4:], fmt='%s', delimiter='\t')


def sum_heatmap(datapath, input_data, out_path):
    # Get total tag enrichment for each modification
    labels = ['ATACSEQ', 'H3K27ac', 'H3K27me3', 'H3K4me1', 'H3K4me2', 'H3K4me3',
              'H3K9ac', 'H3K9me3', 'H3K36me3', 'H2AZ', 'acH2AZ', 'H4K20me3']
    X, C = input_data
    # Processing the chromatin
    # Basically, summing along the bins and along the individual loci to get sum tag counts
    C = np.reshape(C, (-1, 13, 10))
    C = np.sum(C, axis=2)
    C = np.sum(C, axis=0)
    C = list(C)
    # Removing H3 as I'm using it as a control instead.
    del C[-2]
    # Figure: Plotting the total tag enrichment for each modification
    fig, ax = plt.subplots()
    # Adjusting the margins
    fig.subplots_adjust(left=.01, bottom=.50, right=.99, top=.60)
    # Defining figure size:
    width = 4
    height = 2
    sns.heatmap(np.transpose(C).reshape(1, 12)/len(X), cmap="Greys",
                xticklabels=labels, cbar_kws=dict(use_gridspec=False, location="top"))
    # Making a border for the heat-map
    for _, spine in ax.spines.items():
        spine.set_visible(True)
    plt.xticks(rotation=45)
    plt.yticks([], [])
    # Setting the axes positions for the color bar
    cax = ax.figure.axes[-1]
    pos1 = ax.get_position()  # get the original position
    pos2 = [pos1.x0 + 0.2, pos1.y0 + 0.2, pos1.width / 2.0, pos1.height / 2.0]
    cax.set_position(pos2)
    ax.figure.axes[-1].set_xlabel('Average Tags Per Million', size=12)
    # Setting figure size and saving the figure.
    fig.set_size_inches(width, height)
    plt.savefig(out_path + 'Enrichment.pdf')
    plt.show()


def scores_at_domains(model, datapath, out_path):
    # load the entire chromatin data
    chromatin = np.loadtxt(datapath + '.chromtracks')
    domains = np.loadtxt(datapath + '.domains')
    # Label order is from the chromatin design file
    labels = ['ATACSEQ', 'H3K27ac', 'H3K27me3', 'H3K4me1', 'H3K4me2', 'H3K4me3',
              'H3K9ac', 'H3K9me3', 'H3K36me3', 'H2AZ', 'acH2AZ', 'H4K20me3']
    print chromatin.shape
    domain_scores = []
    for idx, label in enumerate(labels):
        enrichment = domains[:, idx]
        curr_chrom = chromatin[enrichment == 1]
        # Take care of the experiments with NO domain calls
        if len(curr_chrom) == 0:
            chrom = np.zeros(shape=(1000, 130))
            seq = np.zeros(shape=(1000, 500, 4))
            # raise an exception here..
        else:
            chrom = curr_chrom
            seq = np.zeros(shape=(len(curr_chrom), 500, 4))
        curr_input = (seq, chrom)
        embeddings = get_embeddings_low_mem(model, curr_input)
        chrom_scores = embeddings[:, 1]
        curr = np.vstack((chrom_scores, np.repeat(label, repeats=len(chrom_scores))))
        curr = np.transpose(curr)
        domain_scores.append(curr)

    dat = np.vstack(domain_scores)
    # Saving the data here in case I want to use it further:-
    np.savetxt(out_path + 'Chrom_scores_at_domains.txt', dat, fmt='%s')
    # Figure
    dat = np.loadtxt(out_path + 'Chrom_scores_at_domains.txt', dtype=str)
    # Defining figure sizes
    width = 4
    height = 3
    # Set style with seaborn
    sns.set_style('ticks')
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)
    plt.rc('axes', labelsize=12)
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=.15, bottom=.25, right=.97, top=.97)
    sns.boxplot(x=dat[:, 1], y=dat[:, 0].astype(float), color='#99A3A4',
                linewidth=1, showfliers=False)
    plt.xticks(range(12), labels, ha='right', rotation=45)
    plt.ylim(-5, 5)
    fig.set_size_inches(width, height)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.5)
    plt.savefig(out_path + 'Figure5A.pdf')


def scores_at_states(model, datapath, out_path):
    # print "Loading chromatin..."
    chromatin = np.loadtxt(datapath + '.chromtracks')
    annotation = np.loadtxt(datapath + '.chromHMM.annotation', dtype=str)
    # Dummy placeholder sequence tensor
    sequence = np.zeros(shape=(len(annotation), 500, 4))
    # Labels (Not considering E12-low signal/repetitive for now)
    labels = ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8',
              'E9', 'E10', 'E11']

    label_names = ['CTCF', 'Quiescent', 'Heterochromatin', 'Enhancer',
                   'Repressed Chromatin', 'Bivalent Promoters', 'Active Promoter', 'Strong Enhancer',
                   'Transcriptional Transition', 'Transcriptional Elongation',
                   'Weak/Poised Enhancers']  # chromatin states

    state = annotation[:, 0]
    domain_scores = []
    median_scores = []
    for idx, label in enumerate(labels):
        curr_chrom = chromatin[state == label]
        curr_seq = sequence[state == label]
        curr_input = (curr_seq, curr_chrom)
        embeddings = get_embeddings_low_mem(model, curr_input)
        chrom_scores = embeddings[:, 1]
        seq_scores = embeddings[:, 0]
        median_scores.append(np.median(chrom_scores))
        curr = np.vstack((chrom_scores, seq_scores, np.repeat(label, repeats=len(chrom_scores))))
        curr = np.transpose(curr)
        domain_scores.append(curr)

    order = np.argsort(median_scores)
    domain_scores = np.array(domain_scores)[order][::-1]
    dat = np.vstack(domain_scores)
    # Saving the data here in case I want to use it further:-
    np.savetxt(out_path + 'chrom_scores_at_states.txt', dat, fmt='%s')

    print "In here..."
    dat = np.loadtxt(out_path + 'chrom_scores_at_states.txt', dtype=str)
    print "Done loading.."
    print dat
    # Defining figure sizes
    width = 4
    height = 3.5
    # Set style with seaborn
    sns.set_style('ticks')
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=.1, bottom=.1, right=.99, top=.97)
    print dat[2, :]
    sns.boxplot(x=dat[:, 0].astype(float), y=dat[:, 2], saturation=0.5, showfliers=False, palette='Blues')
    fig.set_size_inches(width, height)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.2)
    plt.savefig(out_path + 'Figure5C_chromatin.pdf')
    return order


def seq_scores_at_states(model, datapath, input_data, out_path, order):
    # Loading the input sequence:-
    sequence, chromatin = input_data
    print "Done loading ..."
    annotation = np.loadtxt(datapath + '.bound.chromHMM.annotation',
                            dtype=str)  # domains containing idx associated with each window
    state = annotation[:, 0]
    # Taking out E12 cause of no binding here..
    labels = ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8',
              'E9', 'E10', 'E11']  # chromatin states

    label_names = ['CTCF', 'Quiescent', 'Heterochromatin', 'Enhancer',
                   'Repressed Chromatin', 'Bivalent Promoters', 'Active Promoter', 'Strong Enhancer',
                   'Transcriptional Transition', 'Transcriptional Elongation',
                   'Weak/Poised Enhancers']

    domain_scores = []
    median_scores = []
    for idx, label in enumerate(labels):
        curr_chrom = chromatin[state == label]
        curr_seq = sequence[state == label]
        # seq = np.zeros(shape=(min(len(curr_chrom), 500), 500, 4))
        seq = curr_seq[:min(len(curr_seq), 500)]
        chrom = curr_chrom[:min(len(curr_chrom), 500)]
        #
        curr_input = (seq, chrom)
        embeddings = get_embeddings_low_mem(model, curr_input)
        seq_scores = embeddings[:, 0]
        median_scores.append((np.mean(seq_scores)))
        curr = np.vstack((seq_scores, np.repeat(label, repeats=len(seq_scores))))
        curr = np.transpose(curr)
        domain_scores.append(curr)

    # order = np.argsort(median_scores)
    domain_scores = np.array(domain_scores)[order][::-1]
    domain_scores = np.vstack(domain_scores)
    np.savetxt(out_path + 'Seq_scores_at_states.txt', domain_scores, fmt='%s')

    domain_scores = np.loadtxt(out_path + 'Seq_scores_at_states.txt', dtype=str)
    # Defining figure sizes
    width = 4
    height = 3.5
    # Set style with seaborn
    sns.set_style('ticks')
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=.1, bottom=.1, right=.99, top=.97)
    sns.boxplot(y=domain_scores[:, 1], x=domain_scores[:, 0].astype(float), showfliers=False,
                palette='Blues', saturation=0.5)
    # plt.xticks(range(11), labels_ordered, ha='right', rotation=45)
    fig.set_size_inches(width, height)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.2)
    plt.savefig(out_path + 'Figure5C_sequence.pdf')
