"""
Utilities for iterating constructing data sets and iterating over
DNA sequence data.
Pybedtools code from:
https://github.com/uci-cbcl/FactorNet/blob/master/utils.py
Pseudo code structure:
1. Construct a random training set (start with a random negative,
   account for data augmentations later?)
2. Load the data & convert it to onehot. (Look for parallelization tools.)
3. Build a generator
What data does this script take as input or require?
1. The genome sizes file
2. The genome-wide fasta file
3. A blacklist regions file.
4. A ChIP-seq peak file.
"""

import numpy as np
import pandas as pd
import pyfasta
from pybedtools import BedTool
import pyBigWig
import argparse
from subprocess import call
import yaml
import subprocess
import os

# local imports
import utils

# pybedtools.set_tempdir('/storage/home/dvs5680/scratch/')
np.random.seed(9)


class AccessGenome:
    def __init__(self, genome_fasta_file):
        self.genome_fasta_file = genome_fasta_file

    def get_genome_fasta(self):
        f = pyfasta.Fasta(self.genome_fasta_file)
        return f

    @staticmethod
    def get_onehot_array(seqs, window_length):
        """
        Parameters:
            seqs: The sequence array that needs to be converted into one-hot encoded
            features.
            batch_size: mini-batch size
            L: window length
        Returns:
            A one-hot encoded array of shape batch_size * window_len * 4
        """
        onehot_map = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'G': [0, 0, 1, 0],
                      'C': [0, 0, 0, 1], 'N': [0, 0, 0, 0]}
        # note: converting all lower-case nucleotides into upper-case here.
        onehot_seqs = [onehot_map[x.upper()] for seq in seqs for x in seq]
        onehot_data = np.reshape(onehot_seqs, newshape=(len(seqs), window_length, 4))
        return onehot_data

    def rev_comp(self, inp_str):
        rc_dict = {'A': 'T', 'G': 'C', 'T': 'A', 'C': 'G', 'c': 'g',
                   'g': 'c', 't': 'a', 'a': 't', 'n': 'n', 'N': 'N'}
        outp_str = list()
        for nucl in inp_str:
            outp_str.append(rc_dict[nucl])
        return ''.join(outp_str)

    def get_data_at_coordinates(self, coordinates_df, genome_fasta,
                                window_len, chromatin_track_list, nbins):
        """
        This method can be used either by:
        1. class ConstructSets: uses this method to return features and labels
           for a training or validation batch.
        2. class ConstructTestData: uses this method to return features and
           labels for the test chromosome co-ordinates and labels.
        Parameters:
            coordinates_df(dataFrame): This method takes as input a Pandas DataFrame with dimensions N * 4
            Where N is the number of samples.
            The columns are: chr, start, stop, label
            genome_fasta (pyFasta npy record): Pyfasta pointer to the fasta file.
            window_len (int): length of windows used for training
            nbins (int): number of bins for chromatin tracks
        Returns:
            This method returns a one hot encoded numpy array (X) and a np
            vector y.
            Both X and y are numpy arrays.
            X shape: (batch size, L, 4)
            y shape: (batch size,)
        """
        y = coordinates_df['label']
        X_seq = []
        seq_len = []

        bw_list = [pyBigWig.open(bw_file) for bw_file in chromatin_track_list]
        chromatin_out_lists = [[] for x in bw_list]

        batch_size = len(y)
        idx = 0
        for chrom, start, stop, lab in coordinates_df.values:
            fa_seq = genome_fasta[chrom][int(start):int(stop)]
            try:
                for idx, bw_file in enumerate(bw_list):
                    chromatin_out_lists[idx].append(bw_file.stats(chrom, start, stop, nBins=nbins))
            except RuntimeError:
                print(
                    "Error while analyzing the BigWig file.\n"
                    "-> Please ensure that the genome sizes file and genome fasta file is compatible "
                    "with the genome to which the BigWig"
                    "data is aligned. \n"
                    "-> It is possible that chromosome names are different in these file types")
                exit(1)
            # Adding reverse complements into the training process:
            if idx <= int(batch_size/2):
                X_seq.append(fa_seq)
            else:
                X_seq.append(self.rev_comp(fa_seq))
            idx += 1
            seq_len.append(len(fa_seq))
        return X_seq, chromatin_out_lists, y


class ConstructTrainingData(AccessGenome):
    """
    Notes:
        chip_coords is the filtered chip_seq file, it either contains only
        train chromosomes or validation chromosomes based on the input.
    """

    def __init__(self, genome_sizes_file, genome_fasta_file, blacklist_file,
                 chip_coords, window_length, exclusion_df,
                 curr_genome_bed, acc_regions_file, chromatin_track_list, nbins):
        super().__init__(genome_fasta_file)
        self.genome_sizes_file = genome_sizes_file
        self.blacklist_file = blacklist_file
        self.chip_coords = chip_coords
        self.L = window_length
        self.exclusion_df = exclusion_df  # This is df, convert to a bdt object.
        self.curr_genome_bed = curr_genome_bed
        # self.curr_genome_bed is is a df, convert to a bdt obj.
        self.acc_regions_file = acc_regions_file
        self.chromatin_track_list = chromatin_track_list
        self.nbins = nbins

    def apply_random_shift(self, coords):
        """
        This function takes as input a set of bed co-ordinates
        It finds the mid-point for each record or Interval in the bed file,
        shifts the mid-point, and generates a window of
        length self.L.
        Calculating the shift:
        For each interval, find the mid-point.
        In this case, multiGPS is outputting 1 bp windows,
        so just taking the "start" as the mid-point.
        For example:
        Asc1.bed record:
        chr18   71940632   71940633
        mid-point: 71940632
        If training window length is L, then we must ensure that the
        peak center is still within the training window.
        Therefore: -L/2 < shift < L/2
        To add in a buffer: -L/2 + 25 <= shift <= L/2 + 25
        # Note: The 50 here is a tunable hyper-parameter.
        Parameters:
            coords(pandas dataFrame): This is an input bedfile
        Returns:
            shifted_coords(pandas dataFrame): The output bedfile with shifted coords
        """
        # defining the random shift
        low = int(-self.L/2 + 25)
        high = int(self.L/2 - 25)
        coords['random_shift'] = np.random.randint(low=low, high=high,
                                                   size=len(coords))

        coords['s_start'] = coords['start'] + coords['random_shift'] - int(self.L/2)
        coords['s_end'] = coords['start'] + coords['random_shift'] + int(self.L/2)
        # making a new dataFrame containing the new shifted coords.
        shifted_coords = coords.loc[:, ('chr', 's_start', 's_end')]
        shifted_coords.columns = ['chr', 'start', 'end']

        return shifted_coords

    def define_coordinates(self):
        """
        Use the chip-seq peak file and the blacklist files to define a bound
        set and an unbound set of sites. The ratio of bound to unbound is 1:N,
        but can be controlled using the parameter "ratio".
        The unbound/negative set is chosen randomly from the genome.(ha)
        """
        # POS. SAMPLES
        # Take a sample from the chip_coords file,
        # Then apply a random shift that returns 500 bp windows.
        # Create a BedTool object for further use.
        bound_sample_size = int(len(self.chip_coords))
        bound_sample = self.chip_coords.sample(n=(bound_sample_size * 5), replace=True)
        bound_sample_w_shift = self.apply_random_shift(bound_sample)
        bound_sample_bdt_obj = BedTool.from_dataframe(bound_sample_w_shift)
        bound_sample_w_shift['label'] = 1


        # NEG. SAMPLES
        # note: the self.curr_genome_bed.fn contains only training chromosomes.
        # Creates a DF.
        curr_genome_bdt = BedTool.from_dataframe(self.curr_genome_bed)
        exclusion_bdt_obj = BedTool.from_dataframe(self.exclusion_df)
        # unbound_random_bdt_obj = BedTool.shuffle(bound_sample_bdt_obj,
        #                                          g=self.genome_sizes_file,
        #                                          incl=curr_genome_bdt.fn,
        #                                          excl=exclusion_bdt_obj.fn)
        unbound_random_bdt_obj = BedTool().random(l=self.L, n=(bound_sample_size * 4),
                                                  g=self.genome_sizes_file)
        # Choosing only unbound regions that lie in the training set
        unbound_random_bdt_obj = unbound_random_bdt_obj.intersect(curr_genome_bdt)
        # Choosing only unbound regions that do not intersect ChIP-seq peaks or blacklist regions
        unbound_random_bdt_obj = unbound_random_bdt_obj.intersect(exclusion_bdt_obj, v=True)
        unbound_random_df = unbound_random_bdt_obj.to_dataframe()[['chrom', 'start', 'end']] # BedTool random produced 6 columns
        unbound_random_df.columns = ['chr', 'start', 'end']
        unbound_random_df['label'] = 0

        # NEG. SAMPLES: FLANKS
        def make_flanks(lower_lim, upper_lim):
            # getting a list of chip-seq flanking windows:
            # (can be a separate fn in utils)
            flanks_left = self.chip_coords.copy()
            flanks_right = self.chip_coords.copy()
            flanks_left['start'] = self.chip_coords['start'] - upper_lim
            flanks_left['end'] = self.chip_coords['start'] - lower_lim
            flanks_right['start'] = self.chip_coords['start'] + lower_lim
            flanks_right['end'] = self.chip_coords['start'] + upper_lim
            return flanks_left, flanks_right

        fl_r, fl_l = make_flanks(lower_lim=250, upper_lim=750)
        fl_r_2, fl_l_2 = make_flanks(lower_lim=200, upper_lim=700)
        fl_r_3, fl_l_3 = make_flanks(lower_lim=1500, upper_lim=2000)
        fl_r_4, fl_l_4 = make_flanks(lower_lim=1000, upper_lim=1500)
        flanks_df = pd.concat([fl_r, fl_l, fl_r_2, fl_l_2, fl_l_3, fl_r_3, fl_r_4, fl_l_4])
        flanks_df = flanks_df[flanks_df['start'] > 0]

        flanks_bdt = BedTool.from_dataframe(flanks_df)
        unbound_flanks_bdt_obj = flanks_bdt.intersect(curr_genome_bdt)
        unbound_flanks_df = unbound_flanks_bdt_obj.to_dataframe()
        unbound_flanks_df.columns = ['chr', 'start', 'end']
        unbound_flanks_df['label'] = 0
        unbound_flanks_df = unbound_flanks_df.sample(frac=1)

        # NEG. SAMPLES: ACCESSIBLE
        regions_acc_bdt_obj = BedTool(self.acc_regions_file)
        regions_acc_bdt_obj = regions_acc_bdt_obj.intersect(curr_genome_bdt)
        # negative samples/pre-accessible
        # unbound_acc_bdt_obj = BedTool.shuffle(bound_sample_bdt_obj,
        #                                       g=self.genome_sizes_file,
        #                                       incl=regions_acc_bdt_obj.fn,
        #                                       excl=exclusion_bdt_obj)
        unbound_acc_bdt_obj = BedTool().random(l=self.L, n=(bound_sample_size * 4),
                                               g=self.acc_regions_file)
        # unbound_acc_bdt_obj = unbound_acc_bdt_obj.intersect(regions_acc_bdt_obj)
        unbound_acc_bdt_obj = unbound_acc_bdt_obj.intersect(exclusion_bdt_obj, v=True)
        unbound_acc_df = unbound_acc_bdt_obj.to_dataframe()[['chrom', 'start', 'end']]
        unbound_acc_df.columns = ['chr', 'start', 'end']
        unbound_acc_df['label'] = 0

        # Sizes of each set in this training construction are already accounted for.
        training_coords = pd.concat([bound_sample_w_shift, unbound_random_df,
                                    unbound_flanks_df, unbound_acc_df])

        training_coords = training_coords[(training_coords['end'] - training_coords['start'] == 500)]
        # randomly shuffle the dataFrame
        training_coords = training_coords.sample(frac=1)
        return training_coords

    def get_data(self):
        # get mini-batch co-ordinates:
        coords_for_data = self.define_coordinates()
        # get the fasta file:
        genome_fasta = super(ConstructTrainingData, self).get_genome_fasta()

        X_seq, X_chromatin_list, y = super().get_data_at_coordinates(coordinates_df=coords_for_data,
                                               genome_fasta=genome_fasta,
                                               window_len=self.L, chromatin_track_list=self.chromatin_track_list,
                                                                     nbins=self.nbins)
        return X_seq, X_chromatin_list, y, coords_for_data


def construct_training_data(genome_sizes_file, peaks_file, genome_fasta_file,
                            blacklist_file, to_keep, to_filter,
                            window_length, acc_regions_file, out_prefix, chromatin_track_list, nbins):
    """
    This generator can either generate training data or validation data based on
    the to_keep and to_filter arguments.
    The train generate uses the to_filter argument, whereas to_keep=None
    For example:
    train_generator:  to_filter=['chr10', 'chr17, 'chrUn', 'chrM', 'random']
    i.e. In this construction; chr10 and chr17 can be used for testing/validation.
    The val generator uses the to_keep argument, whereas to_filter=None.
    For example:
    val_generator: to_keep=['chr17']
    i.e. In this construction; chr17 data is used for validation.
    Additional Parameters:
        genome_sizes_file: sizes
        peaks_file: multiGPS formatted *events* file
        blacklist_file: BED format blacklist file
        genome_fasta_file: fasta file for the whole genome
        batch_size (int): batch size used for training and validation batches
        window_len (int): the length of windows used for training and testing.
    """
    # Load the genome_sizes_file (Filtering out the validation and test chromosomes):
    curr_genome_bed = utils.get_genome_sizes(genome_sizes_file, to_keep=to_keep,
                                             to_filter=to_filter)
    genome_bed_df = curr_genome_bed.to_dataframe()

    # Loading the chip-seq bed file (Filtering out the validation and test chromosomes):
    chip_seq_coordinates = utils.load_chipseq_data(peaks_file,
                                                   genome_sizes_file=genome_sizes_file,
                                                   to_keep=to_keep,
                                                   to_filter=to_filter)

    # Loading the exclusion bed file (Blacklist + ChIP-seq peaks, use for constructing negative sets):
    exclusion_windows_bdt = utils.exclusion_regions(blacklist_file,
                                                                               chip_seq_coordinates)
    exclusion_windows_df = exclusion_windows_bdt.to_dataframe()

    # constructing the training set
    construct_sets = ConstructTrainingData(genome_sizes_file=genome_sizes_file,
                                           genome_fasta_file=genome_fasta_file,
                                           blacklist_file=blacklist_file,
                                           chip_coords=chip_seq_coordinates,
                                           exclusion_df=exclusion_windows_df,
                                           window_length=window_length,
                                           curr_genome_bed=genome_bed_df,
                                           acc_regions_file=acc_regions_file,
                                           chromatin_track_list=chromatin_track_list,
                                           nbins=nbins)

    X_seq, X_chromatin_list, y, training_coords = construct_sets.get_data()
    # saving the data
    np.savetxt(out_prefix + '.seq', X_seq, fmt='%s')
    for idx, chromatin_track in enumerate(chromatin_track_list):
        chromatin_out_files = [x.split('.')[0] for x in chromatin_track_list]
        np.savetxt(out_prefix + '.' + chromatin_out_files[idx] + '.chromatin', X_chromatin_list[idx], delimiter='\t', fmt='%1.3f')
    np.savetxt(out_prefix + '.labels', y, fmt='%s')
    return training_coords


class ConstructTestData(AccessGenome):

    def __init__(self, genome_fasta_file, genome_sizes_file, peaks_file,
                 blacklist_file, window_len, stride, to_keep, chromatin_track_list,
                 nbins):
        super().__init__(genome_fasta_file)
        self.genome_sizes_file = genome_sizes_file
        self.peaks_file = peaks_file
        self.blacklist_file = blacklist_file
        self.window_len = window_len
        self.stride = stride
        self.to_keep = to_keep
        self.chromatin_track_list = chromatin_track_list
        self.nbins = nbins

    def define_coordinates(self):
        """
        This function loads and returns coords & labels for the test set.
        Logic for assigning test set labels:
        The multiGPS peak files are used as inputs; and expanded to record
        25 bp windows around the peak center.
        if 100% of peak center lies in window:
            label bound.
        elif < 100% of peak center lies in the window:
            label ambiguous.
        else:
            label unbound.
        Returns:
            test_coords (pd dataFrame): A dataFrame with chr, start, end and
            labels
        """
        genome_sizes = pd.read_csv(self.genome_sizes_file, sep="\t",
                                   names=['chr', 'len'])
        # subset the test chromosome:
        genome_test = genome_sizes[genome_sizes['chr'] == self.to_keep[0]]
        # the assumption here is that to_keep is a single chromosome list.
        end_idx = genome_test.iloc[0, 1]
        chromosome = genome_test.iloc[0, 0]
        test_set = []
        start_idx = 0
        while start_idx + self.window_len < end_idx:
            curr_interval = [chromosome, start_idx, start_idx + self.window_len]
            start_idx += self.stride
            test_set.append(curr_interval)

        test_df = pd.DataFrame(test_set, columns=['chr', 'start', 'stop'])
        test_bdt_obj = BedTool.from_dataframe(test_df)

        chip_peaks = utils.load_chipseq_data(chip_peaks_file=self.peaks_file,
                                             to_keep=self.to_keep,
                                             genome_sizes_file=self.genome_sizes_file)
        # note: multiGPS reports 1 bp separated start and end,
        # centered on the ChIP-seq peak.
        chip_peaks['start'] = chip_peaks['start'] - int(self.window_len/2)
        # (i.e. 250 if window_len=500 )
        chip_peaks['end'] = chip_peaks['end'] + int(self.window_len/2 - 1)
        # (i.e. 249 if window_len=500); multiGPS reports 1bp intervals

        chip_peaks = chip_peaks[['chr', 'start', 'end']]
        chip_peaks_bdt_obj = BedTool.from_dataframe(chip_peaks)

        blacklist_exclusion_windows = BedTool(self.blacklist_file)
        # intersecting
        unbound_data = test_bdt_obj.intersect(chip_peaks_bdt_obj, v=True)
        if self.blacklist_file is None:
            bound_data = chip_peaks_bdt_obj
        else:
            unbound_data = unbound_data.intersect(blacklist_exclusion_windows,
                                                  v=True)
            # i.e. if there is any overlap with chip_peaks, that window is not
            # reported
            # removing blacklist windows
            bound_data = chip_peaks_bdt_obj.intersect(blacklist_exclusion_windows,
                                                      v=True)
        # i.e. the entire 500 bp window is the positive window.
        # making data-frames
        bound_data_df = bound_data.to_dataframe()
        bound_data_df['label'] = 1
        unbound_data_df = unbound_data.to_dataframe()
        unbound_data_df['label'] = 0
        # exiting
        test_coords = pd.concat([bound_data_df, unbound_data_df])
        return test_coords

    def get_data(self):
        # get mini-batch co-ordinates:
        test_coords = self.define_coordinates()
        # get the fasta file:
        genome_fasta = super().get_genome_fasta()
        X_seq, X_chromatin_list, y = super().get_data_at_coordinates(coordinates_df=test_coords, genome_fasta=genome_fasta,
                                               window_len=self.window_len, chromatin_track_list=self.chromatin_track_list,
                                                                     nbins=self.nbins)
        return X_seq, X_chromatin_list, y, test_coords


def construct_test_data(genome_sizes_file, peaks_file, genome_fasta_file,
                        blacklist_file, to_keep, window_len, stride, out_prefix, chromatin_track_list,
                        nbins):

    ts = ConstructTestData(genome_fasta_file=genome_fasta_file, genome_sizes_file=genome_sizes_file,
                           peaks_file=peaks_file, blacklist_file=blacklist_file,
                           window_len=window_len, stride=stride, to_keep=to_keep,
                           chromatin_track_list=chromatin_track_list, nbins=nbins)
    X_seq, X_chromatin_list, y_test, test_coords = ts.get_data()

    # Saving the data
    np.savetxt(out_prefix + '.seq', X_seq, fmt='%s')
    for idx, chromatin_track in enumerate(chromatin_track_list):
        chromatin_out_files = [x.split('.')[0] for x in chromatin_track_list]
        np.savetxt(out_prefix + '.' + chromatin_out_files[idx] + '.chromatin', X_chromatin_list[idx], delimiter='\t', fmt='%1.3f')
    np.savetxt(out_prefix + '.labels', y_test, fmt='%d')
    test_coords.to_csv(out_prefix + '.bed', sep='\t')
    return test_coords


def main():

    parser = argparse.ArgumentParser(description='Construct Training Data For Bichrom')
    parser.add_argument('-info', help='Genome sizes file',
                        required=True)
    parser.add_argument('-fa', help='The fasta file for the genome of interest',
                        required=True)
    parser.add_argument('-len', help='Size of training, test and validation windows',
                        type=int, required=True)
    parser.add_argument('-acc_domains', help='Bed file with accessible domains',
                        required=True)
    parser.add_argument('-chromtracks', nargs='+', help='A list of BigWig files for all input chromatin '
                        'experiments', required=True)
    parser.add_argument('-peaks', help='A ChIP-seq or ChIP-exo peak file in multiGPS file format',
                        required=True)
    parser.add_argument('-o', '--outdir', help='Output directory for storing train, test data',
                        required=True)
    parser.add_argument('-nbins', type=int, help='Number of bins for chromatin tracks',
                        required=True)

    parser.add_argument('-blacklist', default=None, help='Optional, blacklist file for the genome of interest')

    args = parser.parse_args()

    if args.outdir[0] == '/':
        # The user has specified a full directory path for the output directory:
        out_dir_path = args.outdir
    elif args.outdir[0] == '~':
        # The user has specified a full path starting with the home directory:
        out_dir_path = args.outdir
    elif args.outdir[0] == '.':
        # The user has specified a relative path.
        print("Please specify an absolute path for the output directory.")
        print("Exiting..")
        exit(1)
    else:
        # The user has specified an output directory within the current wd.
        dir_path = subprocess.run(['pwd'], stdout=subprocess.PIPE)
        # Specifying the full path in the yaml configuration file.
        out_dir_path = (str(dir_path.stdout, 'utf-8')).rstrip() + '/' + args.outdir

    print('Creating output directory')
    call(['mkdir', args.outdir])
    print(out_dir_path)

    print('Recording output paths')
    # Produce a default yaml file recording the output
    yml_training_schema = {'train': {'seq': out_dir_path + '/data_train.seq',
                                     'labels': out_dir_path + '/data_train.labels',
                                     'chromatin_tracks': [out_dir_path + '/data_train.' + x.split('.')[0] + '.chromatin'
                                                          for x in args.chromtracks]},
                           'val':   {'seq': out_dir_path + '/data_val.seq',
                                     'labels': out_dir_path + '/data_val.labels',
                                     'chromatin_tracks': [out_dir_path + '/data_train.' + x.split('.')[0] + '.chromatin'
                                                          for x in args.chromtracks]},
                           'test':  {'seq': out_dir_path + '/data_test.seq',
                                     'labels': out_dir_path + '/data_test.labels',
                                     'chromatin_tracks': [out_dir_path + '/data_train.' + x.split('.')[0] + '.chromatin'
                                                          for x in args.chromtracks]}}

    with open(args.outdir + '/bichrom.yaml', "w") as fp:
        yaml.dump(yml_training_schema, fp)

    print('Constructing train data ...')
    coords = construct_training_data(genome_sizes_file=args.info, peaks_file=args.peaks,
                                     genome_fasta_file=args.fa,
                                     blacklist_file=args.blacklist, window_length=args.len,
                                     acc_regions_file=args.acc_domains,
                                     to_filter=['chr17', 'chr11', 'chrM', 'chrUn'],
                                     to_keep=None,
                                     out_prefix=args.outdir + '/data_train',
                                     chromatin_track_list=args.chromtracks,
                                     nbins=args.nbins)

    print('Constructing validation data ...')
    construct_test_data(genome_sizes_file=args.info, peaks_file=args.peaks,
                        genome_fasta_file=args.fa,
                        blacklist_file=args.blacklist, window_len=args.len,
                        stride=args.len,
                        to_keep=['chr11'],
                        out_prefix=args.outdir + '/data_val',
                        chromatin_track_list=args.chromtracks, nbins=args.nbins)

    print('Constructing test data ...')
    construct_test_data(genome_sizes_file=args.info, peaks_file=args.peaks,
                        genome_fasta_file=args.fa,
                        blacklist_file=args.blacklist, window_len=args.len,
                        stride=args.len,
                        to_keep=['chr17'],
                        out_prefix=args.outdir + '/data_test',
                        chromatin_track_list=args.chromtracks, nbins=args.nbins)


if __name__ == "__main__":
    main()