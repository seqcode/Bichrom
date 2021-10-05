"""
Utilities for constructing training & test data.
Pybedtools code adapted from:
https://github.com/uci-cbcl/FactorNet/blob/master/utils.py

What data does this script take as input or require?
1. The genome sizes file
2. The genome-wide fasta file
3. A blacklist regions file.
4. A ChIP-seq peak file.
"""

import numpy as np
import pandas as pd
import yaml
import pyfasta
import pyBigWig
import argparse
import subprocess
from pybedtools import BedTool
from subprocess import call

# local modules
import utils

# set seed
np.random.seed(9)


class AccessGenome(object):
    def __init__(self,
                 genome_fasta_file,
                 window_length,
                 genome_sizes_file,
                 blacklist_file,
                 chip_seq_coords,
                 exclusion_bdt,
                 curr_genome_bdt,
                 acc_regions_file,
                 chromatin_track_list,
                 nbins):

        self.window_length = int(window_length)  # type: int
        self.py_fasta = pyfasta.Fasta(genome_fasta_file)  # type: PyFasta # TODO
        self.genome_sizes_file = genome_sizes_file  # type: str
        self.blacklist_file = blacklist_file  # type: str
        self.chip_seq_coords = chip_seq_coords  # type: pd.DataFrame
        self.exclusion_bdt = exclusion_bdt  # type: BedTool
        self.curr_genome_bdt = curr_genome_bdt  # type: BedTool
        self.acc_regions_file = acc_regions_file  # TODO
        self.chromatin_track_list = chromatin_track_list  # TODO
        self.nbins = nbins  # type: int

    @staticmethod
    def reverse_complement_sequence(input_str):
        rc_dict = {'A': 'T', 'G': 'C', 'T': 'A', 'C': 'G', 'c': 'g',
                   'g': 'c', 't': 'a', 'a': 't', 'n': 'n', 'N': 'N'}
        output_str = list()
        for nucleotide in input_str:
            output_str.append(rc_dict[nucleotide])
        rev_complemented_str = ''.join(output_str[::-1])  # reverse the string
        return rev_complemented_str

    @staticmethod
    def reverse_chromatin(input_array):
        output_arr = input_array[::-1]  # reverse array (to go with a reverse complemented sequence str)
        return output_arr

    def get_data_at_coordinates(self, coordinates_df, chromatin_track_list):
        """
        Parameters
        ----------
        coordinates_df : pd.DataFrame
            Pandas DataFrame with dimensions N * 4, where N is the number of samples.
            The columns are: chr, start, stop, label

        Returns
        -------
        list_of_sequences : list
        chromatin_out_lists : list
        labels : pd.Series
        """
        labels = coordinates_df['label']  # TODO: Is this a pandas Series object, is that what we want here?
        list_of_sequences = []

        bw_list = [pyBigWig.open(bw_file) for bw_file in chromatin_track_list]
        chromatin_out_lists = [[] for x in bw_list]

        for chromosome, start, stop, label in coordinates_df.values:
            fasta_sequence = self.py_fasta[chromosome][int(start):int(stop)]
            try:
                for idx, bw_file in enumerate(bw_list):
                    chromatin_out_lists[idx].append(bw_file.stats(chromosome, start, stop, nBins=self.nbins))
            except RuntimeError:
                print(
                    "Error while analyzing the BigWig file.\n"
                    "-> Please ensure that the genome sizes file and genome fasta file is compatible "
                    "with the genome to which the BigWig"
                    "data is aligned. \n"
                    "-> It is possible that chromosome names are different in these file types")
                exit(1)

            list_of_sequences.append(fasta_sequence)
        return list_of_sequences, chromatin_out_lists, labels

    def apply_random_shift(self, augmented_coordinates):
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

        Parameters
        ----------
        augmented_coordinates: pd.DataFrame

        Returns
        -------
        shifted_coords : pd.DataFrame
            The output dataframe with shifted coords
        """
        low = int(-1 * self.window_length/2 + 25)
        high = int(self.window_length/2 - 25)
        augmented_coordinates['random_shift'] = np.random.randint(low=low, high=high,
                                                                  size=len(augmented_coordinates))
        augmented_coordinates['s_start'] = augmented_coordinates['start'] + augmented_coordinates['random_shift'] - int(self.window_length/2)
        augmented_coordinates['s_end'] = augmented_coordinates['start'] + augmented_coordinates['random_shift'] + int(self.window_length/2)
        # making a new dataFrame containing the new shifted coords.
        shifted_coords = augmented_coordinates.loc[:, ('chr', 's_start', 's_end')]
        shifted_coords.columns = ['chr', 'start', 'end']
        return shifted_coords

    def construct_positive_set(self, scaling_factor):
        # Total size of positive set: scaling_factor * bound_set_size + bound_set_size (original unshifted data)
        bound_sites = self.chip_seq_coords
        # shift peaks to augment data:
        augmented_bound_sites = self.chip_seq_coords.sample(n=(len(bound_sites) * scaling_factor), replace=True)
        shifted_bound_sites = self.apply_random_shift(augmented_bound_sites)
        # concatenate centered and shifted intervals/windows
        # TODO: check if this works
        positive_set = pd.concat([bound_sites, shifted_bound_sites], axis=0)
        positive_set['label'] = 1
        return positive_set

    def subset_sites(self, input_bdt, remove_chip_intervals=True):
        # Choosing only unbound regions that lie in the training set
        subset_bdt = input_bdt.intersect(self.curr_genome_bdt)
        if remove_chip_intervals:
            # Choosing only unbound regions that do not intersect ChIP-seq peaks or blacklist regions
            subset_bdt = subset_bdt.intersect(self.exclusion_bdt, v=True)
        return subset_bdt

    def construct_randomly_sampled_negative_set(self, scaling_factor):
        unbound_random_bdt = BedTool().random(l=self.window_length,
                                              n=(len(self.chip_seq_coords) * scaling_factor),
                                              g=self.genome_sizes_file)
        # filter for sites within the training chromosomes and exclude blacklists + chip peaks
        unbound_random_bdt_filtered = self.subset_sites(unbound_random_bdt)
        # annotate the data
        unbound_random_df = unbound_random_bdt_filtered.to_dataframe()[['chrom', 'start', 'end']]
        unbound_random_df.columns = ['chr', 'start', 'end']
        unbound_random_df['label'] = 0
        return unbound_random_df

    def extract_flanking_regions(self, lower_lim, upper_lim):
        # getting a list of chip-seq flanking windows:
        flanks_left = self.chip_seq_coords.copy()
        flanks_right = self.chip_seq_coords.copy()
        flanks_left['start'] = self.chip_seq_coords['start'] - upper_lim
        flanks_left['end'] = self.chip_seq_coords['start'] - lower_lim
        flanks_right['start'] = self.chip_seq_coords['start'] + lower_lim
        flanks_right['end'] = self.chip_seq_coords['start'] + upper_lim
        return flanks_left, flanks_right

    def construct_flanks_based_negative_set(self):
        fl_r, fl_l = self.extract_flanking_regions(lower_lim=250, upper_lim=750)
        fl_r_2, fl_l_2 = self.extract_flanking_regions(lower_lim=200, upper_lim=700)
        fl_r_3, fl_l_3 = self.extract_flanking_regions(lower_lim=1500, upper_lim=2000)
        fl_r_4, fl_l_4 = self.extract_flanking_regions(lower_lim=1000, upper_lim=1500)
        flanks_df = pd.concat([fl_r, fl_l, fl_r_2, fl_l_2, fl_l_3, fl_r_3, fl_r_4, fl_l_4])
        flanks_df = flanks_df[flanks_df['start'] > 0]

        flanks_bdt = BedTool.from_dataframe(flanks_df)
        # filter for sites within the training chromosomes and exclude blacklists + chip peaks
        unbound_flanks_bdt = self.subset_sites(flanks_bdt, remove_chip_intervals=True)
        # annotate the data
        unbound_flanks_df = unbound_flanks_bdt.to_dataframe()
        unbound_flanks_df.columns = ['chr', 'start', 'end']
        unbound_flanks_df['label'] = 0
        unbound_flanks_df = unbound_flanks_df.sample(frac=1)
        return unbound_flanks_df

    def construct_accessible_negative_set(self, scaling_factor):
        bound_sample_size = int(len(self.chip_seq_coords))
        unbound_acc_bdt = BedTool().random(l=self.window_length,
                                           n=(bound_sample_size * scaling_factor),
                                           g=self.acc_regions_file)
        # filter for sites within the training chromosomes and exclude blacklists + chip peaks
        unbound_acc_bdt = self.subset_sites(unbound_acc_bdt)
        # annotate the data
        unbound_acc_df = unbound_acc_bdt.to_dataframe()[['chrom', 'start', 'end']]
        unbound_acc_df.columns = ['chr', 'start', 'end']
        unbound_acc_df['label'] = 0

    def define_coordinates(self, bichrom):
        if not bichrom:
            training_coords = pd.concat([self.construct_positive_set(scaling_factor=4),
                                         self.construct_randomly_sampled_negative_set(scaling_factor=4),
                                         self.construct_flanks_based_negative_set(),
                                         self.construct_accessible_negative_set(scaling_factor=4)])
        else:
            training_coords = pd.concat([self.construct_positive_set(scaling_factor=4),
                                         self.construct_randomly_sampled_negative_set(scaling_factor=8)])

        training_coords = training_coords[(training_coords['end'] - training_coords['start'] == 500)]
        # randomly shuffle the dataFrame
        training_coords = training_coords.sample(frac=1)
        return training_coords

    def get_data(self, data_for_bichrom):
        coordinates_df = self.define_coordinates(bichrom=data_for_bichrom)
        list_of_sequences, chromatin_list, y = self.get_data_at_coordinates(
            coordinates_df=coordinates_df,
            chromatin_track_list=self.chromatin_track_list)
        return list_of_sequences, chromatin_list, y, coordinates_df


def construct_training_data(genome_sizes_file, peaks_file, genome_fasta_file,
                            blacklist_file, to_keep, to_filter,
                            window_length, acc_regions_file, out_prefix, chromatin_track_list, nbins,
                            data_for_bichrom=False):
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
    curr_genome_bdt = utils.get_genome_sizes(genome_sizes_file, to_keep=to_keep,
                                             to_filter=to_filter)
    # Loading the chip-seq bed file (Filtering out the validation and test chromosomes):
    chip_seq_coordinates = utils.load_chipseq_data(peaks_file,
                                                   genome_sizes_file=genome_sizes_file,
                                                   to_keep=to_keep,
                                                   to_filter=to_filter)
    # Loading the exclusion bed file (Blacklist + ChIP-seq peaks, use for constructing negative sets):
    exclusion_windows_bdt = utils.exclusion_regions(blacklist_file, chip_seq_coordinates)

    # constructing the training set
    construct_sets = AccessGenome(genome_sizes_file=genome_sizes_file,
                                  genome_fasta_file=genome_fasta_file,
                                  blacklist_file=blacklist_file,
                                  chip_seq_coords=chip_seq_coordinates,
                                  exclusion_bdt=exclusion_windows_bdt,
                                  window_length=window_length,
                                  curr_genome_bdt=curr_genome_bdt,
                                  acc_regions_file=acc_regions_file,
                                  chromatin_track_list=chromatin_track_list,
                                  nbins=nbins)

    list_of_sequences, chromatin_lists, y, training_coords = construct_sets.get_data(data_for_bichrom)
    # saving the data
    np.savetxt(out_prefix + '.seq', list_of_sequences, fmt='%s')
    for idx, chromatin_track in enumerate(chromatin_lists):
        chromatin_out_files = [x.split('/')[-1].split('.')[0] for x in chromatin_track_list]
        np.savetxt(out_prefix + '.' + chromatin_out_files[idx] + '.chromatin', chromatin_lists[idx], delimiter='\t', fmt='%1.3f')
    np.savetxt(out_prefix + '.labels', y, fmt='%s')
    return training_coords


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

    print([x.split('/')[-1].split('.')[0] for x in args.chromtracks])

    # Produce a default yaml file recording the output
    yml_training_schema = {'train': {'seq': out_dir_path + '/data_train.seq',
                                     'labels': out_dir_path + '/data_train.labels',
                                     'chromatin_tracks': [out_dir_path + '/data_train.' + x.split('/')[-1].split('.')[0] + '.chromatin'
                                                          for x in args.chromtracks]},
                           'val':   {'seq': out_dir_path + '/data_val.seq',
                                     'labels': out_dir_path + '/data_val.labels',
                                     'chromatin_tracks': [out_dir_path + '/data_val.' + x.split('/')[-1].split('.')[0] + '.chromatin'
                                                          for x in args.chromtracks]},
                           'test':  {'seq': out_dir_path + '/data_test.seq',
                                     'labels': out_dir_path + '/data_test.labels',
                                     'chromatin_tracks': [out_dir_path + '/data_test.' + x.split('/')[-1].split('.')[0] + '.chromatin'
                                                          for x in args.chromtracks]}}
    # Note: The x.split('/')[-1].split('.')[0] accounts for input chromatin bigwig files with
    # associated directory paths

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


if __name__ == "__main__":
    main()
