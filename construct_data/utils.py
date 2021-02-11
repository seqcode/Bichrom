"""
Utilities for iterating constructing data sets and iterating over
DNA sequence data.
"""
import pandas as pd
from pybedtools import Interval, BedTool
import numpy as np


def filter_chromosomes(input_df, to_filter=None, to_keep=None):
    """
    This function takes as input a pandas DataFrame
    Parameters:
        input_df (dataFrame): A pandas dataFrame, the first column is expected to
        be a chromosome. Example: chr1.
        to_filter (list): Default None (bool = False), will iterate over list
        objects and filter the listed chromosomes.
        ( Default: None, i.e. this condition will not be triggered unless a list
        is supplied)
        to_keep (list): Default None, will iterate over list objects and only
        retain the listed chromosomes.
    Returns:
          output_df (dataFrame): The filtered pandas dataFrame
    """
    if to_filter:
        output_df = input_df.copy()
        for chromosome in to_filter:
            # note: using the str.contains method to remove all
            # contigs; for example: chrUn_JH584304
            bool_filter = ~(output_df['chr'].str.contains(chromosome))
            output_df = output_df[bool_filter]
    elif to_keep:
        # keep only the to_keep chromosomes:
        # note: this is slightly different from to_filter, because
        # at a time, if only one chromosome is retained, it can be used
        # sequentially.
        filtered_chromosomes = []
        for chromosome in to_keep:
            filtered_record = input_df[(input_df['chr'] == chromosome)]
            filtered_chromosomes.append(filtered_record)
        # merge the retained chromosomes
        output_df = pd.concat(filtered_chromosomes)
    else:
        output_df = input_df
    return output_df


def get_genome_sizes(genome_sizes_file, to_filter=None, to_keep=None):
    """
    Loads the genome sizes file which should look like this:
    chr1    45900011
    chr2    10001401
    ...
    chrX    9981013
    This function parses this file, and saves the resulting intervals file
    as a BedTools object.
    "Random" contigs, chrUns and chrMs are filtered out.
    Parameters:
        genome_sizes_file (str): (Is in an input to the class,
        can be downloaded from UCSC genome browser)
        to_filter (list): Default None (bool = False), will iterate over list
        objects and filter the listed chromosomes.
        ( Default: None, i.e. this condition will not be triggered unless a list
        is supplied)
        to_keep (list): Default None, will iterate over list objects and only
        retain the listed chromosomes.
    Returns:
        A BedTools (from pybedtools) object containing all the chromosomes,
        start (0) and stop (chromosome size) positions
    """
    genome_sizes = pd.read_csv(genome_sizes_file, sep='\t',
                               header=None, names=['chr', 'length'])

    genome_sizes_filt = filter_chromosomes(genome_sizes, to_filter=to_filter,
                                           to_keep=to_keep)

    genome_bed_data = []
    # Note: Modifying this to deal with unexpected (incorrect) edge case \
    # BedTools shuffle behavior.
    # While shuffling data, BedTools shuffle is placing certain windows at the \
    # edge of a chromosome
    # Why it's doing that is unclear; will open an issue on GitHub.
    # It's probably placing the "start" co-ordinate within limits of the genome,
    # with the end coordinate not fitting.
    # This leads to the fasta file returning an incomplete sequence \
    # (< 500 base pairs)
    # This breaks the generator feeding into Model.fit.
    # Therefore, in the genome sizes file, buffering 550 from the edges
    # to allow for BedTools shuffle to place window without running of the
    # chromosome.
    for chrom, sizes in genome_sizes_filt.values:
        genome_bed_data.append(Interval(chrom, 0 + 550, sizes - 550))
    genome_bed_data = BedTool(genome_bed_data)
    return genome_bed_data


def load_chipseq_data(chip_peaks_file, genome_sizes_file, to_filter=None,
                      to_keep=None):
    """
    Loads the ChIP-seq peaks data.
    The chip peaks file is an events bed file:
    chr1:451350
    chr2:91024
    ...
    chrX:870000
    This file can be constructed using a any peak-caller. We use multiGPS.
    Also constructs a 1 bp long bedfile for each coordinate and a
    BedTools object which can be later used to generate
    negative sets.
    """
    chip_seq_data = pd.read_csv(chip_peaks_file, delimiter=':', header=None,
                             names=['chr', 'start'])
    chip_seq_data['end'] = chip_seq_data['start'] + 1

    chip_seq_data = filter_chromosomes(chip_seq_data, to_filter=to_filter,
                                       to_keep=to_keep)

    sizes = pd.read_csv(genome_sizes_file, names=['chr', 'chrsize'],
                        sep='\t')

    # filtering out any regions that are close enough to the edges to
    # result in out-of-range windows when applying data augmentation.
    chrom_sizes_dict = (dict(zip(sizes.chr, sizes.chrsize)))
    chip_seq_data['window_max'] = chip_seq_data['end'] + 500
    chip_seq_data['window_min'] = chip_seq_data['start'] - 500

    chip_seq_data['chr_limits_upper'] = chip_seq_data['chr'].map(
        chrom_sizes_dict)
    chip_seq_data = chip_seq_data[chip_seq_data['window_max'] <=
                                  chip_seq_data['chr_limits_upper']]
    chip_seq_data = chip_seq_data[chip_seq_data['window_min'] >= 0]
    chip_seq_data = chip_seq_data[['chr', 'start', 'end']]

    return chip_seq_data


def exclusion_regions(blacklist_file, chip_seq_data):
    """
    This function takes as input a bound bed file (from multiGPS).
    The assumption is that the bed file reports the peak center
    For example: chr2   45  46
    It converts these peak centers into 501 base pair windows, and adds them to
    the exclusion list which will be used when constructing negative sets.
    It also adds the mm10 blacklisted windows to the exclusion list.
    Parameters:
        blacklist_file (str): Path to the blacklist file.
        chip_seq_data (dataFrame): The pandas chip-seq data loaded by load_chipseq_data
    Returns:
        exclusion_windows (BedTool): A bedtools object containing all exclusion windows.
        bound_exclusion_windows (BedTool): A bedtool object containing only
        those exclusion windows where there exists a binding site.
    """
    temp_chip_file = chip_seq_data.copy()  # Doesn't modify OG array.
    temp_chip_file['start'] = temp_chip_file['start'] - 250
    temp_chip_file['end'] = temp_chip_file['end'] + 250

    if blacklist_file is None:
        print('No blacklist file specified ...')
        exclusion_windows = BedTool.from_dataframe(temp_chip_file[['chr', 'start','end']])
    else:
        bound_exclusion_windows = BedTool.from_dataframe(temp_chip_file[['chr', 'start','end']])
        blacklist_exclusion_windows = BedTool(blacklist_file)
        exclusion_windows = BedTool.cat(
            *[blacklist_exclusion_windows, bound_exclusion_windows])
    return exclusion_windows
