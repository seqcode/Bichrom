import argparse
import yaml
import subprocess
import numpy as np
import pandas as pd
from pybedtools import BedTool

import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)

# local imports
import utils

def define_training_coordinates(chip_coords: pd.DataFrame, genome_sizes_file: str,
                                acc_bdt: BedTool, curr_genome_bdt: BedTool,
                                blacklist_bdt: BedTool, L,
                                bound_shift: int, unbound_flank_dists: list, unbound_random_acc: int, 
                                unbound_random_inacc: int):
    """
    Use the chip-seq peak file and the blacklist files to define a bound
    set and an unbound set of sites. The ratio of bound to unbound is 1:N,
    but can be controlled using the parameter "ratio".
    The unbound/negative set is chosen randomly from the genome.(ha)
    """

    chip_coords_bdt = BedTool.from_dataframe(chip_coords.assign(start = lambda x: x["start"]-L/2,
                                                                end = lambda x: x["end"]+L/2-1)
                                                        .astype({"start": int, "end":int}))

    # POS. SAMPLES
    # Take a sample from the chip_coords file,
    # Then apply a random shift that returns 500 bp windows.
    # Create a BedTool object for further use.
    bound_sample_shift = (chip_coords.sample(n=bound_shift, replace=True)
                                            .pipe(utils.make_random_shift, L)
                                            .pipe(utils.clean_bed))
    bound_sample_bdt_obj = BedTool.from_dataframe(bound_sample_shift).intersect(blacklist_bdt, v=True)
    bound_sample_shift = bound_sample_bdt_obj.to_dataframe().assign(type="pos_shift", label=1)

    bound_sample_acc_size = bound_sample_bdt_obj.intersect(acc_bdt, u=True).count()
    bound_sample_inacc_size = bound_sample_bdt_obj.intersect(acc_bdt, v=True).count()
    
    logging.debug(f"Bound samples in total: {bound_sample_bdt_obj.count()}")
    logging.debug(f" Bound samples in accessible region: {bound_sample_acc_size}")
    logging.debug(f" Bound samples NOT in accessible region: {bound_sample_inacc_size}")

    # NEG. SAMPLES
    # note: the self.curr_genome_bed.fn contains only training chromosomes.
    # NEG. SAMPLES: FLANK
    if unbound_flank_dists is not None:
        unbound_flank_df = (pd.concat([utils.make_flank(chip_coords, L, dist) for dist in unbound_flank_dists])
                                .pipe(utils.clean_bed))
        # remove negative flanking samples that happen to be overlapped with adjacent bound sites
        unbound_flank_df = (BedTool().from_dataframe(unbound_flank_df)
                                    .intersect(chip_coords_bdt, v=True)
                                    .intersect(blacklist_bdt, v=True)
                                    .to_dataframe()
                                    .assign(type="neg_flank", label=0))

        logging.debug(f"# Unbound flank sample: {unbound_flank_df.shape[0]}")
    else:
        unbound_flank_df = None

        logging.debug(f"# Unbound flank sample: 0")
    

    # NEG. SAMPLES: ACCESSIBLE
    unbound_acc_df = (utils.random_coords(gs=genome_sizes_file,
                                        l=L, 
                                        n=bound_sample_acc_size if unbound_random_acc is None else unbound_random_acc,  # if unbound_random_acc not set, use # bound samples intersected with acc
                                        incl=acc_bdt.intersect(curr_genome_bdt),
                                        excl=chip_coords_bdt.cat(blacklist_bdt))
                            .assign(type="neg_acc", label=0))

    logging.debug(f"# Unbound accessible sample: {unbound_acc_df.shape[0]}")

    # NEG. SAMPLES: INACCESSIBLE
    unbound_inacc_df = (utils.random_coords(gs=genome_sizes_file,
                                            l=L, n=bound_sample_inacc_size if unbound_random_inacc is None else unbound_random_inacc, # if unbound_random_acc not set, use # bound samples intersected with inacc
                                            incl=curr_genome_bdt,
                                            excl=acc_bdt.cat(chip_coords_bdt).cat(blacklist_bdt))
                                .assign(type="neg_inacc", label=0))

    logging.debug(f"# Unbound inaccessible sample: {unbound_inacc_df.shape[0]}")

    # NEG. SAMPLES: ACROSS WHOLE GENOME 
    # This negative sets will be used for chromatin network training
    unbound_genome_df = (utils.random_coords(gs=genome_sizes_file,
                                            l=L, n=bound_sample_bdt_obj.count(),
                                            incl=curr_genome_bdt,
                                            excl=chip_coords_bdt.cat(blacklist_bdt))
                                .assign(type="neg_genome", label=0))

    # TRAINING SET FOR SEQ NETWORK
    logging.info("Constructing training set for sequence network")
    logging.info("It should satisfy two requirements: 1. Positive and Negative sample size should equal 2. Ratio of accessible region intersection should be balanced")
    # concatenate all types of negative samples
    training_coords_seq_neg = pd.concat([unbound_flank_df, unbound_acc_df, unbound_inacc_df])

    # balance ratio of positive and negative samples in accessible regions
    training_coords_seq_neg_acc = (BedTool.from_dataframe(training_coords_seq_neg)
                                        .intersect(acc_bdt, wa=True)
                                        .to_dataframe()
                                        .sample(n=bound_sample_acc_size)
                                        .rename(columns={"name": "type", "score": "label"}))
    
    training_coords_seq_neg_inacc = (BedTool.from_dataframe(training_coords_seq_neg)
                                        .intersect(acc_bdt, v=True)
                                        .to_dataframe()
                                        .sample(n=bound_sample_inacc_size)
                                        .rename(columns={"name": "type", "score": "label"}))

    logging.debug(f"training coordinates negative samples in accessible regions: {training_coords_seq_neg_acc.shape[0]}")
    logging.debug(f"training coordinates negative samples in inaccessible regions: {training_coords_seq_neg_inacc.shape[0]}")

    training_coords_seq = pd.concat([bound_sample_shift, training_coords_seq_neg_acc, training_coords_seq_neg_inacc])
    training_coords_seq = training_coords_seq.sample(frac=1) # randomly shuffle the dataFrame

    # TRAINING SET FOR BICHROM NETWORK
    training_coords_bichrom = pd.concat([bound_sample_shift, unbound_genome_df])
    training_coords_bichrom = training_coords_bichrom.sample(frac=1) # randomly shuffle the dataFrame

    # logging summary
    logging.debug(training_coords_seq.groupby(["label", "type"]).size())
    logging.debug(training_coords_bichrom.groupby(["label", "type"]).size())

    return training_coords_seq, training_coords_bichrom

def construct_training_set(genome_sizes_file, genome_fasta_file, peaks_file, blacklist_file, to_keep, to_filter,
                            window_length, acc_regions_file, out_prefix, chromatin_track_list, nbins, p=1):

    # prepare files for defining coordiantes
    curr_genome_bdt = utils.get_genome_sizes(genome_sizes_file, to_keep=to_keep, to_filter=to_filter)

    chip_seq_coordinates = utils.load_chipseq_data(peaks_file, genome_sizes_file=genome_sizes_file,
                                                    to_keep=to_keep, to_filter=to_filter)

    acc_bdt = BedTool(acc_regions_file)

    blacklist_bdt = BedTool(blacklist_file)

    # get the coordinates for training samples
    train_coords_seq, train_coords_bichrom = define_training_coordinates(chip_seq_coordinates, genome_sizes_file, acc_bdt, curr_genome_bdt,
                                blacklist_bdt, window_length, len(chip_seq_coordinates)*5, [450, -450, 500, -500, 1250, -1250, 1750, -1750], None, None)
    train_coords_seq.to_csv(out_prefix + "_seq.bed", header=False, index=False, sep="\t")
    train_coords_bichrom.to_csv(out_prefix + "_bichrom.bed", header=False, index=False, sep="\t")

    # get fasta sequence and chromatin coverage according to the coordinates
    # write TFRecord output
    TFRecord_file_seq_f = utils.get_data_TFRecord(train_coords_seq, genome_fasta_file, chromatin_track_list, 
                            nbins, outprefix=out_prefix + "_seq_forward" ,reverse=False, numProcessors=p)
    TFRecord_file_seq_r = utils.get_data_TFRecord(train_coords_seq, genome_fasta_file, chromatin_track_list, 
                            nbins, outprefix=out_prefix + "_seq_reverse",reverse=True, numProcessors=p)
    TFRecord_file_bichrom_f = utils.get_data_TFRecord(train_coords_bichrom, genome_fasta_file, chromatin_track_list, 
                         nbins, outprefix=out_prefix + "_bichrom_forward" ,reverse=False, numProcessors=p)
    TFRecord_file_bichrom_r = utils.get_data_TFRecord(train_coords_bichrom, genome_fasta_file, chromatin_track_list, 
                            nbins, outprefix=out_prefix + "_bichrom_reverse",reverse=True, numProcessors=p)
    
    return TFRecord_file_seq_f + TFRecord_file_seq_r, TFRecord_file_bichrom_f + TFRecord_file_bichrom_r

def construct_test_set(genome_sizes_file, genome_fasta_file, peaks_file, blacklist_file, to_keep,
                        window_length, stride, out_prefix, chromatin_track_list, nbins, p=1):

    # prepare file for defining coordinates
    blacklist_bdt = BedTool(blacklist_file)

    # get the coordinates for test samples
    bound_chip_peaks = (utils.load_chipseq_data(peaks_file, genome_sizes_file=genome_sizes_file,
                                                to_keep=to_keep)
                            .assign(start = lambda x: x["start"] - int(window_length/2),
                                    end = lambda x: x["end"] + int(window_length/2-1)))

    bound_chip_peaks_bdt = BedTool.from_dataframe(bound_chip_peaks).intersect(blacklist_bdt, v=True)
    bound_chip_peaks = bound_chip_peaks_bdt.to_dataframe().assign(label=1, type="pos_peak")
    
    unbound_genome_chop = (utils.chop_genome(genome_sizes_file, to_keep, excl=bound_chip_peaks_bdt.cat(blacklist_bdt), stride=stride, l=window_length)
                                .assign(label=0, type="neg_chop"))
    
    test_coords = pd.concat([bound_chip_peaks, unbound_genome_chop])
    test_coords.to_csv(out_prefix + ".bed", header=False, index=False, sep="\t")

    # write TFRecord output
    TFRecord_file = utils.get_data_TFRecord(test_coords, genome_fasta_file, chromatin_track_list, 
                            nbins, outprefix=out_prefix + "_forward" ,reverse=False, numProcessors=p)    

    return TFRecord_file

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
    parser.add_argument('-p', type=int, help='Number of processors', default=1)

    parser.add_argument('-blacklist', default=None, help='Optional, blacklist file for the genome of interest')

    parser.add_argument('-val_chroms', default=['chr11'], nargs='+', help='A list of chromosomes to use for the validation set.')

    parser.add_argument('-test_chroms', default=['chr17'], nargs='+', help='A list of chromosomes to use for the test set')

    args = parser.parse_args()

    if len(set.intersection(set(args.val_chroms), set(['chrM', 'chrUn']))) or len(set.intersection(set(args.test_chroms), set(['chrM', 'chrUn']))) :
        raise ValueError("Validation and Test Sets must not use chrM, chrUn")

    if len(set.intersection(set(args.val_chroms), set(args.test_chroms))):
        raise ValueError("Validation and Test Sets must not have any intersection")

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
    subprocess.call(['mkdir', args.outdir])
    print(out_dir_path)

    print('Recording output paths')

    print([x.split('/')[-1].split('.')[0] for x in args.chromtracks])

    print('Constructing train data ...')
    TFRecords_train_seq, TFRecords_train_bichrom = construct_training_set(genome_sizes_file=args.info, genome_fasta_file=args.fa,
                                    peaks_file=args.peaks,
                                    blacklist_file=args.blacklist, window_length=args.len,
                                    acc_regions_file=args.acc_domains,
                                    to_filter=args.val_chroms + args.test_chroms + ['chrM', 'chrUn'],
                                    to_keep=None,
                                    out_prefix=args.outdir + '/data_train',
                                    chromatin_track_list=args.chromtracks,
                                    nbins=args.nbins, p=args.p)

    print('Constructing validation data ...')
    TFRecords_val = construct_test_set(genome_sizes_file=args.info,
                        peaks_file=args.peaks,
                        genome_fasta_file=args.fa,
                        blacklist_file=args.blacklist, window_length=args.len,
                        stride=args.len,
                        to_keep=args.val_chroms,
                        out_prefix=args.outdir + '/data_val',
                        chromatin_track_list=args.chromtracks, nbins=args.nbins, p=args.p)

    print('Constructing test data ...')
    TFRecords_test = construct_test_set(genome_sizes_file=args.info,
                        peaks_file=args.peaks,
                        genome_fasta_file=args.fa,
                        blacklist_file=args.blacklist, window_length=args.len,
                        stride=args.len,
                        to_keep=args.test_chroms,
                        out_prefix=args.outdir + '/data_test',
                        chromatin_track_list=args.chromtracks, nbins=args.nbins, p=args.p)

    # Produce a default yaml file recording the output
    yml_training_schema = {'train_seq': {'seq': 'seq',
                                     'labels': 'labels',
                                     'chromatin_tracks': args.chromtracks,
                                     'TFRecord': TFRecords_train_seq},
                           'train_bichrom': {'seq': 'seq',
                                     'labels': 'labels',
                                     'chromatin_tracks': args.chromtracks,
                                     'TFRecord': TFRecords_train_bichrom},
                           'val':   {'seq': 'seq',
                                     'labels': 'labels',
                                     'chromatin_tracks': args.chromtracks,
                                     'TFRecord': TFRecords_val},
                           'test':  {'seq': 'seq',
                                     'labels': 'labels',
                                     'chromatin_tracks': args.chromtracks,
                                     'TFRecord': TFRecords_test}}

    # Note: The x.split('/')[-1].split('.')[0] accounts for input chromatin bigwig files with
    # associated directory paths

    with open(args.outdir + '/bichrom.yaml', "w") as fp:
        yaml.dump(yml_training_schema, fp)

if __name__ == "__main__":
    main()
