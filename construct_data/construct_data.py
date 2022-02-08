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
                                            .pipe(utils.make_random_shift, L))
    bound_sample_bdt_obj = BedTool.from_dataframe(bound_sample_shift).intersect(blacklist_bdt, v=True)
    bound_sample_shift = bound_sample_bdt_obj.to_dataframe().assign(label=1, type="pos_shift")

    bound_sample_acc_size = bound_sample_bdt_obj.intersect(acc_bdt, wa=True).count()
    bound_sample_inacc_size = bound_shift - bound_sample_acc_size
    
    logging.info(f"Bound samples in accessible region: {bound_sample_acc_size}")
    logging.info(f"Bound samples NOT in accessible region: {bound_sample_inacc_size}")

    # NEG. SAMPLES
    # note: the self.curr_genome_bed.fn contains only training chromosomes.
    # NEG. SAMPLES: FLANK
    if unbound_flank_dists is not None:
        unbound_flank_df = pd.concat([utils.make_flank(chip_coords, L, dist) for dist in unbound_flank_dists])
        unbound_flank_df = utils.clean_bed(unbound_flank_df)
        # remove negative flanking samples that happen to be overlapped with adjacent bound sites
        unbound_flank_df = (BedTool().from_dataframe(unbound_flank_df)
                                    .intersect(chip_coords_bdt, v=True)
                                    .to_dataframe()
                                    .assign(label=0, type="neg_flank"))

        logging.debug(f"# Unbound flank sample: {unbound_flank_df.shape[0]}")
    else:
        unbound_flank_df = None

        logging.debug(f"# Unbound flank sample: 0")
    

    # NEG. SAMPLES: ACCESSIBLE
    unbound_acc_df = (utils.random_coords(gs=genome_sizes_file,
                                        l=L, 
                                        n=bound_sample_acc_size if unbound_random_acc is None else unbound_random_acc,
                                        incl=acc_bdt.intersect(curr_genome_bdt),
                                        excl=chip_coords_bdt)
                            .assign(label=0, type="neg_acc"))

    logging.debug(f"# Unbound accessible sample: {unbound_acc_df.shape[0]}")

    # NEG. SAMPLES: INACCESSIBLE
    unbound_inacc_df = (utils.random_coords(gs=genome_sizes_file,
                                            l=L, n=bound_sample_inacc_size if unbound_random_inacc is None else unbound_random_inacc,
                                            incl=curr_genome_bdt,
                                            excl=acc_bdt.cat(chip_coords_bdt))
                                .assign(label=0, type="neg_inacc"))

    logging.debug(f"# Unbound inaccessible sample: {unbound_inacc_df.shape[0]}")

    # concatenate all types of negative samples
    training_coords_neg = pd.concat([unbound_flank_df, unbound_acc_df, unbound_inacc_df])

    # balance ratio of positive and negative samples in accessible regions
    training_coords_neg_acc = (BedTool.from_dataframe(training_coords_neg)
                                        .intersect(blacklist_bdt, v=True)
                                        .intersect(acc_bdt, wa=True)
                                        .to_dataframe()
                                        .sample(n=bound_sample_acc_size)
                                        .rename(columns={"name": "label", "score": "type"}))
    
    training_coords_neg_inacc = (BedTool.from_dataframe(training_coords_neg)
                                        .intersect(blacklist_bdt, v=True)
                                        .intersect(acc_bdt, v=True)
                                        .to_dataframe()
                                        .sample(n=bound_sample_inacc_size)
                                        .rename(columns={"name": "label", "score": "type"}))

    logging.info(f"training coordinates negative samples in accessible regions: {training_coords_neg_acc.shape[0]}")
    logging.info(f"training coordinates negative samples in inaccessible regions: {training_coords_neg_inacc.shape[0]}")

    training_coords = pd.concat([bound_sample_shift, training_coords_neg_acc, training_coords_neg_inacc])

    # clean
    #training_coords = training_coords[(training_coords['end'] - training_coords['start'] == 500)]
    training_coords = utils.clean_bed(training_coords)

    # logging summary
    logging.debug(training_coords.groupby(["label", "type"]).size())

    # randomly shuffle the dataFrame
    training_coords = training_coords.sample(frac=1)
    return training_coords

def construct_training_set(genome_sizes_file, genome_fasta_file, peaks_file, blacklist_file, to_keep, to_filter,
                            window_length, acc_regions_file, out_prefix, chromatin_track_list, nbins, p=1):

    # prepare files for defining coordiantes
    curr_genome_bdt = utils.get_genome_sizes(genome_sizes_file, to_keep=to_keep, to_filter=to_filter)

    chip_seq_coordinates = utils.load_chipseq_data(peaks_file, genome_sizes_file=genome_sizes_file,
                                                    to_keep=to_keep, to_filter=to_filter)

    acc_bdt = BedTool(acc_regions_file)

    blacklist_bdt = BedTool(blacklist_file)

    # get the coordinates for training samples
    train_coords = define_training_coordinates(chip_seq_coordinates, genome_sizes_file, acc_bdt, curr_genome_bdt,
                                blacklist_bdt, window_length, len(chip_seq_coordinates)*5, [450, -450, 500, -500, 1250, -1250, 1750, -1750], None, None)
    train_coords.to_csv(out_prefix + ".bed", header=False, index=False, sep="\t")

    # get fasta sequence and chromatin coverage according to the coordinates
    # write TFRecord output
    TFRecord_file_f = utils.get_data_TFRecord(train_coords, genome_fasta_file, chromatin_track_list, 
                            nbins, outprefix=out_prefix + "_forward" ,reverse=False, numProcessors=p)
    TFRecord_file_r = utils.get_data_TFRecord(train_coords, genome_fasta_file, chromatin_track_list, 
                            nbins, outprefix=out_prefix + "_reverse",reverse=True, numProcessors=p)
    
    return (TFRecord_file_r + TFRecord_file_f)

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
    subprocess.call(['mkdir', args.outdir])
    print(out_dir_path)

    print('Recording output paths')

    print([x.split('/')[-1].split('.')[0] for x in args.chromtracks])

    print('Constructing train data ...')
    TFRecords_train = construct_training_set(genome_sizes_file=args.info, genome_fasta_file=args.fa,
                                    peaks_file=args.peaks,
                                    blacklist_file=args.blacklist, window_length=args.len,
                                    acc_regions_file=args.acc_domains,
                                    to_filter=['chr17', 'chr11', 'chrM', 'chrUn'],
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
                        to_keep=['chr11'],
                        out_prefix=args.outdir + '/data_val',
                        chromatin_track_list=args.chromtracks, nbins=args.nbins, p=args.p)

    print('Constructing test data ...')
    TFRecords_test = construct_test_set(genome_sizes_file=args.info,
                        peaks_file=args.peaks,
                        genome_fasta_file=args.fa,
                        blacklist_file=args.blacklist, window_length=args.len,
                        stride=args.len,
                        to_keep=['chr17'],
                        out_prefix=args.outdir + '/data_test',
                        chromatin_track_list=args.chromtracks, nbins=args.nbins, p=args.p)

    # Produce a default yaml file recording the output
    yml_training_schema = {'train': {'seq': 'seq',
                                     'labels': 'labels',
                                     'chromatin_tracks': args.chromtracks,
                                     'TFRecord': TFRecords_train},
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