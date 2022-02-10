import argparse
import yaml
from os import remove
from subprocess import call
from train import train_bichrom
from tensorflow.keras.models import load_model
from scan_genome import evaluate_models

if __name__ == '__main__':
    # parsing
    parser = argparse.ArgumentParser(description='Evaluate Bichrom')
    parser.add_argument('-training_schema_yaml', required=True,
                        help='YAML file with paths to train, test and val data')
    parser.add_argument('-mseq', required=True,
                        help='Sequence Model')
    parser.add_argument('-msc', required=True,
                        help='Bichrom Model')
    parser.add_argument('-len', help='Size of genomic windows',
                        required=True, type=int)
    parser.add_argument('-outdir', required=True, help='Output directory')
    parser.add_argument('-dataset', choices=['test', 'train', 'val'], required=True, help='train, test, or val')
    args = parser.parse_args()

    # load the yaml file with input data paths:
    with open(args.training_schema_yaml, 'r') as f:
        try:
            data_paths = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
    # create the output directory:
    outdir = args.outdir
    call(['mkdir', outdir])

    seq_len=args.len
             

    # Evaluate both models on held-out test sets and plot metrics
    probas_out_seq = outdir + '/seqnet/' + args.dataset + '_probs.txt'
    probas_out_sc = outdir + '/bichrom/' + args.dataset + '_probs.txt'
    try:
        remove(probas_out_seq)
        remove(probas_out_sc)
    except OSError:
        pass

    records_file_path = outdir + '/metrics_' + args.dataset
    print(records_file_path)
    #exit(1)
    mseq = load_model(args.mseq)
    msc = load_model(args.msc)
    evaluate_models(sequence_len=seq_len, path=data_paths[args.dataset],
                    probas_out_seq=probas_out_seq, probas_out_sc=probas_out_sc,
                    model_seq=mseq, model_sc=msc,
                    records_file_path=records_file_path)
