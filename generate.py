import argparse
import logging
from pathlib import Path
import pandas as pd
from model.model import load_gpt_model
from model.sampler import sample
import torch
import math


def main(args):
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(f'device:\t{args.device}')

    gpt_path = args.model_path
    out_path = args.out_file

    model_def = Path(gpt_path).with_suffix('.json')
    model = load_gpt_model(model_def, gpt_path, args.device, copy_to_cpu=True)

    logger.info(f'Generate samples...')
    num_to_sample = args.n_samples
    sample_seqs = sample(model, num_to_sample=num_to_sample, device=args.device, batch_size=args.batch_size,
                         max_len=args.max_len)
    uniq_seqs = list(set(sample_seqs))

    logger.info(f"Totally {len(uniq_seqs)} unique sequences!")
    # Save seqs
    df_seqs = pd.DataFrame(sample_seqs, columns=['seqs'])
    df_seqs.to_csv(out_path, index=False)

    logger.info(f'Generation finished!')


def get_args():
    parser = argparse.ArgumentParser(description='Generate SMILES from a GPT model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_path', type=str, help='Full path to GPT model')
    parser.add_argument('--out_file', type=str, help='Output file path')

    optional = parser.add_argument_group('Optional')
    optional.add_argument('--n_samples', default=1000, type=int, help='Molecules to sample, default=1000')
    optional.add_argument('--template', action='store_true', help='Template to use, default=False')
    optional.add_argument('--device', default='cuda', type=str, help='Use cuda or cpu, default=cuda')
    optional.add_argument('--batch_size', default=64, type=int, help='Batch_size during sampling, default=64')
    optional.add_argument('--max_len', default=200, type=int, help='Maximum seqs length, default=13')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(args)
