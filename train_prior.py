import logging
import argparse
import json
import os
import pandas as pd
from typing import List
import torch
from model.model import GPT, GPTConfig, load_gpt_model
from prior.trainer import Trainer, TrainerConfig
from utils.utils import set_random_seed
from utils.dataset import load_seqs_from_list, get_tensor_dataset, HelmDictionary
from pathlib import Path
import shutil
from model.sampler import sample


logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger.addHandler(logging.NullHandler())


def load_pretrain_model(prior_path, device='cuda'):
    logger.info("Loading pretrained models")
    model_def = Path(prior_path).with_suffix('.json')
    logger.info(f"Loading prior & agent to device {device}")
    try:
        prior = load_gpt_model(model_def, prior_path, device, copy_to_cpu=False)
        return prior
    except:
        raise Exception(f"Device '{device}' or model not available")


def train(training_set: List[str], validation_set: List[str], output_dir, n_epochs=10, lr=1e-3, batch_size=64,
          n_layer=8, n_embd=512, n_head=8, max_len=100, device='cpu', num_workers=1, seed=42, model_path=None, 
          num_to_sample=1000, sample_batch_size=64):
    logger.info(f'Running device:\t{device}')
    device = torch.device(device)
    set_random_seed(seed, device)

    # load data
    train_seqs, _ = load_seqs_from_list(training_set, max_len=max_len, rm_duplicates=False)
    valid_seqs, _ = load_seqs_from_list(validation_set, max_len=max_len, rm_duplicates=False)

    train_set = get_tensor_dataset(train_seqs)
    test_set = get_tensor_dataset(valid_seqs)

    sd = HelmDictionary()
    n_characters = sd.get_char_num()
    block_size = max_len + 2  # add start & end

    # build network
    if model_path:
        model = load_pretrain_model(model_path, device=device)
    else:
        mconf = GPTConfig(n_characters, block_size=block_size, n_layer=n_layer, n_head=n_head, n_embd=n_embd)
        model = GPT(mconf)

    # initialize a trainer instance and kick off training
    tconf = TrainerConfig(learning_rate=lr, lr_decay=True, warmup_tokens=0.1 * len(train_set) * max_len,
                          final_tokens=n_epochs * len(train_set) * max_len, output_dir=output_dir)
    trainer = Trainer(model, tconf)
    trainer.fit(train_set, test_set, n_epochs=n_epochs, batch_size=batch_size, num_workers=num_workers, save_model=True)
    
    logger.info(f'Training done, the trained model is in {output_dir}')
    return trainer.model


def main(args):
    df_train = pd.read_csv(args.train_data)
    df_valid = pd.read_csv(args.valid_data)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(os.path.join(args.output_dir, 'commandline_args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    column = 'HELM'

    logger.info(f"Training prior model started, the results are saved in {args.output_dir}")
    train(training_set=df_train[column].tolist(), validation_set=df_valid[column].tolist(),
                  output_dir=args.output_dir, n_epochs=args.n_epochs, lr=args.lr, batch_size=args.batch_size,
                  n_layer=args.n_layers, n_embd=args.n_embd, n_head=args.n_head,
                  device=args.device, max_len=args.max_len, model_path=args.model_path,)



def parse_args():
    parser = argparse.ArgumentParser(description='Train prior RNN model on sequence',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train_data', '-t', type=str, help='Full path to sequence file containing training data')
    parser.add_argument('--valid_data', '-v', type=str, help='Full path to sequence file containing validation data')
    parser.add_argument('--output_dir', '-o', type=str, help='Output directory')

    optional = parser.add_argument_group('Optional')
    optional.add_argument('--n_epochs', default=10, type=int, help='Number of training epochs, default=10')
    optional.add_argument('--lr', default=1e-3, type=float, help='RNN learning rate, default=1e-3')
    optional.add_argument('--n_layers', default=8, type=int, help='Number of layers for training, default=8')
    optional.add_argument('--batch_size', default=512, type=int, help='Size of batch for training, default=512')
    optional.add_argument('--n_embd', default=256, type=int, help='Number of embeddings for GPT model, default=256')
    optional.add_argument('--n_head', default=8, type=int, help='Number of attention heads for GPT model, default=8')
    optional.add_argument('--device', default='cuda', type=str, help='Use cuda or cpu, default=cuda')
    optional.add_argument('--max_len', default=100, type=int, help='Max length of a sequence, default=100')
    optional.add_argument('--model_path', default=None, type=str, help='Prior model path to fine-tune')
    optional.add_argument('--num_to_sample', default=1000, type=int, help='Molecules to sample, default=1000')    
    optional.add_argument('--sample_batch_size', default=512, type=int, help='Batch size for sampling, default=512')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
