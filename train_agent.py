import copy
import os
import argparse
import logging
import json
from pathlib import Path
import torch
from agent.agent_trainer import AgentTrainer
from model.model import load_gpt_model
from utils.dataset import HelmDictionary

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def load_pretrain_model(model_path, device='cuda'):
    logger.info("Loading pretrained model")
    model_def = Path(model_path).with_suffix('.json')
    logger.info(f"Loading prior & agent to device {device}")
    try:
        return load_gpt_model(model_def, model_path, device, copy_to_cpu=False)
    except:
        raise Exception(f"Device '{device}' or model path '{model_path}' not found")


def train_agent(prior_path, agent_path, output_dir, device='cuda', task='all', lr=1e-4, batch_size=64,
                n_steps=3000, sigma=60, save_per_n_steps=100, max_len=140, loss_type='reinvent', alpha=0.5,
                use_amp=True,):
    sd = HelmDictionary()
    n_characters = sd.get_char_num()

    if not os.path.exists(prior_path):
        raise Exception(f"prior model doesn't exist in path {prior_path}")

    prior = load_pretrain_model(prior_path, device)
    agent = load_pretrain_model(agent_path, device) if agent_path else copy.deepcopy(prior)

    logger.info(f"Start training agent! Results will be saved to folder {args.output_dir}")

    score_type = 'weight'
    if task == 'permeability':
        score_fns, score_weights = ['permeability'], [1.]
    elif task == 'kras_kd':
        score_fns, score_weights = ['kras_kd'], [1.]
    elif task == 'kras_perm':
        score_fns, score_weights = ['permeability', 'kras_kd'], [1., 1.]
        score_type = 'sum'
    else:
        raise Exception(f"type must be 'permeability', 'kras_kd' or 'kras_perm', received {type}!")

    trainer = AgentTrainer(prior_model=prior, agent_model=agent, save_dir=output_dir,
                           device=device, learning_rate=lr, batch_size=batch_size, n_steps=n_steps,
                           sigma=sigma, score_type=score_type, max_seq_len=max_len,
                           score_fns=score_fns, score_weights=score_weights,
                           save_per_n_steps=save_per_n_steps, loss_type=loss_type, alpha=alpha,
                           use_amp=use_amp)
    trainer.train()


def main(args):
    logger.info(f'device: {args.device}, totally {torch.cuda.device_count()} GPUs, '
                f'{torch.cuda.get_device_properties(0).total_memory // 1024 ** 3} GB per one')
    logger.info('Training gpt agent started!')

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(os.path.join(args.output_dir, 'commandline_args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    prior_path = None if not args.prior else args.prior

    train_agent(prior_path, args.agent, output_dir=args.output_dir, task=args.task,
                device=args.device, lr=args.lr, batch_size=args.batch_size, n_steps=args.n_steps,
                sigma=args.sigma, save_per_n_steps=args.save_per_n_steps, max_len=args.max_len, 
                loss_type=args.loss_type, alpha=args.alpha, use_amp=not(args.no_amp))
    logger.info(f"Training agent finished! Results saved to folder {args.output_dir}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prior', '-p', type=str, help='Path to prior checkpoint (.pt)')
    parser.add_argument('--output_dir', '-d', type=str, help='Output directory')

    optional = parser.add_argument_group('Optional')
    parser.add_argument('--agent', '-a', type=str, default=None, help='Path to agent checkpoint, prior by default')
    optional.add_argument('--task', default='permeability', type=str,
                          help="Running type, must be 'permeability' or 'all', default='permeability'")
    optional.add_argument('--batch_size', type=int, default=64, help='Batch size (default is 64)')
    optional.add_argument('--n_steps', type=int, default=3000, help='Number of training steps (default is 3000)')
    optional.add_argument('--sigma', type=int, default=60,
                          help='Sigma value used to calculate augmented likelihood (default is 60)')
    optional.add_argument('--device', default='cuda', type=str, help='Use cuda or cpu, default=cuda')
    optional.add_argument('--lr', default=1e-4, type=float, help='Learning rate, default=1e-4')
    optional.add_argument('--save_per_n_steps', default=100, type=int,
                          help='Save checkpoint every n steps, default=100')
    optional.add_argument('--max_len', default=140, type=int, help='Max sequence length, default=140')
    optional.add_argument('--loss_type', default='reinvent_cpl', type=str, help='Loss type: reinvent, cpl or reinvent_cpl, default=reinvent_cpl')
    optional.add_argument('--alpha', default=1.0, type=float, help='Alpha value for cpl loss, default=1.0')
    optional.add_argument('--no_amp', action='store_true', help='Use automatic mixed precision, default=False')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
