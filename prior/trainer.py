"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import math
import time
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model.model import GPT, GPTConfig, save_gpt_model, save_gpt_config
from model.sampler import generate_samples
from utils.utils import time_since, get_path
from utils.helm_utils import get_validity
from loguru import logger


class TrainerConfig:
    # optimization parameters
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1  # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6  # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9  # (at what point we reach 10% of original LR)
    # checkpoint settings
    output_dir = '../'

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Trainer:

    def __init__(self, model: GPT, config: TrainerConfig):
        self.model = model
        self.config = config
        self.writer = SummaryWriter(self.config.output_dir)

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def fit(self, train_dataset, test_dataset,
            n_epochs=10, batch_size=64, num_workers=0,
            save_model=False, check_validity=True):
        model, config = self.model, self.config

        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)
        start_time = time.time()

        def run_epoch(split):
            is_train = split == 'train'
            model.train(is_train)
            data = train_dataset if is_train else test_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=batch_size,
                                num_workers=num_workers)

            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader), ncols=120) if is_train else enumerate(loader)
            for it, (x, y) in pbar:

                # place data on the correct device
                x = x.to(self.device)
                y = y.to(self.device)

                # forward the model
                with torch.set_grad_enabled(is_train):
                    logits, loss = model(x, y)
                    loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())

                if is_train:

                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()

                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (y >= 0).sum()  # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(
                                max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    # report progress
                    pbar.set_description(f"epoch {epoch + 1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")

            loss = float(np.mean(losses))
            logger.info(f'{split}, elapsed: {time_since(start_time)}, epoch: {epoch + 1}/{n_epochs}, loss: {loss:.4f}')
            self.writer.add_scalar('loss', loss, epoch + 1)

            if not is_train:
                return loss  # return test loss

        best_loss = float('inf')
        self.tokens = 0  # counter used for learning rate decay
        for epoch in range(n_epochs):

            run_epoch('train')
            if test_dataset is not None:
                test_loss = run_epoch('test')

            # check validity of samples
            if check_validity:
                validity = self._check_validity(config.output_dir, f'gpt_model_{epoch+1}', num_to_sample=100)
                logger.info(f'100 samples, epoch: {epoch + 1}/{n_epochs}, validity: {validity:.4f}')
                self.writer.add_scalar('validity', validity, epoch + 1)

            # supports early stopping based on the test loss, or just save always if no test set is provided
            good_model = test_dataset is None or test_loss < best_loss
            if self.config.output_dir is not None and good_model and save_model:
                best_loss = test_loss
                self._save_model(self.config.output_dir, str(epoch+1), test_loss)

        if self.config.output_dir is not None and 'test_loss' in locals() and save_model:
            self._save_model(self.config.output_dir, 'final', test_loss)
    
    def _save_model(self, base_dir, info, valid_loss):
        """
        Save a copy of the model with format:
                gpt_model_{info}_{valid_loss}
        """
        base_name = f'gpt_model_{info}_{valid_loss:.3f}'
        logger.info(f'Save model {base_name}')
        save_gpt_model(self.model, base_dir, base_name)
    
    def _check_validity(self, base_dir, base_name, num_to_sample=100, sample_batch_size=100, max_len=200):
        out_path = get_path(base_dir, base_name, f'_{num_to_sample}_samples.csv')
        df_seqs = generate_samples(self.model, out_path, num_to_sample=num_to_sample, device=self.device, 
                                   batch_size=sample_batch_size, max_len=max_len) 
        validity = get_validity(df_seqs.seqs)
        return validity
