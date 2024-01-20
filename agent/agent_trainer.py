import os
from loguru import logger
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from utils.utils import unique
from utils.dataset import HelmDictionary, rnn_start_token_vector
from model.model import save_gpt_model
from prior.trainer import TrainerConfig
from agent.scoring_functions import ScoringFunctions
import random

def biased_bce_with_logits(reg1, reg2, y, bias=1.0):
    # Apply the log-sum-exp trick.
    # y = 1 if we prefer x1 to x2, that is we have regrets for x2

    logit21 = reg2 - bias * reg1
    logit12 = reg1 - bias * reg2
    max21 = torch.clamp(-logit21, min=0, max=None)
    max12 = torch.clamp(-logit12, min=0, max=None)
    nlp21 = torch.log(torch.exp(-max21) + torch.exp(-logit21 - max21)) + max21
    nlp12 = torch.log(torch.exp(-max12) + torch.exp(-logit12 - max12)) + max12
    loss = y * nlp21 + (1 - y) * nlp12
    loss = loss.mean()

    # Now compute the accuracy
    with torch.no_grad():
        accuracy = ((reg2 > reg1) == torch.round(y)).float().mean()

    return loss, accuracy

class AgentTrainer:
    def __init__(self, prior_model, agent_model, save_dir, device='cuda', learning_rate=1e-4, batch_size=64,
                 n_steps=3000, sigma=60, score_type='sum', score_fns=None, max_seq_len=100,
                 score_weights=None, save_per_n_steps=100, multi_gpu=True, loss_type='reinvent', alpha=1,
                 use_amp=True):
        logger.info("Initializing agent trainer ...")

        self.save_dir = save_dir
        self.device = device
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.save_per_n_steps = save_per_n_steps
        self.sigma = sigma
        self.hd = HelmDictionary()
        self.loss_type = loss_type
        self.alpha = alpha
        self.use_amp = use_amp

        self.prior_model = prior_model.module if hasattr(prior_model, "module") else prior_model
        self.agent_model = agent_model.module if hasattr(agent_model, "module") else agent_model
        self.tconf = TrainerConfig(learning_rate=self.learning_rate, lr_decay=True)
        self.optimizer = self.agent_model.configure_optimizers(self.tconf)  # Use adamW with lr_decay

        if multi_gpu:  # Enable using multiple GPUs
            self.agent_model = torch.nn.DataParallel(self.agent_model)

        self.max_seq_len = max_seq_len
        self.score_type = score_type
        self.score_fns = ['permeability'] if score_fns is None else score_fns
        self.scoring_function = ScoringFunctions(scoring_func_names=self.score_fns, score_type=self.score_type,
                                                 weights=score_weights)
        self.writer = SummaryWriter(self.save_dir)

    def save_step(self, step, scores_df, agent_likelihoods, prior_likelihoods, augmented_likelihoods):
        """
            Save step to a CSV file
        """
        scores_df['step'] = step * np.ones(len(scores_df))
        scores_df['agent_likelihood'] = agent_likelihoods.data.cpu().numpy()
        scores_df['prior_likelihood'] = prior_likelihoods.data.cpu().numpy()
        scores_df['augmented_likelihood'] = augmented_likelihoods.data.cpu().numpy()
        scores_df.to_csv(os.path.join(self.save_dir, f"step_{step}_aa_seqs.csv"), index=False)

    def nll_loss(self, inputs, targets):
        """
            Custom Negative Log Likelihood loss that returns loss per example, rather than for the entire batch.

            Args:
                inputs : (batch_size, num_classes) *Log probabilities of each class*
                targets: (batch_size) *Target class index*

            Outputs:
                loss : (batch_size) *Loss for each example*
        """
        target_expanded = torch.zeros(inputs.size()).to(inputs.device)
        target_expanded.scatter_(1, targets.contiguous().view(-1, 1).detach(), 1.0)  # One_hot encoding
        loss = torch.sum(target_expanded * inputs, 1)
        return loss

    def sample(self, model, num_samples: int):
        """
            Sample molecules from agent and calculate likelihood
            Args:
                model: model to sample from
                num_samples: number of samples to produce for each step, i.e. batch_size
            Returns:
                sample_idxes: a list of SMILES indexes, with no beginning nor end symbols
                log_probs: log likelihood for SMILES generated
            """
        sequences = []

        x = rnn_start_token_vector(num_samples, self.device)
        sequences.append(x)
        finished = torch.zeros(num_samples).byte().to(x.device)
        log_probs = torch.zeros(num_samples).to(x.device)
        for step in range(self.max_seq_len):
            logits, _ = model(x)
            probs = F.softmax(logits[:, -1, :], dim=-1)  # only for last time-step
            
            sampled_idx = Categorical(probs=probs).sample().squeeze()
            sequences.append(sampled_idx.view(-1, 1))
            x = torch.cat(sequences, 1)  # assign x with all sequence generated
            log_probs += self.nll_loss(probs.log(), sampled_idx)  # update log_probs

            # Stop if EOS sampled
            end_sampled = (sampled_idx == self.hd.end_idx).detach()  # Check if end in current step
            finished = torch.ge(finished + end_sampled, 1)  # Check if all ends sampled
            if torch.prod(finished) == 1:
                break

        sequences = torch.cat(sequences, 1)

        return sequences.detach(), log_probs

    def likelihood(self, model, x):
        """
        Retrieves the likelihood of a given sequence
            Args: x
                model: GPT model to calculate likelihood
                x: A tensor of aa sequence index, with (num_samples, seq_length) shape
            Outputs:
                log_probs : (batch_size) Log likelihood for each example
        """
        num_samples, seq_length = x.size()
        log_probs = torch.zeros(num_samples).to(x.device)

        model.eval()
        with torch.no_grad():
            for step in range(1, seq_length):
                logits, _ = model(x[:, :step])
                log_prob = F.log_softmax(logits[:, -1, :], dim=-1).squeeze()
                log_probs += self.nll_loss(log_prob, x[:, step])
        return log_probs

    def _reinvent_loss(self, agent_likelihoods, prior_likelihoods, scores):
        """
            Calculate loss for REINVENT
            Args:
                agent_likelihoods: likelihoods of agent
                prior_likelihoods: likelihoods of prior
                scores: scores of agent

            Returns:
                loss: loss for REINVENT
        """

        augmented_likelihoods = (prior_likelihoods + self.sigma * torch.from_numpy(scores).to(agent_likelihoods.device))
        loss = torch.pow((augmented_likelihoods - agent_likelihoods), 2)

        loss = loss.mean()
        loss -= 5 * 1e3 * (1 / agent_likelihoods).mean()  # Penalize small likelihoods, stimulate learning
        return loss, augmented_likelihoods
    
    def _cpl_loss(self, agent_likelihoods, prior_likelihoods, scores):
        """
            Calculate loss for CPL
            Args:
                agent_likelihoods: likelihoods of agent
                prior_likelihoods: likelihoods of prior
                scores: scores of agent

            Returns:
                loss: loss for CPL
        """
        regrets = prior_likelihoods - agent_likelihoods
        reg1, reg2 = torch.chunk(regrets, 2, dim=0)
        score1, score2 = torch.chunk(torch.from_numpy(scores).to(agent_likelihoods.device), 2, dim=0)
        # handle non-even cases
        if reg1.shape[0] != reg2.shape[0]:
            reg1 = reg1[:-1]
            score1 = score1[:-1]
        labels = (score1 > score2).float()
        loss, accuracy = biased_bce_with_logits(reg1, reg2, labels, bias=0.5)
        return loss, accuracy
    
    def _run_step(self, step):
        sample_idxes, agent_likelihoods = self.sample(self.agent_model, self.batch_size)  # Sample from agent
        uniq_ids = unique(sample_idxes)  # Remove duplicates
        uniq_token_seqs = sample_idxes[uniq_ids]

        agent_likelihoods = agent_likelihoods[uniq_ids].to(self.device)
        prior_likelihoods = self.likelihood(self.prior_model, uniq_token_seqs).to(agent_likelihoods.device)
        aa_seqs = self.hd.matrix_to_seqs(uniq_token_seqs)
        scores_df = self.scoring_function.scores(aa_seqs, step)
        scores = scores_df[self.score_type].to_numpy()

        if self.loss_type == 'reinvent':
            loss, augmented_likelihoods = self._reinvent_loss(agent_likelihoods, prior_likelihoods, scores)
        elif self.loss_type == 'cpl':
            loss, augmented_likelihoods = self._cpl_loss(agent_likelihoods, prior_likelihoods, scores)
        elif self.loss_type == 'reinvent_cpl':
            reinvent_loss, augmented_likelihoods = self._reinvent_loss(agent_likelihoods, prior_likelihoods, scores)
            cpl_loss, _ = self._cpl_loss(agent_likelihoods, prior_likelihoods, scores)
            loss = reinvent_loss + self.alpha * cpl_loss

        else:
            raise Exception(f"Unknown loss type: {self.loss_type}, only reinvent and cpl are supported.")
        return loss, scores, agent_likelihoods, prior_likelihoods, augmented_likelihoods, scores_df


    def train(self):
        for param in self.prior_model.parameters():  # Don't update Prior
            param.requires_grad = False

        logger.info("Start training agent...")
        for step in range(self.n_steps):
            if self.device == 'cuda':
                with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.use_amp):
                    loss, scores, agent_likelihoods, prior_likelihoods, augmented_likelihoods, scores_df = self._run_step(step)
                    loss = loss.mean()
            else:
                loss, scores, agent_likelihoods, prior_likelihoods, augmented_likelihoods, scores_df = self._run_step(step)
            
            self.optimizer.zero_grad()
            loss.backward() # retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.agent_model.parameters(), self.tconf.grad_norm_clip)
            self.optimizer.step()

            goals = [scores_df[f'raw_{goal}'].to_numpy().mean() for goal in self.score_fns]
            [self.writer.add_scalar(goal, score, step + 1) for goal, score in zip(self.score_fns, goals)]
            print_goals = ', '.join([f'{goal}: {score:6.2f}' for goal, score in zip(self.score_fns, goals)])

            score = scores.mean()
            logger.info(f"Step {step}, Avg score: {score:6.2f}, {print_goals}")
            self.writer.add_scalar('Avg score', score, step + 1)

            self.save_step(step, scores_df, agent_likelihoods, prior_likelihoods, augmented_likelihoods)

            if step % self.save_per_n_steps == 0 and step != 0:  # save model every save_per_n_steps steps
                save_gpt_model(self.agent_model, self.save_dir, f'Agent_{step}_{score:.3f}')

        save_gpt_model(self.agent_model, self.save_dir, f'Agent_final_{score:.3f}')
