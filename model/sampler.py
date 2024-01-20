import torch
import torch.nn.functional as F
from utils.utils import set_random_seed
from model.model import GPT
from utils.dataset import HelmDictionary, rnn_start_token_vector
from tqdm import tqdm
import pandas as pd
from loguru import logger

sd = HelmDictionary()


def _sample_batch(model: GPT, batch_size: int, device, max_len, temperature, template=None) -> torch.Tensor:
    x = rnn_start_token_vector(batch_size, device)
    indices = torch.zeros((batch_size, max_len), dtype=torch.long).to(device)
    indices[:, 0] = x.squeeze()

    def get_aa_idx_on_refseq(pos):
        return sd.char_idx[template.refSeq[pos]] if pos < len(template.refSeq) else sd.end_idx

    for char in range(1, max_len):
        if template is not None and (char + 1 not in template.positions):  # Use template if not None
            action = torch.LongTensor(batch_size, 1).fill_(get_aa_idx_on_refseq(char)).to(device)
        else:  # Sampling AA from AA candidates for opt positions
            logits, _ = model(x)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            distribution = torch.distributions.Categorical(probs=probs)
            action = distribution.sample()

        indices[:, char] = action.squeeze()
        x = indices[:, :char + 1]  # assign x with all sequence generated

    return indices


def sample(model: GPT, num_to_sample=10000, device='cpu', batch_size=64, max_len=100, temperature=1.0, seed=42,
           template=None):  # consider to add top_k function
    set_random_seed(seed, device)

    # Round up division to get the number of batches that are necessary:
    number_batches = (num_to_sample + batch_size - 1) // batch_size
    remaining_samples = num_to_sample

    indices = torch.LongTensor(num_to_sample, max_len).to(device)

    model.eval()
    with torch.no_grad():
        batch_start = 0
        for i in tqdm(range(number_batches), desc='Sampling', ncols=80):
            batch_size = min(batch_size, remaining_samples)
            batch_end = batch_start + batch_size

            indices[batch_start:batch_end, :] = _sample_batch(model, batch_size, device, max_len, temperature,
                                                              template=template)

            batch_start += batch_size
            remaining_samples -= batch_size

        return sd.matrix_to_seqs(indices)


def generate_samples(model: GPT, out_path, num_to_sample=1000, device='cpu', batch_size=64, max_len=200, temperature=1.0):
    sample_seqs = sample(model, num_to_sample=num_to_sample, device=device, batch_size=batch_size,
                         max_len=max_len, temperature=temperature)
    uniq_seqs = list(set(sample_seqs))

    df_seqs = pd.DataFrame(uniq_seqs, columns=['seqs'])
    df_seqs.to_csv(out_path, index=False)
    return df_seqs
