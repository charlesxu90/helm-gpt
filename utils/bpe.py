"""
Byte Pair Encoder.
Adapted from https://github.com/karpathy/minGPT/blob/master/mingpt/bpe.py

Originally from openai's gpt2 encoder.py:
https://github.com/openai/gpt-2/blob/master/src/encoder.py
"""

import os
import json
import regex as re
import argparse
from pathlib import Path
import pandas as pd
from utils.utils import parse_config
from collections import Counter, defaultdict
from loguru import logger


def get_pairs(word):
    """ return bigrams from word """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

class BPEEncoder:

    def __init__(self):
        self.preserved_tokens = ['<PAD>', '<SOS>', '<EOS>', '<MASK>', '<UNK>', '<SEP>', '<CLS>']
        self.encoder = None
        self.decoder = None
        self.bpe_ranks = None
        self.cache = {}
    
    def get_vocab_size(self) -> int:
        return len(self.encoder)

    def bpe(self, token):
        """ use self.bpe_ranks to iteratively merge all the possible bpe tokens """
        if token in self.cache:
            return self.cache[token]

        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token
        
        while True:
            # get bigram with smallest index, will return the first in pairs if not match
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf'))) 
            if bigram not in self.bpe_ranks:
                break # break if nothing to be merged
            first_token, second_token = bigram

            new_word = []
            i = 0
            while i < len(word):  # iteratively merge all pairs occurred in word
                try:
                    j = word.index(first_token, i)  # find the next occurence of first_token
                    new_word.extend(word[i:j])      # add the jump region unchanged
                    i = j

                    if word[i] == first_token and i < len(word)-1 and word[i+1] == second_token:
                        new_word.append(first_token+second_token)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                except:
                    new_word.extend(word[i:])       # add the rest region unchanged if not match
                    break
            
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word) # update pairs after merging
    
        word = ' '.join(word)
        self.cache[token] = word  # update cache
        return word

    def encode(self, text):
        """ string to integers """
        tokens = self.bpe(text).split(' ')
        # logger.info(f"tokens: {tokens}")
        bpe_idx = [self.encoder[bpe_token] for bpe_token in tokens]
        return bpe_idx

    def decode(self, bpe_idx):
        """ list of integers to a string """
        tokens = [self.decoder[token] for token in bpe_idx]
        text = ''.join(tokens)
        return text
    
    def _initialize_bpe(self, encoder, bpe_merges):
        """ Initialize BPE encoder and decoder from a list of bpe merges """
        self.encoder = encoder
        self.decoder = {v:k for k,v in self.encoder.items()}
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
    
    @classmethod
    def load(cls, save_dir):
        """ load encoder from save_dir """
        bpe_encoder = cls()

        with open(os.path.join(save_dir, 'encoder.json'), 'r') as f:
            encoder = json.load(f)
        with open(os.path.join(save_dir, 'vocab.bpe'), 'r') as f:
            bpe_merges = [tuple(merge_str.split()) for merge_str in f.read().split('\n')[:-1]]
        
        bpe_encoder._initialize_bpe(encoder, bpe_merges)
        return bpe_encoder
    
    def _init_vocab(self, str_list):
        """ initialize vocab with all unique characters """
        unique_chars = set()
        for s in str_list:
            unique_chars.update(list(s))
        
        logger.info(f"{len(unique_chars)} unique characters in dataset!")
        # initialize bpe with preserved_tokens
        unique_chars = unique_chars.union(set(''.join(self.preserved_tokens)))  
        vocab = self.preserved_tokens + sorted(list(unique_chars)) 
        bpe_merges = []
        for token in self.preserved_tokens:
            for i in range(len(token)-1):
                if (token[:i+1], token[i+1]) in bpe_merges:
                    continue
                logger.info(f"{(token[:i+1], token[i+1])}")
                bpe_merges.append((token[:i+1], token[i+1]))

        return vocab, bpe_merges

    @staticmethod
    def _update_bpe_vocab(str_list, vocab,  bpe_merges, vocab_size=100, bpe_per_merge=1):
        word_counts = Counter(re.findall("[a-zA-Z]+", ' '.join(str_list)))
        logger.info(f"{len(word_counts)} unique words in dataset!")
        word_encodings = {word: [c for c in word] for word in word_counts.keys()}

        num_bpe = vocab_size - len(vocab)
        for _ in range(num_bpe):
            bp_counts = defaultdict(int)
            bp_words = defaultdict(list)
            for word, encoding in word_encodings.items():
                for start_token, end_token in zip(encoding[:-1], encoding[1:]):
                    bp_merged = start_token + end_token
                    if bp_merged not in vocab:
                        bp_counts[(start_token, end_token)] += word_counts[word]
                        bp_words[(start_token, end_token)].append(word)

            if len(bp_counts) == 0:
                break

            best_bp = sorted(bp_counts, key=bp_counts.get, reverse=True)[:bpe_per_merge]
            for (start_token, end_token) in best_bp:
                bp_pre_merge = start_token + ' ' + end_token
                bp_merged = start_token + end_token

                bpe_merges.append((start_token, end_token))
                vocab.append(bp_merged)
                for word in bp_words[(start_token, end_token)]:
                    word_encodings[word] = " ".join(word_encodings[word]).replace(bp_pre_merge, bp_merged).split(" ")
        
        return vocab, bpe_merges
    
    def _learn_bpe_vocab(self, str_list, vocab_size=100):
        """ learn bpe vocab from a list of strings """
        vocab, bpe_merges = self._init_vocab(str_list)
        vocab, bpe_merges = self._update_bpe_vocab(str_list, vocab, bpe_merges, vocab_size)
        encoder = {char: idx for idx, char in enumerate(vocab)}
        self._initialize_bpe(encoder, bpe_merges)
    
    @classmethod
    def from_corpus(cls, corpus, save_dir='./bpe', vocab_size=500):
        """ train bpe from a list of strings """
        bpe_encoder = cls()
        bpe_encoder._learn_bpe_vocab(corpus, vocab_size)
        bpe_encoder.save(save_dir)
        return bpe_encoder
    
    def save(self, save_dir='./bpe'):
        """ save bpe merges to save_dir """
        with open(os.path.join(save_dir, 'encoder.json'), 'w') as f:
            json.dump(self.encoder, f)
        
        with open(os.path.join(save_dir, 'vocab.bpe'), 'w') as f:
            for bp in self.bpe_ranks.keys():
                f.write(f"{bp[0]}\t {bp[1]}\n")


def train(str_list, save_dir='./bpe'):
    """ train bpe from a list of strings """
    encoder = BPEEncoder.from_corpus(str_list, save_dir)
    logger.info(f"encoder.bpe_ranks: {encoder.bpe_ranks}")
    logger.info(f"encoder.encoder: {encoder.encoder}")

    return encoder

def test(text, save_dir='./bpe'):
    encoder = BPEEncoder.load(save_dir)
    logger.info(f"text: {text}")
    idx = encoder.encode(text)
    logger.info(f"idx: {idx}")
    text2 = encoder.decode(idx)
    logger.info(f"text2: {text2}")

    assert text == text2, "text and decoded text should be the same"


def main(args, config):

    df_data = pd.read_csv(config.data)
    data = df_data[config.column].tolist()

    # train tokenizer
    train(data, args.output_dir)

    test(data[0], args.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='result/helm_bpe/helm_bpe.yaml')
    parser.add_argument('--output_dir', default='result/helm_bpe/')
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    config = parse_config(args.config)
    main(args, config)
