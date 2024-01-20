# Adapted from https://github.com/MolecularAI/Reinvent
import io
import subprocess
import warnings
import numpy as np
import pandas as pd
from typing import List
from loguru import logger
from tqdm import tqdm

from agent.scoring.permeability import Permeability
from agent.scoring.kras import KRASInhibition
from agent.scoring.kras_ic50 import KRASIC50
from utils.helm_utils import get_cycpep_smi_from_helm


class ScoringFunctions:
    def __init__(self, scoring_func_names=None, score_type='weight', weights=None):
        """
            scoring_func_names: List of scoring function names, default=['HER2']
            weights: List of int weights for each scoring function, default=[1]
        """
        self.scoring_func_names = ['permeability'] if scoring_func_names is None else scoring_func_names
        self.score_type = score_type
        self.weights = np.array([1] * len(self.scoring_func_names) if weights is None else weights)
        self.all_funcs = {'permeability': Permeability, 'kras_kd': KRASInhibition, 'kras_ic50': KRASIC50,}

    def scores(self, helm_seqs: List, step: int):
        scores, raw_scores = [], []
        for fn_name in self.scoring_func_names:
            # logger.debug(f"Scoring function: {fn_name}")
            score, raw_score = self.all_funcs[fn_name]()(helm_seqs)

            scores.append(score)
            raw_scores.append(raw_score)
        scores = np.float32(scores).T
        raw_scores = np.float32(raw_scores).T
        # logger.debug(f"Scores: {scores}, raw scores: {raw_scores}")

        if self.score_type == 'sum':
            final_scores = scores.sum(axis=1)
        elif self.score_type == 'product':
            final_scores = scores.prod(axis=1)
        elif self.score_type == 'weight':
            final_scores = (scores * self.weights / self.weights.sum()).sum(axis=1)
        else:
            raise Exception('Score type error!')

        np_step = np.ones(len(helm_seqs)) * step
        # logger.debug(f"Final scores: {final_scores}")
        scores_df = pd.DataFrame({'step': np_step, 'helm_seqs': helm_seqs, self.score_type: final_scores})
        scores_df[self.scoring_func_names] = pd.DataFrame(scores, index=scores_df.index)
        raw_names = [f'raw_{name}' for name in self.scoring_func_names]
        scores_df[raw_names] = pd.DataFrame(raw_scores, index=scores_df.index)
        return scores_df


def create_helm_from_aa_seq(aa_seq, cyclic=False):
    linear_helm = ".".join(aa_seq)
    helm = f"PEPTIDE1{{{linear_helm}}}$PEPTIDE1,PEPTIDE1,1:R1-{len(aa_seq)}:R2$$$" if cyclic else f"PEPTIDE1{{{linear_helm}}}$$$$"
    return helm


def unit_tests():

    # helm_seqs = ['PEPTIDE2{[Abu].[Sar].[meL].V.[meL].A.[dA].[meL].[meL].[meV].[Me_Bmt(E)]}$PEPTIDE2,PEPTIDE2,1:R1-11:R2$$$',
    #              'PEPTIDE1{[dL].[dL].L.[dL].P.Y}$PEPTIDE1,PEPTIDE1,1:R1-6:R2$$$',
    #              'PEPTIDE1{[dL].[dL].[dL].[dL].P.Y}$PEPTIDE1,PEPTIDE1,1:R1-6:R2$$$',
    #              'PEPTIDE1{L.L.L.[dL].P.Y}$PEPTIDE1,PEPTIDE1,1:R1-6:R2$$$',
    #              'PEPTIDE1{L.[dL].[dL].[dL].P.Y}$PEPTIDE1,PEPTIDE1,1:R1-6:R2$$$',]
    # sfn = ScoringFunctions()
    # print(sfn.scores(helm_seqs, step=1))

    # sfn = ScoringFunctions(scoring_func_names=['permeability'])
    # print(sfn.scores(helm_seqs, step=1))

    aa_seqs = ["YPEDILDKHLQRVIL", "SGKVSYPEDILDKHLQRVIL","EGEKQYPEDILDKHLQRVIL","SQRPYPEDILDKHLQRVIL","QGSQPYPEDILDKHLQRVIL"]
    # docker = DockScorer()
    # print(docker._call_adcp_dock(aa_seqs))

    # helms = [create_helm_from_aa_seq(aa_seq) for aa_seq in aa_seqs]
    # sfn = ScoringFunctions(scoring_func_names=['beta_catenin'])
    # print(sfn.scores(helms, step=1))

    helms = [create_helm_from_aa_seq(aa_seq, cyclic=True) for aa_seq in aa_seqs]
    sfn = ScoringFunctions(scoring_func_names=['beta_catenin'])
    print(sfn.scores(helms, step=1))


if __name__ == "__main__":
    unit_tests()
