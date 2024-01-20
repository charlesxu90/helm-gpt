import warnings
import numpy as np
from rdkit import Chem, rdBase, DataStructs
from rdkit.Chem import AllChem
from typing import List
import joblib
from loguru import logger

from utils.helm_utils import get_cycpep_smi_from_helm
from agent.scoring.transformation import TransformFunction


rdBase.DisableLog('rdApp.error')
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def fingerprints_from_smiles(smiles: List, size=2048):
    """ Create ECFP fingerprints of smiles, with validity check """
    fps = []
    valid_mask = []
    for i, smile in enumerate(smiles):
        mol = Chem.MolFromSmiles(smile)
        valid_mask.append(int(mol is not None))
        fp = fingerprints_from_mol(mol, size=size) if mol else np.zeros((1, size))
        fps.append(fp)

    fps = np.concatenate(fps, axis=0)
    return fps, valid_mask


def fingerprints_from_mol(molecule, radius=3, size=2048, hashed=False):
    """ Create ECFP fingerprint of a molecule """
    if hashed:
        fp_bits = AllChem.GetHashedMorganFingerprint(molecule, radius, nBits=size)
    else:
        fp_bits = AllChem.GetMorganFingerprintAsBitVect(molecule, radius, nBits=size)
    fp_np = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp_bits, fp_np)
    return fp_np.reshape(1, -1)


def get_smi_from_helms(helm_seqs: list):
    valid_idxes = []
    valid_smiles = []

    for idx, helm in enumerate(helm_seqs):
        # Ignore helm which cannot converted into molecules
        try:
            smi = get_cycpep_smi_from_helm(helm)
            if smi:
                valid_idxes.append(idx)
                valid_smiles.append(smi)
        except Exception as e:
            # logger.debug(f'Error: {e} in helm {helm}')
            pass
    return valid_smiles, valid_idxes

def check_smi_validity(smiles: list):
    valid_smi, valid_idx = [], []
    for idx, smi in enumerate(smiles):
        try:
            mol = Chem.MolFromSmiles(smi) if smi else None
            if mol:
                valid_smi.append(smi)
                valid_idx.append(idx)
        except Exception as e:
            # logger.debug(f'Error: {e} in smiles {smi}')
            pass 
    return valid_smi, valid_idx

class KRASIC50:
    """
        Predict permeability of peptides using helms as inputs
    """
    def __init__(self, model_path='data/kras/kras_xgboost_reg.pkl', scaler_path='data/kras/kras_xgboost_reg_scaler.pkl', input_type='helm'):
        self.predictor, self.scaler = self.load_predictor(model_path, scaler_path)
        self.trans_fn = TransformFunction('rsigmoid', 0, 2, params={'k': 1.})  # high: < 0 (1 nM), low > 2 (100 nM)
        self.input_type = input_type if input_type in ['helm', 'smiles'] else 'helm'

    @staticmethod
    def load_predictor(model_path, scaler_path):
        predictor = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return predictor, scaler

    def get_features(self, input_seqs: list):
        if self.input_type == 'helm':
            valid_smiles, valid_idxes = get_smi_from_helms(input_seqs)
        else:
            valid_smiles, valid_idxes = check_smi_validity(input_seqs)

        X_fps = fingerprints_from_smiles(valid_smiles)[0]

        if len(X_fps) == 0:
            valid_features = np.zeros((0, X_fps.shape[1]))
        else:
            valid_features = X_fps
            
        return valid_features, valid_idxes

    def __call__(self, input_seqs: list):
        scores = self.get_scores(input_seqs)
        scores_tfd = self.trans_fn(scores)
        return scores_tfd, scores
    
    def get_scores(self, input_seqs: list):
        scores = 5 * np.ones(len(input_seqs))
        valid_features, valid_idxes = self.get_features(input_seqs)
        valid_features = np.clip(valid_features, np.finfo(np.float32).min, np.finfo(np.float32).max)
        if len(valid_features) > 0:
            try:
                scores[np.array(valid_idxes)] = self.predictor.predict(valid_features)
            except:
                logger.debug(f'Error in features {valid_features} ')
                pass
        return scores
    

def unittest():
    helms = ['PEPTIDE1{[ac].C.P.L.Y.I.S.Y.D.P.V.C.[-NH2]}$PEPTIDE1,PEPTIDE1,2:R3-12:R3$$$', 'PEPTIDE1{[ac].R.R.C.P.L.Y.I.S.Y.D.P.V.C.R.R.[-NH2]}$PEPTIDE1,PEPTIDE1,4:R3-14:R3$$$']
    kras_inhibition = KRASIC50()
    scores_tfd, scores = kras_inhibition(helms)
    print(scores_tfd)
    print(scores)

if __name__ == '__main__':
    unittest()