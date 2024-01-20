import warnings
import numpy as np
from loguru import logger
from sklearn.ensemble import RandomForestRegressor
from rdkit.Chem import Descriptors, rdMolDescriptors
import joblib
from agent.scoring.transformation import TransformFunction
from utils.helm_utils import get_cycpep_smi_from_helm
from rdkit import Chem, rdBase, DataStructs
from rdkit.Chem import AllChem
from typing import List

rdBase.DisableLog('rdApp.error')
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def fingerprints_from_mol(molecule, radius=3, size=2048, hashed=False):
    """
        Create ECFP fingerprint of a molecule
    """
    if hashed:
        fp_bits = AllChem.GetHashedMorganFingerprint(molecule, radius, nBits=size)
    else:
        fp_bits = AllChem.GetMorganFingerprintAsBitVect(molecule, radius, nBits=size)
    fp_np = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp_bits, fp_np)
    return fp_np.reshape(1, -1)


def fingerprints_from_smiles(smiles: List, size=2048):
    """ Create ECFP fingerprints of smiles, with validity check """
    fps = []
    valid_mask = []
    for i, smile in enumerate(smiles):
        mol = Chem.MolFromSmiles(smile)
        valid_mask.append(int(mol is not None))
        fp = fingerprints_from_mol(mol, size=size) if mol else np.zeros((1, size))
        fps.append(fp)
    
    fps = np.concatenate(fps, axis=0) if len(fps) > 0 else np.zeros((0, size))
    return fps, valid_mask


def getMolDescriptors(mol, missingVal=0):
    """ calculate the full list of descriptors for a molecule """

    values, names = [], []
    for nm, fn in Descriptors._descList:
        try:
            val = fn(mol)
        except:
            val = missingVal
        values.append(val)
        names.append(nm)

    custom_descriptors = {'hydrogen-bond donors': rdMolDescriptors.CalcNumLipinskiHBD,
                          'hydrogen-bond acceptors': rdMolDescriptors.CalcNumLipinskiHBA,
                          'rotatable bonds': rdMolDescriptors.CalcNumRotatableBonds,}
    
    for nm, fn in custom_descriptors.items():
        try:
            val = fn(mol)
        except:
            val = missingVal
        values.append(val)
        names.append(nm)
    return values, names


def get_pep_dps_from_smi(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
    except:
        print(f"convert smi {smi} to molecule failed!")
        mol = None
    
    dps, _ = getMolDescriptors(mol)
    return np.array(dps)


def get_pep_dps(smi_list):
    if len(smi_list) == 0:
        return np.zeros((0, 211))
    return np.array([get_pep_dps_from_smi(smi) for smi in smi_list])


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

class Permeability:
    """
        Predict permeability of peptides using helms as inputs
    """
    def __init__(self, model_path='data/cpp/regression_rf.pkl', batch_size=1000, input_type='helm'):
        self.predictor = self.load_predictor(model_path)
        self.trans_fn = TransformFunction('sigmoid', -8, -4, params={'k': 1.})  # high: > -6, low < -6
        self.batch_size = batch_size
        self.input_type = input_type if input_type in ['helm', 'smiles'] else 'helm'

    @staticmethod
    def load_predictor(model_path):
        # logger.debug(f'Predictor {predictor_name} -----')
        predictor = joblib.load(model_path)
        return predictor

    def get_features(self, input_seqs: list):
        if self.input_type == 'helm':
            valid_smiles, valid_idxes = get_smi_from_helms(input_seqs)
        else:
            valid_smiles, valid_idxes = check_smi_validity(input_seqs)

        X_fps = fingerprints_from_smiles(valid_smiles)[0]
        X_dps = get_pep_dps(valid_smiles)
        # logger.debug(f'X_fps.shape: {X_fps.shape}, X_dps.shape: {X_dps.shape}')

        if len(X_fps) == 0:
            valid_features = np.zeros((0, X_fps.shape[1] + X_dps.shape[1]))
        else:
            valid_features = np.concatenate([X_fps, X_dps], axis=1)
            
        return valid_features, valid_idxes

    def __call__(self, input_seqs: list):
        scores = self.get_scores(input_seqs)
        scores_tfd = self.trans_fn(scores)
        return scores_tfd, scores
    
    def get_scores(self, input_seqs: list):
        scores = -10 * np.ones(len(input_seqs))
        valid_features, valid_idxes = self.get_features(input_seqs)
        if len(valid_features) == 0:
            return scores
        
        valid_features = np.nan_to_num(valid_features, nan=0.)
        valid_features = np.clip(valid_features, np.finfo(np.float32).min, np.finfo(np.float32).max)
        scores[valid_idxes] = self.predictor.predict(valid_features)
        return scores


def unittest():
    permeability = Permeability()
    helm_seqs = ['PEPTIDE2{[Abu].[Sar].[meL].V.[meL].A.[dA].[meL].[meL].[meV].[Me_Bmt(E)]}$PEPTIDE2,PEPTIDE2,1:R1-11:R2$$$',
                 'PEPTIDE1{[dL].[dL].L.[dL].P.Y}$PEPTIDE1,PEPTIDE1,1:R1-6:R2$$$',
                 'PEPTIDE1{[dL].[dL].[dL].[dL].P.Y}$PEPTIDE1,PEPTIDE1,1:R1-6:R2$$$',
                 'PEPTIDE1{L.L.L.[dL].P.Y}$PEPTIDE1,PEPTIDE1,1:R1-6:R2$$$',
                 'PEPTIDE1{L.[dL].[dL].[dL].P.Y}$PEPTIDE1,PEPTIDE1,1:R1-6:R2$$$',
                 'PEPTIDE1{H.G.D.G.S.F.S.D.E.M.N.T.I.L.D.N.L.A.A.R.D.F.I.N.W.[am]}$$$$',
                 'PEPTIDE1{[ac].P.V.L.D.E.F.R.E.K.L.N.E.E.L.E.A.[X221].K.Q.K.L.K.[am]}$$$$', 
                 'PEPTIDE1{F.I.[X816]}$$$$', 
                 'PEPTIDE1{R.R.P.P.Q.[X697].[am]}$$$$']
    print(permeability(helm_seqs=helm_seqs))

    """
    python -m agent.scoring.permeability
    (array([0.00841793, 0.00408923, 0.00408923, 0.00408923, 0.00408923]), 
     array([-5.71336985, -6.65973759, -6.65973759, -6.65973759, -6.65973759]))
    """


if __name__ == '__main__':
    unittest()
