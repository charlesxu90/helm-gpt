import warnings
import numpy as np
from rdkit import Chem, rdBase, DataStructs
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from typing import List
from loguru import logger

from utils.metrics_utils import mapper

rdBase.DisableLog('rdApp.error')
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def fingerprints_from_mol(molecule, radius=3, size=2048, hashed=False):
    """ Create ECFP fingerprint of a molecule """
    if hashed:
        fp_bits = AllChem.GetHashedMorganFingerprint(molecule, radius, nBits=size)
    else:
        fp_bits = AllChem.GetMorganFingerprintAsBitVect(molecule, radius, nBits=size)
    fp_np = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp_bits, fp_np)
    return fp_np.reshape(1, -1)

def fingerprints_from_smi(smi: str, size=2048):
    """ Create ECFP fingerprints of a smiles, with validity check """
    mol = Chem.MolFromSmiles(smi)
    is_valid = int(mol is not None)
    fp = fingerprints_from_mol(mol, size=size) if mol else np.zeros((1, size))
    return fp, is_valid

def fingerprints_from_smiles(smiles: List, size=2048, n_jobs=1):
    """ Create ECFP fingerprints of smiles, with validity check """
    results = mapper(n_jobs)(fingerprints_from_smi, smiles)

    fps = np.vstack([x[0] for x in results])
    valid_mask = np.array([x[1] for x in results])
    return fps, valid_mask

def is_smi_valid(smi: str):
    mol = Chem.MolFromSmiles(smi) if smi else None
    return int(mol is not None)

def check_smi_validity(smiles: list, n_jobs=1):
    valid_mask = mapper(n_jobs)(is_smi_valid, smiles)
    valid_idx = np.where(valid_mask)[0]
    valid_smi = [smiles[i] for i in valid_idx]
    return valid_smi, valid_idx

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

def get_dps_from_smi(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
    except:
        print(f"convert smi {smi} to molecule failed!")
        mol = None
    
    dps, _ = getMolDescriptors(mol)
    return np.array(dps)


def get_dps(smi_list, n_jobs=1):
    if len(smi_list) == 0:
        return np.zeros((0, 211))
    
    dps = mapper(n_jobs)(get_dps_from_smi, smi_list)
    return np.array(dps)
