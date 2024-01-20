from rdkit import Chem
import re
import copy
import pandas as pd
from loguru import logger
from rdkit import RDLogger
from utils.utils import mapper

RDLogger.DisableLog('rdApp.*')

# Relabel R_group: `'*:_R1'` to `'*:1'` or vice verse

def relabel_rgroup2index(smi):
    """
        Input: SMILES of a molecule with R groups, e.g. 'CCCC[C@H](N(C)[*:_R1])C([*:_R2])=O'
        Output: SMILES of a molecule with R groups, e.g. 'CCCC[C@H](N(C)[*:1])C([*:2])=O'
    """
    r_group_name = re.findall(r'\[\*\:(_R\d)\]', smi)
    r_group_name = list(r_group_name)
    # logger.debug(f"r_group_name: {r_group_name}")
    
    checked_rgroup = []
    for name in r_group_name:
        # logger.debug(f"name: {name}, name[2:]: {name[2:]}, checked_rgroup: {checked_rgroup}")
        if name in checked_rgroup:  # For the case of _R3 and _R3, to *:3 and *:4
            # logger.debug(f"name in checked_rgroup: {name}")
            smi = smi.replace(name, f'{int(name[2:])+1}', 1)  # Replace the first occurence
            name = f'_R{int(name[2:])+1}'
        else:  # For the case of _R1 and _R2, to *:1 and *:2
            # logger.debug(f"name not in checked_rgroup: {name}")
            smi = smi.replace(name, f'{name[2:]}', 1)  # Replace the first occurence
            # logger.debug(f"smi: {smi}")
        checked_rgroup.append(name)
        # logger.debug(f"smi: {smi}")
    return smi


def relabel_rgroup2label(smi):
    """
        Input: A SMILES with R groups, e.g. 'CCCC[C@H](N(C)[*:1])C([*:2])=O'
        Output: A SMILES with R groups, e.g. 'CCCC[C@H](N(C)[*:_R1])C([*:_R2])=O'
    """
    r_group_name = re.findall(r'\[(\*\:\d)\]', smi)
    r_group_name = list(r_group_name)
    for name in r_group_name:
        smi = smi.replace(name, f'*:_R{name[2:]}')
    return smi


def get_smi_from_cxsmiles(cxsmiles):
    """
    Get SMILES from CXSMILES
    Input: 'CCCC[C@H](N(C)[*])C([*])=O |$;;;;;;;_R1;;_R2;$|'
    Output: 'CCCC[C@H](N(C)[*:_R1])C([*:_R2])=O'
    """
    smi_list = cxsmiles.split('|')
    smi, pos = smi_list[0], smi_list[1]
    labels = pos.split('$')[1].split(';')

    # Replace * with *:label according to the occurence of * in the SMILES
    for label in labels:
        if len(label) == 0:
            continue
        index = smi.index('[*]')
        smi = smi[:index] + f'[*:{label}]' + smi[index+3:]

    return smi.strip()


def get_cxsmiles_from_smi(smi):
    """
    Get CXSMILES from SMILES
    Input: 'CCCC[C@H](N(C)[*:_R1])C([*:_R2])=O'
    Output: 'CCCC[C@H](N(C)[*])C([*])=O|$;;;;;;;_R1;;_R2;$|'
    """
    cxsmiles = smi
    # Get all labels with pattern [*:label]
    labels = re.findall(r'\[\*\:(.*?)\]', smi)
    r_groups = []
    for label in labels:
        # Replace [*:label] with [*]
        cxsmiles = cxsmiles.replace(f'[*:{label}]', '[*]')
        r_groups.append(f'{label}')     # Record label to r_groups list

    pos = list()
    r_group_idx = 0
    # Iterate through the SMILES and add ';'s in pos
    for i in range(len(cxsmiles)):
        if cxsmiles[i] in ('H', '@', '[', ']', '(', ')', '=', '-', '#', ':', '+',
                           '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '/', '\\',
                           'l', 'r'):  # 'l' for Cl, 'r' for 'Br'
            # print(cxsmiles[i])
            continue
        elif cxsmiles[i] == '*':
            if r_group_idx >= len(r_groups):
                raise Exception(
                    f'Erorr in converting SMILES to CXSMILES, {smi}')
            pos.append(f"{r_groups[r_group_idx]}")
            r_group_idx += 1
        else:
            pos.append('')
    pos = '|$' + ';'.join(pos) + '$|'
    return f'{cxsmiles} {pos}'


def combine_monomer_unused_rgroup(smi, r_group):
    """ 
    Combine two molecules with unused R groups
    smi: SMILES of the monomer, e.g. 'CCCC[C@H](N(C)[*:1])C([*:2])=O'
    r_group: SMILES of the second molecule, e.g. '[*:1][H]', '[*:2][OH]'

    Note: the R group should be in the form of [*:1], [*:2], etc. Not [*:_R1], [*:_R2], etc.
    """
    # [*]
    m1 = Chem.MolFromSmiles(smi)
    m2 = Chem.MolFromSmiles(r_group)

    mol = Chem.molzip(m1, m2)
    return Chem.MolToSmiles(mol)


def replace_unused_r_groups(mol_smi, r_groups: dict, used_r_groups: list):
    # logger.debug(f"before replace unused r group mol_smi: {mol_smi}, r_groups: {r_groups}, used_r_groups: {used_r_groups}")
    mol_smi = relabel_rgroup2index(mol_smi)
    # logger.debug(f"mol_smi: {mol_smi}")
    for r_group in r_groups.keys():
        if r_group not in used_r_groups:
            # e.g. '[H][*:1]'
            r_group_smi = f"[{r_groups[r_group]}][*:{r_group[1:]}]"
            # logger.debug(f"mol_smi: {mol_smi}, r_group_smi: {r_group_smi}")
            mol_smi = combine_monomer_unused_rgroup(mol_smi, r_group_smi)
    # Remove [H] in smiles, as it is useless
    mol_smi = mol_smi.replace('[H]', '')
    # logger.debug(f"after replace unused r group mol_smi: {mol_smi}")
    return relabel_rgroup2label(mol_smi)


def clean_dummy_labels_in_cxsmiles(smi):
    """
    Clean dummy labels in cxsmiles
    Input: '*C(=O)[C@@H]1CCCN1C(C)=O |$_R2;;;;;;;;;;$,atomProp:0.dummyLabel.*|'
    Output: '[*]C(=O)[C@@H]1CCCN1C(C)=O |$_R2;;;;;;;;;;$|'
    """
    smi_parts = smi.split('|')
    # logger.info(smi_parts)
    smi = smi_parts[0].replace('*', '[*]') + '|$' + \
        smi_parts[1].split('$')[1] + '$|'
    return smi


def combine_fragments(smi1, smi2):
    # logger.info(f"Combine {smi1} and {smi2}...")

    m1 = Chem.MolFromSmiles(get_cxsmiles_from_smi(smi1))
    m2 = Chem.MolFromSmiles(get_cxsmiles_from_smi(smi2))

    for atm in m1.GetAtoms():
        if atm.HasProp("atomLabel") and atm.GetProp("atomLabel") == "_R2":
            atm.SetAtomMapNum(10)  # Use 10 as the atom map number
    for atm in m2.GetAtoms():
        if atm.HasProp("atomLabel") and atm.GetProp("atomLabel") == "_R1":
            atm.SetAtomMapNum(10)  # Use 10 as the atom map number
    mol = Chem.molzip(m1, m2)

    smi = Chem.MolToCXSmiles(mol)
    
    if '|' in smi:
        # logger.info(smi)
        smi = get_smi_from_cxsmiles(clean_dummy_labels_in_cxsmiles(smi))
    # logger.info(f"Combined smiles: {smi}")
    return smi


def get_linear_peptide(monomer_smis):
    """Get linear peptide from monomers"""

    for idx, monomer in enumerate(monomer_smis):
        if idx == 0:
            smi = monomer
        else:
            smi = combine_fragments(smi, monomer)
    return smi


def connect_mapped_atoms(smi, end1, end2):
    ''' 
        smi: smiles contain the two merge ends, e.g. '[*:3]O[C@H](C)[C@H](NC(=O)[C@@H](CC(C)C)N(C)C(=O)[C@@H]1CCCN1C(C)=O)C(=O)N(C)[C@@H](CC(C)C)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](CC(C)C)C(=O)N(C)[C@@H](C)C(=O)N[C@H](CC(C)C)C(=O)N1CCC[C@@H]1C([*:2])=O'
        end1: atom map number of the first end, e.g. 3
        end2: atom map number of the second end, e.g. 2

        Note: this will not get atomic stereochemistry right
    '''
    res = Chem.RWMol(Chem.MolFromSmiles(smi))

    dummy1 = None
    dummy2 = None

    for atom in res.GetAtoms():
        if atom.GetAtomMapNum() == end1:
            dummy1 = atom
        elif atom.GetAtomMapNum() == end2:
            dummy2 = atom

    assert dummy1 is not None and dummy2 is not None
    assert dummy1.GetDegree() == 1
    assert dummy2.GetDegree() == 1

    nbr1 = dummy1.GetNeighbors()[0]
    nbr2 = dummy2.GetNeighbors()[0]

    res.BeginBatchEdit()
    res.RemoveAtom(dummy1.GetIdx())
    res.RemoveAtom(dummy2.GetIdx())

    res.AddBond(nbr1.GetIdx(), nbr2.GetIdx(), Chem.BondType.SINGLE)
    res.CommitBatchEdit()
    return Chem.MolToSmiles(res)


def cyclize_linpep_from_smi(smi, link):
    """
    Cyclize a peptide through a link, through definiting a user-defined reaction
    smi: SMILES of a linear peptide, e.g. '[*:_R3]O[C@H](C)[C@H](NC(=O)[C@@H](CC(C)C)N(C)C(=O)[C@@H]1CCCN1C(C)=O)C(=O)N(C)[C@@H](CC(C)C)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](CC(C)C)C(=O)N(C)[C@@H](C)C(=O)N[C@H](CC(C)C)C(=O)N1CCC[C@@H]1C([*:_R2])=O'
    link: the link between the two ends, e.g. '4:R3-10:R2', what if '4:R3-14:R3'
    """
    smi = relabel_rgroup2index(smi) # *:_R3 -> *:3
    link = link.split('-')
    start, end = (int(end.split(':')[1][1:]) for end in link) # 3, 3
    if end == start: end = end + 1 # 3, 3 -> 3, 4
    smi = connect_mapped_atoms(smi, start, end)
    return smi


def get_links_between_monomers(monomer_smis, cyclic_link):
    monomer_links = {}

    def add_link(source_idx, source_r_group, target_idx, target_r_group):
        # Add link to source monomer
        if source_idx not in monomer_links:
            monomer_links[source_idx] = {source_r_group: (target_idx, target_r_group)}
        else:
            monomer_links[source_idx][source_r_group] = (target_idx, target_r_group)

        # Add link to target monomer
        if target_idx not in monomer_links:
            monomer_links[target_idx] = {target_r_group: None}
        else:
            monomer_links[target_idx][target_r_group] = None

    # Add linear links to monomer_links: R2-R1
    for idx in range(len(monomer_smis)-1):
        add_link(idx, '_R2', idx+1, '_R1')

    # Add cyclic link to monomer_links: '4:R3-10:R2'
    def get_idx_rgroup(node):
        idx, r_group = node.split(':')
        return int(idx) - 1, f'_{r_group}'
    
    if cyclic_link:
        source, target = cyclic_link.split('-')
        source_idx, source_r_group = get_idx_rgroup(source)
        target_idx, target_r_group = get_idx_rgroup(target)
        add_link(source_idx, source_r_group, target_idx, target_r_group)

    return monomer_links


def restore_unused_rgroup(monomer_smis, monomer_r_groups, monomer_links):
    # Replace unused R groups with default _R1, _R2, _R3
    for idx, mol_cxsmi in enumerate(monomer_smis):
        # logger.debug(f"mol_cxsmi: {mol_cxsmi}")
        r_groups = monomer_r_groups[idx]
        used_r_groups = [r_group[1:] for r_group in list(monomer_links[idx].keys())]
        # logger.debug(f"used_r_groups: {used_r_groups}, r_groups: {r_groups}")
        mol_smi = replace_unused_r_groups(mol_cxsmi, r_groups, used_r_groups)
        # logger.debug(f"after replace rgroup mol_smi: {mol_smi}")
        monomer_smis[idx] = mol_smi
    return monomer_smis


df_monomers = pd.read_csv('data/prior/monomer_library.csv')

monomers2smi_dict = {}
monomers2r_groups_dict = {}
for index, row in df_monomers.iterrows():
    smi = get_smi_from_cxsmiles(row['CXSMILES'])
    monomers2smi_dict[row['Symbol']] = smi
    monomers2r_groups_dict[row['Symbol']] = {}
    for r_group in ['R1', 'R2', 'R3']:
        if row[r_group] != '-':
            monomers2r_groups_dict[row['Symbol']][r_group] = row[r_group]


def cyclize_linpep_from_helm(linear_helm, cyclic_link):
    # logger.info(f"linear_helm: {linear_helm}")
    monomer_list = linear_helm[1:-1].replace('[', '').replace(']', '').split('.')

    # default monomers with R group denoted by R1, R2, R3
    monomer_smis = [copy.deepcopy(monomers2smi_dict[monomer]) for monomer in monomer_list]
    # Default R groups if no linker
    monomer_r_groups = [copy.deepcopy(monomers2r_groups_dict[monomer]) for monomer in monomer_list]
    # logger.info(f"monomer_smis: {monomer_smis}")

    monomer_links = get_links_between_monomers(monomer_smis, cyclic_link)
    # logger.info(f"monomer_links: {monomer_links}")
    monomer_smis = restore_unused_rgroup(monomer_smis, monomer_r_groups, monomer_links)
    # logger.info(f"monomer_smis: {monomer_smis}")

    # Merge the monomers one by one into a linear peptide
    pep_smi = get_linear_peptide(monomer_smis)
    # logger.info(f"pep_smi: {pep_smi}")
    
    if cyclic_link: # Cyclize the linear peptide
        pep_smi = cyclize_linpep_from_smi(pep_smi, cyclic_link)
        # logger.debug(f"cyclized pep_smi: {pep_smi}")

    return pep_smi

def get_cycpep_smi_from_helm(helm):
    helms = helm.split('$') if '$' in helm else None
    # logger.info(helms)
    if helms is None or len(helms) != 5:
        return None
    # logger.info(helms)
    try:
        pep_idx = helms[0].index('{')
    except:
        return None
    linear_helm = helms[0][pep_idx:]
    # logger.info(linear_helm)

    linker = helms[1].split(',') if ',' in helms[1] else None
    cyclic_linker = linker[2] if linker is not None and len(linker) == 3 else None
    # logger.info(cyclic_linker)

    try:
        smi = cyclize_linpep_from_helm(linear_helm, cyclic_linker)
        return smi
    except Exception as e:
        return None


def is_helm_valid(helm):
    try:
        smi = get_cycpep_smi_from_helm(helm)
        # logger.debug(f"smi {smi}")
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return False
        return True
    except Exception as e:
        return False


def get_smi_from_helms(helm_seqs: list, n_jobs=1):
    smiles = mapper(n_jobs)(get_cycpep_smi_from_helm, helm_seqs)
    valid_idxes = [i for i, x in enumerate(smiles) if x is not None]
    valid_smiles = [x for x in smiles if x is not None]
    return valid_smiles, valid_idxes


def get_validity(helms):
    num_valid = sum([is_helm_valid(x) for x in helms])
    return num_valid/ len(helms)

def get_uniqueness(helms):
    valid_helms = [x for x in helms if is_helm_valid(x)]
    num_unique = len(set(valid_helms))
    return num_unique / len(valid_helms)


def test():

    # Relabel R groups
    print(relabel_rgroup2index('CCCC[C@H](N(C)[*:_R1])C([*:_R2])=O'))
    print(relabel_rgroup2label('CCCC[C@H](N(C)[*:1])C([*:2])=O'))

    # Get molecules from CXSMILES
    smi = get_smi_from_cxsmiles('CCCC[C@H](N(C)[*])C([*])=O |$;;;;;;;_R1;;_R2;$|')
    print(relabel_rgroup2index(smi))

    # Get CXSMILES from SMILES
    smi = get_cxsmiles_from_smi('[*:_R1]N1CCC[C@H]1C([*:_R2])=O')
    print(smi)
    print(combine_monomer_unused_rgroup('CCCC[C@H](N(C)[*:1])C([*:2])=O', '[*:1][H]'))
    replace_unused_r_groups('CCCC[C@H](N(C)[*:_R1])C([*:_R2])=O', monomers2r_groups_dict['Me_dL'], ['R1'])

    clean_dummy_labels_in_cxsmiles('*C(=O)[C@@H]1CCCN1C(C)=O |$_R2;;;;;;;;;;$,atomProp:0.dummyLabel.*|')
    monomer_smis = ['CC[C@H](N[*:_R1])C([*:_R2])=O', 'CN([*:_R1])CC([*:_R2])=O', 'CC(C)C[C@H](N(C)[*:_R1])C([*:_R2])=O', 
                    'CC(C)[C@H](N[*:_R1])C([*:_R2])=O', 'CC(C)C[C@H](N(C)[*:_R1])C([*:_R2])=O', 'C[C@H](N[*:_R1])C([*:_R2])=O', 
                    'C[C@@H](N[*:_R1])C([*:_R2])=O', 'CC(C)C[C@H](N(C)[*:_R1])C([*:_R2])=O', 'CC(C)C[C@H](N(C)[*:_R1])C([*:_R2])=O', 
                    'CC(C)[C@H](N(C)[*:_R1])C([*:_R2])=O', 'C/C=C/C[C@@H](C)[C@@H](O[*:_R3])[C@@H](C([*:_R2])=O)N(C)[*:_R1]']

    get_linear_peptide(monomer_smis)

    connect_mapped_atoms('[*:3]O[C@H](C)[C@H](NC(=O)[C@@H](CC(C)C)N(C)C(=O)[C@@H]1CCCN1C(C)=O)C(=O)N(C)[C@@H](CC(C)C)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](CC(C)C)C(=O)N(C)[C@@H](C)C(=O)N[C@H](CC(C)C)C(=O)N1CCC[C@@H]1C([*:2])=O', 3, 2)

    smi = '[*:_R3]O[C@H](C)[C@H](NC(=O)[C@@H](CC(C)C)N(C)C(=O)[C@@H]1CCCN1C(C)=O)C(=O)N(C)[C@@H](CC(C)C)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](CC(C)C)C(=O)N(C)[C@@H](C)C(=O)N[C@H](CC(C)C)C(=O)N1CCC[C@@H]1C([*:_R2])=O'
    link = '4:R3-10:R2'
    smi = cyclize_linpep_from_smi(smi, link)
    print(smi)

    smi = cyclize_linpep_from_helm('{[Abu].[Sar].[meL].V.[meL].A.[dA].[meL].[meL].[meV].[Me_Bmt(E)]}', '1:R1-11:R2', monomers2smi_dict, monomers2r_groups_dict) 
    print(smi)

    helm = 'PEPTIDE1{[dV].[meA].[meF].[meL].H.[Sar].V.[meL].[meL].D.[meF].A.[-pip]}$PEPTIDE1,PEPTIDE1,1:R1-10:R3$$$'

    smi = get_cycpep_smi_from_helm(helm, monomers2smi_dict, monomers2r_groups_dict)
    print(smi)


if __name__ == '__main__':
    test()
