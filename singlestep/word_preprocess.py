import numpy as np
from rdkit import Chem
from tqdm import tqdm
import pandas as pd

def from_file(pth):
    df = pd.read_table(pth, header=None)
    # print(df)
    mol = list(df[1])
    # print(mol)
    return mol

def get_np(pth):
    df = pd.read_table(pth)
    bio = list(df['biocatalysis'])
    nplike = list(df['NP_Like'])
    print(len(bio), len(nplike))
    # for i in np:
    #     if i == 'NP_Like':
    #         print('>????????>')
    return nplike


def smi_tokenizer(smi):
    """
    Tokenize a SMILES molecule or reaction
    """
    import re
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(tokens)


def to_file(reactions, type, pth):
    assert type in ['train', 'val']

    reactants = []
    products = []
    for reaction in tqdm(reactions):
        reactant, product = reaction.split('>>')
        # print(reactant)
        try:
            reactant = Chem.MolToSmiles(Chem.MolFromSmarts(reactant))
            product = Chem.MolToSmiles(Chem.MolFromSmarts(product))
        except:
            print('wrong:', reactant)
            continue
        reactant = smi_tokenizer(reactant)
        reactants.append(reactant)
        product = smi_tokenizer(product)
        products.append(product)
    print('reactants', len(reactants))
    f = open(pth + 'new-src-' + type + '.txt', 'w')
    fp = open(pth + 'new-tgt-' + type + '.txt', 'w')
    for i, v in enumerate(reactants):
        if products[i] != '' and reactants[i] != '':
            f.write(products[i] + '\n')
            fp.write(reactants[i] + '\n')


if __name__ == '__main__':
    molecule = from_file('./biogenisis_reaction.txt')
    nplike = get_np('./reactions.txt')
    # split_exp = 0.8
    # print(len(molecule+np))
    # all_mol = molecule + np
    np.random.shuffle(molecule)
    np.random.shuffle(nplike)

    train_mol = molecule[:int(len(molecule)*0.9)] + nplike[:int(len(nplike)*0.9)]
    val_mol = molecule[int(len(molecule)*0.9):] + nplike[int(len(nplike)*0.9):]
    to_file(train_mol, 'train', './mol_trans/all_train/')
    to_file(val_mol, 'val', './mol_trans/all_train/')

    train_mol = molecule[:int(len(molecule)*0.9)]
    val_mol = molecule[int(len(molecule)*0.9):]
    to_file(train_mol, 'train', './mol_trans/bio_train/')
    to_file(val_mol, 'val', './mol_trans/bio_train/')

    train_mol = nplike[:int(len(nplike)*0.9)]
    val_mol = nplike[int(len(nplike)*0.9):]
    to_file(train_mol, 'train', './mol_trans/nplike_train/')
    to_file(val_mol, 'val', './mol_trans/nplike_train/')

