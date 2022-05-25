# -*- coding: utf-8 -*- 
# @Time   : 2021/3/25 5:35 下午 
# @Author : liuyong
# @File   : filter_rules.py 
# @Email  : yong.liu@galixir.com

from collections import defaultdict

import numpy as np
from rdkit import Chem

CARBON_IDX = 6


def resort_reactant_and_score(reactants, scores):
    data = [(score, reactant) for score, reactant in zip(scores, reactants)]
    data = sorted(data, key=lambda x: x[0], reverse=True)
    scores = [each[0] for each in data]
    reactants = [each[1] for each in data]
    return reactants, scores


def get_carbon_num_by_string(smi):
    smi = smi.upper()
    return smi.count('C')


def get_atom_num_map(mol):
    atoms = mol.GetAtoms()
    atom_num_map = defaultdict(int)
    for a in atoms:
        atom_num_map[a.GetAtomicNum()] += 1
    return atom_num_map


def punish_by_num_atoms(target_mol, reactant_mol):
    num_atoms_target = len(target_mol.GetAtoms())
    num_atoms_reactant = len(reactant_mol.GetAtoms())
    if num_atoms_target > 16 and num_atoms_reactant < 8:
        return True
    return False


def punish_by_num_carbon_atoms(target_mol, reactant_mol):
    target_atom_num_map = get_atom_num_map(target_mol)
    reactant_atom_num_map = get_atom_num_map(reactant_mol)
    if target_atom_num_map[CARBON_IDX] > 10 and reactant_atom_num_map[CARBON_IDX] < 3:
        return True
    return False


def manual_rules_for_rxn(target, result):
    mol_target = Chem.MolFromSmiles(target)

    reactants = result['reactants']
    scores = result['scores']
    assert len(reactants) == len(scores)

    for i in range(len(reactants)):
        reactant = reactants[i]
        # 目前只考虑单个反应物的情况
        if "." in reactant or '*' in reactant:
            continue
        mol_reactant = Chem.MolFromSmiles(reactant)

        if punish_by_num_carbon_atoms(mol_target, mol_reactant):
            scores[i] /= np.e**20

    reactants, scores = resort_reactant_and_score(reactants, scores)
    result['reactants'] = reactants
    result['scores'] = scores
    return result


def manual_rules_for_rxn_without_rdkit(target, result):
    reactants = result['reactants']
    scores = result['scores']
    assert len(reactants) == len(scores)

    for i in range(len(reactants)):
        reactant = reactants[i]
        # 目前只考虑单个反应物的情况
        if "." in reactant or '*' in reactant:
            continue
        if get_carbon_num_by_string(target) > 10 and get_carbon_num_by_string(reactant) < 3:
            scores[i] /= np.e**20

    reactants, scores = resort_reactant_and_score(reactants, scores)
    result['reactants'] = reactants
    result['scores'] = scores
    return result
