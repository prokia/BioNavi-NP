#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from rdkit import Chem
import pandas as pd
import numpy as np
def canonicalize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    else:
        return ''

def get_rank(row, base, max_rank):
    for i in range(1, max_rank+1):
        if row['target'] == row['{}{}'.format(base, i)]:
            return i
    return 0

def main(opt):
    with open(opt.targets, 'r') as f:
        targets = [''.join(line.strip().split(' ')) for line in f.readlines()]

    print('okk')
    bio = []
    # with open('reactions.txt', 'r') as f:
        # for i in f:
            # temp = i.strip().split('\t')[0]
            # temp = temp.split('>>')[0]
            # bio.append(temp)
    f = pd.read_table('biogenisis_reaction.txt', header=None)
    print(f)
    bioly = list(f[1])
    for i in bioly:
        if isinstance(i, str):
            s = i.split('>>')[0]
            s = Chem.MolToSmiles(Chem.MolFromSmarts(s))
            bio.append(s)
    # bio = [i.split('>>')[0] for i in bioly]
    # print(bioly[0])
    # print(type(bioly[0]))
    # bio.pop(0)
    print(len(bio))
    bio = set(bio)
    print(len(bio))
    ori_targets = targets[:]
    predictions = [[] for i in range(opt.beam_size)]

    targets = []
    ori_pred = []
    cntt = 0
    with open(opt.predictions, 'r') as f:
        lines = f.readlines()
        for i,v in enumerate(lines):
            if i % 10 == 0:
                it = i // 10
                # print(ori_targets[it])
                if ori_targets[it] in bio:
                # if True:
                    cntt += 1
                    ori_pred.extend(lines[i:i+10])
                    targets.append(ori_targets[it])
    print(cntt, 'cntt')
    print(len(targets))
    print(len(ori_pred))
    test_df = pd.DataFrame(targets)
    test_df.columns = ['target']
    total = len(test_df)


    for i, line in enumerate(ori_pred):
        predictions[i % opt.beam_size].append(''.join(line.strip().split(' ')))
            
    
    for i, preds in enumerate(predictions):
        test_df['prediction_{}'.format(i + 1)] = preds
        test_df['canonical_prediction_{}'.format(i + 1)] = test_df['prediction_{}'.format(i + 1)].apply(
            lambda x: canonicalize_smiles(x))

    test_df['rank'] = test_df.apply(lambda row: get_rank(row, 'canonical_prediction_', opt.beam_size), axis=1)
    correct = 0
    invalid_smiles = 0
    for i in range(1, opt.beam_size+1):
        correct += (test_df['rank'] == i).sum()
        invalid_smiles += (test_df['canonical_prediction_{}'.format(i)] == '').sum()
        if opt.invalid_smiles:
            print('Top-{}: {:.1f}% || Invalid SMILES {:.2f}%'.format(i, correct/total*100,
                                                                     invalid_smiles/(total*i)*100))
        else:
            print('Top-{}: {:.1f}%'.format(i, correct / total * 100))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='score_predictions.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-beam_size', type=int, default=10,
                       help='Beam size')
    parser.add_argument('-invalid_smiles', action="store_true",
                       help='Show %% of invalid SMILES')
    parser.add_argument('-predictions', type=str, required=True,
                       help="Path to file containing the predictions")
    parser.add_argument('-targets', type=str, required=True,
                       help="Path to file containing targets")

    opt = parser.parse_args()
    main(opt)
