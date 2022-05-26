import argparse
from rdkit import Chem
import pandas as pd
import numpy as np
from tqdm import tqdm
from rdkit import RDLogger


lg = RDLogger.logger()
lg.setLevel(4)


def save_txt(data, path):
    with open(path, 'w') as f:
        for each in data:
            f.write(each + '\n')


def read_txt(path):
    data = []
    with open(path, 'r') as f:
        for each in f.readlines():
            data.append(each.strip('\n'))
        return data


def remove_chiral(smi):
    if '>>' not in smi:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            #print('Invalid smiles: ', smi)
            return smi
        ret = Chem.MolToSmiles(mol, isomericSmiles=False)
    else:
        reactants, product = smi.split('>>')[0], smi.split('>>')[-1]
        mol_react, mol_prod = Chem.MolFromSmiles(reactants), Chem.MolFromSmiles(product)
        ret_react, ret_prod = Chem.MolToSmiles(mol_react, isomericSmiles=False), Chem.MolToSmiles(mol_prod, isomericSmiles=False)
        ret = '>>'.join([ret_react, ret_prod])
    return ret


def cal_acc(preds, target, n_best):
    # 默认输出top1 3 5 10的acc
    assert len(target) * n_best == len(preds)
    correct_cnt = {'top1': 0, 'top3': 0, 'top5': 0, 'top10': 0}
    for i, tgt_smi in enumerate(target):
        pred_list = preds[i*n_best: i*n_best + n_best]
        if tgt_smi in pred_list[:1]:
            correct_cnt['top1'] += 1
        if tgt_smi in pred_list[:3]:
            correct_cnt['top3'] += 1
        if tgt_smi in pred_list[:5]:
            correct_cnt['top5'] += 1
        if tgt_smi in pred_list[:10]:
            correct_cnt['top10'] += 1
    acc_dict = {key: value / len(target) for key, value in correct_cnt.items()}
    return acc_dict


tgt_path = '/gxr/liuyong/Projects/retro_star/singlestep/dataset/20210302/test_data-tgt.txt'
target = read_txt(tgt_path)
target = [remove_chiral(each.replace(' ', '')) for each in target]

#pred_path = '/gxr/liuyong/Projects/retro_star/singlestep/prediction/extend_chiral/pred.txt'
#pred_path = '/gxr/liuyong/Projects/retro_star/singlestep/prediction/extend_no_chiral/pred.txt'
#pred_path = '/gxr/liuyong/Projects/retro_star/singlestep/prediction/chiral/pred.txt'
#pred_path = '/gxr/liuyong/Projects/retro_star/singlestep/prediction/extend_chiral/pred.txt'
#pred_path = '/gxr/liuyong/Projects/retro_star/singlestep/prediction/np-like/pred_ensemble.txt'
#pred_path = '/gxr/liuyong/Projects/retro_star/singlestep/prediction/25w_steps/np-like/preds.txt'

def evaluate_all():
    train_set = ['extend_chiral', 'extend_no_chiral', 'chiral', 'no_chiral', 'np-like']
    
    print("------------------------------")
    for each in train_set:
        pred_path = f'/gxr/liuyong/Projects/retro_star/singlestep/prediction/3w_steps/{each}/preds.txt'
        
        preds = read_txt(pred_path)
        preds = [remove_chiral(each.replace(' ', '')) for each in preds]
        #preds = [each.replace(' ', '') for each in tqdm(preds)]
        
        #print("------------------------------")
        acc_dict = cal_acc(preds, target, 10)
        print(each, ' = ', acc_dict)


def evaluate_ensemble():
    pred_path = '/gxr/liuyong/Projects/retro_star/singlestep/prediction/np-like/preds_ensemble.txt'
    preds = read_txt(pred_path)
    preds = [remove_chiral(each.replace(' ', '')) for each in preds]
    #preds = [each.replace(' ', '') for each in tqdm(preds)]
    
    #print("------------------------------")
    acc_dict = cal_acc(preds, target, 10)
    print('np-like-ensemble = ', acc_dict)


if __name__=='__main__':
    evaluate_ensemble()
