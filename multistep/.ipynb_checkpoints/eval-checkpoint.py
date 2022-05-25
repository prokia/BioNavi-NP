import ast
import re
import pandas as pd
from rdkit import Chem


def get_dict_from_log(pth):  # 从log中获取dict
    '''

    :param pth: the path of running log
    :return: a dict, format: {
                    smiles1: {0: top 1 reaction path; 1: top 2 reaction path; ...}
                    smiles2: {0: top 1 reaction path; 1: top 2 reaction path; ...}
                }
    '''
    mol_dict = {}
    with open(pth, 'r') as f:
        l = [i.strip('\n') for i in f.readlines()]

    pos = 0
    length = len(l)
    while pos < length:
        if l[pos] == 'cuda:1':
            smiles = l[pos - 1]
            pos += 1
            while l[pos][0] != '{' and l[pos] != 'None':
                pos += 1
            if l[pos] == 'None':
                mol_dict[smiles] = None
            else:
                smiles_dict = ast.literal_eval(l[pos])
                mol_dict[smiles] = smiles_dict
        pos += 1

    return mol_dict


def get_ground_truth_dict(pth):
    '''

    :param pth: the path of ground truth file(.txt)
    :return: a dict, format: {
                    smiles1: {0: ground truth path1; 1: ground truth path2; ...}
                    smiles2: {0: ground truth path1; 1: ground truth path2; ...}
                }
    '''
    ground_truth_dict = {}
    with open(pth, 'r', encoding='gbk') as f:
        l = [i.strip('\n') for i in f.readlines()]

    l.pop(0)
    for line in l:
        ele = line.split('\t')
        while ele[-1] == '':
            ele.pop(-1)
        path_len = int(ele[0])
        if path_len == 1 or path_len > 10:
            continue

        target = ele[1]
        if target in ground_truth_dict:
            pos = len(ground_truth_dict[target])
            ground_truth_dict[target].update({pos: '|'.join(ele[1:])})
        else:
            ground_truth_dict[target] = {0: '|'.join(ele[1:])}

    return ground_truth_dict


def compare(ori_path, pred_path):  # 得到两条合成路径的相同部分长度
    '''

    :param ori_path: ground truth path
    :param pred_path: pred path
    :return: num: 1 if ori_path == pred_path else 0
            intersect: the number of the same reactions
    '''
    ori_set = set(ori_path)
    pred_set = set(pred_path)
    intersect = len(ori_set & pred_set)
    num = 0

    if len(ori_path) == len(pred_path) and intersect == len(ori_path):
        # print(ori_path)
        # print(pred_path)
        num = 1
    return num, intersect - 1


def get_lst_from_pred_dict(s):  # 将字符串转换为list存储的路径
    '''

    :param s: 'smiles1>score1>smiles2|smiles2>score2>smiles3|smiles3>score3>smiles4...'
    :return: [smiles1, smiles2, smiles3, ...]
    '''
    lst = []
    rea = s.split('|')
    for i in range(len(rea)):
        line = rea[i].split('>')
        if i == 0:
            lst.append(line[0])
        lst.append(line[-1])
    return [Chem.MolToSmiles(Chem.MolFromSmarts(mol)) for mol in lst]


def get_lst_from_gt_dict(s):
    '''

    :param s: 'smiles1|smiles2|smiles3...'
    :return: [smiles1, smiles2, smiles3, ...]
    '''
    l = s.split('|')
    return [Chem.MolToSmiles(Chem.MolFromSmarts(mol)) for mol in l]


def to_csv(result_dict):
    df = pd.DataFrame(result_dict)
    df2 = pd.DataFrame(df.values.T, index=df.columns, columns=df.index)
    df2.to_csv('./result.csv')


if __name__ == '__main__':
    building_block_num = 0

    mol_dict = get_dict_from_log('testset_in.log')
    ground_truth_dict = get_ground_truth_dict('testset_in.txt')
    # print(mol_dict)

    print(len(mol_dict))
    print(len(ground_truth_dict))
    print(ground_truth_dict['CC(=O)Nc1ccc(O)c(OS(=O)(=O)O)c1'])
    print(mol_dict['CC(=O)Nc1ccc(O)c(OS(=O)(=O)O)c1'])

    none_num = 0  # 没预测出路径的分子个数
    result_dict = {}
    all_correct = 0  # 全对个数
    max_max_intersect = 0  # 最大相同个数

    for smiles, v_dict in mol_dict.items():
        if v_dict is None:
            none_num += 1
            continue

        building_block_same = False
        gt_dict = ground_truth_dict[smiles]

        max_intersect = 0
        max_pred = ''
        max_gt = ''
        for routes_score in v_dict.values():
            for item in gt_dict.values():
                gt_lst = get_lst_from_gt_dict(item)
                pred_lst = get_lst_from_pred_dict(routes_score['routes'])
                if gt_lst[-1] == pred_lst[-1]:
                    building_block_same = True
                num, intersect = compare(gt_lst, pred_lst)
                all_correct += num
                if intersect > max_intersect:
                    max_intersect = intersect
                    max_pred = pred_lst
                    max_gt = gt_lst
        max_max_intersect = max(max_max_intersect, max_intersect)
        if building_block_same:
            building_block_num += 1
        result_dict[smiles] = {'max_intersect': max_intersect, 'max_pred': '\t'.join(max_pred),
                               'max_gt': '\t'.join(max_gt)}
    print(max_max_intersect)
    print(all_correct)
    print(none_num)
    print(building_block_num)
    to_csv(result_dict)

    # for k, v in result_dict.items():
    #     print(k, v, sep='\n')
