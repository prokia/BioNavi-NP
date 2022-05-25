import os
import sys
import time

import pynvml
from rdkit import Chem

from retro_star.api import RSPlanner


def get_avai_gpu():
    pynvml.nvmlInit()
    for gpu_id in range(8):
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        if meminfo.free / meminfo.total > 0.2:
            return gpu_id
    return -1


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


def run(input_dict):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'

    # canonicalization
    mol = Chem.MolToSmiles(Chem.MolFromSmarts(input_dict['target_mol']))
    one_step_model_path = [
        '../singlestep/checkpoints/np-like/model_step_30000.pt',
        '../singlestep/checkpoints/np-like/model_step_50000.pt',
        '../singlestep/checkpoints/np-like/model_step_80000.pt',
        '../singlestep/checkpoints/np-like/model_step_100000.pt'
    ]
    value_fn_model_path = './retro_star/saved_models/best_epoch_final_4.pt'
    viz_dir = os.path.join('viz_' + str(int(time.time())))
    ret_file_path = os.path.join('./viz/tmp/', viz_dir + '.zip')
    planner = RSPlanner(
        gpu=get_avai_gpu(),
        use_value_fn=True,
        value_fn_model_path=value_fn_model_path,
        fp_dim=2048,
        iterations=input_dict['expansion_iters'],
        expansion_topk=input_dict['expansion_topk'],
        route_topk=input_dict['route_topk'],
        one_step_model_type='onmt',
        buliding_block_path=input_dict['building_blocks'],
        mlp_templates_path=None,
        one_step_model_path=one_step_model_path,
        beam_size=20,
        viz=True,
        viz_dir=viz_dir
    )

    result = planner.plan(mol)

    return result


def get_input():
    input_data = {
        'target_mol': 'N[C@@H](CNC(=O)C(=O)O)C(=O)O',
        'expansion_topk': 50,
        'max_depth': 10,
        'expansion_iters': 10,
        'route_topk': 5,
    }
    print('Input: ')
    target_mol = sys.stdin.readline().strip('\n')
    input_data['target_mol'] = target_mol
    return input_data


def main_biosynthesis():
    input_dict = {
        'target_mol': 'N[C@@H](CNC(=O)C(=O)O)C(=O)O',
        'expansion_topk': 50,
        'max_depth': 10,
        'expansion_iters': 10,
        'route_topk': 5,
        #'building_blocks': ['N[C@@H](CO)C(=O)O'] #one can assign a specific building block with this command
        'building_blocks':'retro_star/dataset/bio_data/bio_building_blocks_all/building_block.csv'
    }
    

    res = run(input_dict)
    print(res)


if __name__ == '__main__':
    main_biosynthesis() 
