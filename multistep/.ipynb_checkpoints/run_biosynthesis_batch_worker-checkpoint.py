import os

from tqdm import tqdm
from rdkit import Chem
import random, time, queue
from multiprocessing.managers import BaseManager
from multiprocessing import Pool

from config import Config
from retro_star.api import RSPlanner

task_queue = queue.Queue()
result_queue = queue.Queue()


class QueueManager(BaseManager):
    pass


QueueManager.register('get_task_queue', callable=lambda: task_queue)
QueueManager.register('get_result_queue', callable=lambda: result_queue)

manager = QueueManager(address=('10.10.10.7', 5000), authkey=b'abc123')
manager.start()
task_q = manager.get_task_queue()
result_q = manager.get_result_queue()


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


def run(conf):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3, 4, 5, 6, 7'

    # canonicalization
    mol = Chem.MolToSmiles(Chem.MolFromSmarts(conf.target_mol))

    planner = RSPlanner(
        gpu=conf.gpu,
        use_value_fn=conf.use_value_fn,
        value_fn_model_path=conf.value_fn_model_path,
        fp_dim=conf.fp_dim,
        iterations=conf.iterations,
        expansion_topk=conf.expansion_topk,
        route_topk=conf.route_topk,
        one_step_model_type=conf.one_step_model_type,
        buliding_block_path=conf.buliding_block_path,
        mlp_templates_path=conf.mlp_templates_path,
        one_step_model_path=conf.one_step_model_path,
        beam_size=conf.beam_size,
        viz=conf.viz,
        viz_dir=conf.viz_dir
    )

    result = planner.plan(mol)
    result_routes_list = []

    if result is None:
        return None

    for i, route in enumerate(result):
        route_dict = {
            'route_id': i,
            'route': route[0],
            'route_score': route[1]
        }
        result_routes_list.append(route_dict)
    return result_routes_list


if __name__ == '__main__':
    pool = Pool(20)
    conf_list = []
    while True:
        conf = task_q.get(timeout=1000)

        if conf is None:
            break

        conf_list.append(conf)

        if len(conf_list) == 20:
            for res in pool.imap(run, conf_list):
                result_q.put(res)
            conf_list = []

    if len(conf_list) > 0:
        for res in pool.imap(run, conf_list):
            result_q.put(res)


