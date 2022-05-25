import os

from tqdm import tqdm
from rdkit import Chem
import random, time, queue
from multiprocessing.managers import BaseManager

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


def read_txt(path):
    data = []
    with open(path, 'r') as f:
        for each in f.readlines():
            data.append(each.strip('\n'))
        return data


if __name__ == '__main__':
    conf = Config('config/example_conf.yaml')
    data_path = 'retro_star/dataset/bio_data/test_20210324/test.txt'
    data = read_txt(data_path)
    products = [each.split('\t')[0] for each in data][1:]
    building_blocks = [each.split('\t')[1] for each in data][1:]

    conf_list = []

    for i, product in enumerate(tqdm(products)):
        conf.target_mol = product
        
        # single building block的时候把building_block_path设置成只有一个分子的list就可以
        #building_block_path = [building_blocks[i]]
        
        # 如果是全部的building blocks就可以把整个building的文件路径传进去
        building_block_path = 'retro_star/dataset/bio_data/bio_building_blocks_all/building_block.csv'

        conf.buliding_block_path = building_block_path
        viz_dir = f"viz/20210324/mol_{i}"
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)
        conf.viz_dir = viz_dir

        conf_list.append(conf)

        task_q.put(conf)

    for _ in range(len(conf_list)):
        result = result_q.get(timeout=1000)
        if result is not None:
            for route in result:
                print(f"route id: {route['route_id']}\n"
                      f"route score:  {route['route_score']}\n"
                      f"route: {route['route']}\n")
