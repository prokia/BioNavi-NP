import os

from tqdm import tqdm
from rdkit import Chem

from config import Config
from retro_star.api import RSPlanner


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
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'

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
    conf = Config('config/example_conf.yaml')
    data_path = 'retro_star/dataset/bio_data/test_20210324/test.txt'
    data = read_txt(data_path)
    products = [each.split('\t')[0] for each in data][1:]
    building_blocks = [each.split('\t')[1] for each in data][1:]

    for i, product in enumerate(tqdm(products)):
        conf.target_mol = product

        # For single building block, set building_block_path to a list of only one molecule
        # building_block_path = [building_blocks[i]]

        # If it's all building blocks setting, you can input the entire building file path
        building_block_path = 'retro_star/dataset/bio_data/bio_building_blocks_all/building_block.csv'

        conf.buliding_block_path = building_block_path
        viz_dir = f"viz/mol_{i}"
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)
        conf.viz_dir = viz_dir

        try:
            result = run(conf)
        except Exception as e:
            result = None
            print(e)

        if result is not None:
            for route in result:
                print(f"route id: {route['route_id']}\n"
                      f"route score:  {route['route_score']}\n"
                      f"route: {route['route']}\n")
