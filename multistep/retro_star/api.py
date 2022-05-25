import logging

from retro_star.model import ValueMLP
from retro_star.utils import setup_logger
from retro_star.common import prepare_starting_molecules, prepare_mlp, \
    prepare_molstar_planner, smiles_to_fp, onmt_trans

import torch


class RSPlanner(object):
    def __init__(self, gpu, expansion_topk, iterations, use_value_fn, buliding_block_path, mlp_templates_path, fp_dim,
                 one_step_model_path, value_fn_model_path, one_step_model_type, beam_size, route_topk, viz, viz_dir):

        setup_logger()
        device = torch.device('cuda:%d' % gpu if gpu >= 0 and torch.cuda.is_available() else 'cpu')
        print(device)
        starting_mols = prepare_starting_molecules(buliding_block_path)
        print("number of starting mols: ", len(starting_mols))

        assert one_step_model_type in ['onmt', 'mlp']
        if one_step_model_type == 'mlp':
            one_step = prepare_mlp(mlp_templates_path, one_step_model_path)
            one_step_handler = lambda x: one_step.run(x, topk=expansion_topk)
        elif one_step_model_type == 'onmt':
            one_step_handler = lambda x: onmt_trans(
                x,
                topk=expansion_topk,
                model_path=one_step_model_path,
                beam_size=beam_size,
                device=gpu
            )

        self.top_k = route_topk

        if use_value_fn:
            model = ValueMLP(
                n_layers=1,
                fp_dim=fp_dim,
                latent_dim=128,
                dropout_rate=0.1,
                device=device
            ).to(device)
            logging.info('Loading value nn from %s' % value_fn_model_path)
            model.load_state_dict(torch.load(value_fn_model_path, map_location=device))
            model.eval()

            def value_fn(mol):
                fp = smiles_to_fp(mol, fp_dim=fp_dim).reshape(1, -1)
                fp = torch.FloatTensor(fp).to(device)
                v = model(fp).item()
                return v
        else:
            value_fn = lambda x: 0.

        self.plan_handle = prepare_molstar_planner(
            expansion_handler=one_step_handler,
            value_fn=value_fn,
            starting_mols=starting_mols,
            iterations=iterations,
            viz=viz,
            viz_dir=viz_dir,
            route_topk=route_topk
        )

    def plan(self, target_mol):
        succ, msg = self.plan_handle(target_mol)

        if succ:
            ori_list = msg[3]
            routes_list = []
            for i in ori_list:
                routes_list.append(i.serialize_with_score())
            return routes_list[:self.top_k]
            
        else:
            logging.info('Synthesis path for %s not found. Please try increasing '
                         'the number of iterations.' % target_mol)
            return None


if __name__ == '__main__':
    planner = RSPlanner(
        gpu=0,
        use_value_fn=True,
        iterations=100,
        expansion_topk=50
    )

    result = planner.plan('CCCC[C@@H](C(=O)N1CCC[C@H]1C(=O)O)[C@@H](F)C(=O)OC')
    print(result)

    result = planner.plan('CCOC(=O)c1nc(N2CC[C@H](NC(=O)c3nc(C(F)(F)F)c(CC)[nH]3)[C@H](OC)C2)sc1C')
    print(result)

    result = planner.plan('CC(C)c1ccc(-n2nc(O)c3c(=O)c4ccc(Cl)cc4[nH]c3c2=O)cc1')
    print(result)

