import os
import sys
import pickle
import logging

import pandas as pd

from retro_star.alg import molstar
from mlp_retrosyn.mlp_inference import MLPModel
from onmt.bin.translate import load_model, run


def prepare_starting_molecules(filename):
    # 输入直接就是starting mols了，不从文件里读
    if isinstance(filename, list):
        return filename

    logging.info('Loading starting molecules from %s' % filename)
    if filename[-3:] == 'csv':
        starting_mols = set(list(pd.read_csv(filename)['mol']))
    else:
        assert filename[-3:] == 'pkl'
        with open(filename, 'rb') as f:
            starting_mols = pickle.load(f)

    logging.info('%d starting molecules loaded' % len(starting_mols))
    return starting_mols


def prepare_mlp(templates, model_dump):
    logging.info('Templates: %s' % templates)
    logging.info('Loading trained mlp model from %s' % model_dump)
    one_step = MLPModel(model_dump, templates, device=-1)
    return one_step


def onmt_trans(x, topk, model_path, beam_size=20, device='cpu'):
    opt, translator = load_model(
        model_path=model_path,
        beam_size=beam_size,
        topk=topk,
        device=device,
        tokenizer='char')
    res_dict = run(translator, opt, x)
    res_dict['templates'] = [None for _ in range(len(res_dict['scores']))]
    return res_dict


def prepare_molstar_planner(expansion_handler, value_fn, starting_mols, iterations, viz=False, viz_dir=None, route_topk=5):
    plan_handler = lambda x, y=0: molstar(
        target_mol=x,
        target_mol_id=y,
        starting_mols=starting_mols,
        expand_fn=expansion_handler,
        value_fn=value_fn,
        iterations=iterations,
        viz=viz,
        viz_dir=viz_dir,
        route_topk=route_topk
    )
    return plan_handler


def prepare_MCTS_planner(one_step, starting_mols,
                         expansion_beam, expansion_topk,
                         rollout_beam, rollout_topk, iterations, max_depth):
    if use_gln:
        expansion_handle = lambda x: one_step.run(x,
                                                  beam_size=expansion_beam,
                                                  topk=expansion_topk)
        rollout_handle = lambda x: one_step.run(x,
                                                beam_size=rollout_beam,
                                                topk=rollout_topk)
    else:
        expansion_handle = lambda x: one_step.run(x, topk=expansion_topk)
        rollout_handle = lambda x: one_step.run(x, topk=rollout_topk)

    plan_handle = lambda x, y: mcts_plan(
        target_mol=x,
        expansion_handle=expansion_handle,
        rollout_handle=rollout_handle,
        starting_mols=starting_mols,
        iterations=iterations,
        max_depth=max_depth
    )
    return plan_handle
