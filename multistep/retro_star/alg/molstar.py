import os
import numpy as np
import logging
import time

from .mol_tree import MolTree
from retro_star.utils.filter_rules import manual_rules_for_rxn_without_rdkit


def molstar(target_mol, target_mol_id, starting_mols, expand_fn, value_fn, iterations, viz=False, viz_dir=None, route_topk=5):
    mol_tree = MolTree(target_mol=target_mol, known_mols=starting_mols, value_fn=value_fn)

    i = -1
    succ_values = []

    if not mol_tree.succ:
        for i in range(iterations):
            if not (i + 1) % 10:
                logging.info('No.%s iteration is going on ...' % (i + 1))
            scores = []
            for m in mol_tree.mol_nodes:
                if m.open:
                    scores.append(m.v_target())
                else:
                    scores.append(np.inf)
            scores = np.array(scores)

            if np.min(scores) == np.inf:
                logging.info('No open nodes!')
                break

            metric = scores

            mol_tree.search_status = np.min(metric)
            m_next = mol_tree.mol_nodes[np.argmin(metric)]
            assert m_next.open

            t = time.time()
            result = expand_fn(m_next.mol)
            result = manual_rules_for_rxn_without_rdkit(target_mol, result)
            mol_tree.expand_fn_time += (time.time() - t)

            if result is not None and (len(result['scores']) > 0):
                reactants = result['reactants']
                scores = result['scores']
                # costs = 0.0 - np.log(np.clip(np.array(scores), 1e-3, 1.0))
                costs = 0.0 - np.log(np.clip(np.array(scores), 0., 1.0))
                # costs = 1.0 - np.array(scores)
                if 'templates' in result.keys():
                    templates = result['templates']
                else:
                    templates = result['template']

                reactant_lists = []
                for j in range(len(scores)):
                    reactant_list = list(set(reactants[j].split('.')))
                    reactant_lists.append(reactant_list)

                assert m_next.open
                mol_tree.expand(m_next, reactant_lists, costs, templates)
                
                succ_values.append(mol_tree.root.succ_value)

            else:
                mol_tree.expand(m_next, None, None, None)
                succ_values.append(mol_tree.root.succ_value)
                logging.info('Expansion fails on %s!' % m_next.mol)

        logging.info('Final search status | success value | iter: %s | %s | %d'
                     % (str(mol_tree.search_status), str(mol_tree.root.succ_value), i+1))

    best_route = None
    routes = []
    if mol_tree.succ:
        best_route = mol_tree.get_best_route()
        routes = mol_tree.get_routes()
        routes = sorted(routes, key=lambda x: x.total_cost)
        assert best_route is not None

    possible = True
    if viz:
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)

        if mol_tree.succ:
            if best_route.optimal:
                f = '%s/mol_%d_route_optimal' % (viz_dir, target_mol_id)
            else:
                f = '%s/mol_%d_route' % (viz_dir, target_mol_id)
            best_route.viz_route(f)

            for i, route in enumerate(routes):
                if i == route_topk:
                    break

                filename = 'route_cost_%.4f' % route.total_cost
                viz_file_path = os.path.join(viz_dir, filename)
                route.viz_route(viz_file_path)

            f = '%s/mol_%d_search_tree' % (viz_dir, target_mol_id)
            mol_tree.viz_search_tree(f)

        else:
            logging.info('Unable to find the solution with the current one step model')

    return mol_tree.succ, (best_route, possible, i+1, routes, succ_values)
