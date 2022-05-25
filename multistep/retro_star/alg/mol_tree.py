import numpy as np
import time
from queue import Queue
import logging
import networkx as nx
from copy import deepcopy
from graphviz import Digraph
from networkx.drawing.nx_pydot import write_dot
from .mol_node import MolNode
from .reaction_node import ReactionNode
from .syn_route import SynRoute


class MolTree:
    def __init__(self, target_mol, known_mols, value_fn, zero_known_value=True):
        self.target_mol = target_mol
        self.known_mols = known_mols
        self.value_fn = value_fn
        self.zero_known_value = zero_known_value
        self.mol_nodes = []
        self.reaction_nodes = []

        self.value_fn_time = 0
        self.expand_fn_time = 0

        self.root = self._add_mol_node(target_mol, None)
        self.succ = target_mol in known_mols
        self.search_status = 0

        if self.succ:
            logging.info('Synthesis route found: target in starting molecules')

    def _add_mol_node(self, mol, parent):
        # mol = cano_smi(mol)
        is_known = mol in self.known_mols

        t = time.time()
        init_value = self.value_fn(mol)
        self.value_fn_time += (time.time() - t)

        mol_node = MolNode(
            mol=mol,
            init_value=init_value,
            parent=parent,
            is_known=is_known,
            zero_known_value=self.zero_known_value
        )
        self.mol_nodes.append(mol_node)
        mol_node.id = len(self.mol_nodes)

        # if not is_known:
        #     mol_node.value = - mol_node.depth

        return mol_node

    def _add_reaction_and_mol_nodes(self, cost, mols, parent, template, ancestors):
        assert cost >= 0

        for mol in mols:
            if mol in ancestors:
                return

        reaction_node = ReactionNode(parent, cost, template)
        for mol in mols:
            self._add_mol_node(mol, reaction_node)
        reaction_node.init_values()
        self.reaction_nodes.append(reaction_node)
        reaction_node.id = len(self.reaction_nodes)

        return reaction_node

    def expand(self, mol_node, reactant_lists, costs, templates):
        assert not mol_node.is_known and not mol_node.children

        if costs is None:      # No expansion results
            assert mol_node.init_values(no_child=True)[0] == np.inf
            if mol_node.parent:
                mol_node.parent.backup(np.inf, np.inf, from_mol=mol_node.mol)
            return self.succ

        assert mol_node.open
        ancestors = mol_node.get_ancestors()
        for i in range(len(costs)):
            self._add_reaction_and_mol_nodes(costs[i], reactant_lists[i],
                                             mol_node, templates[i], ancestors)

        assert mol_node.open
        if len(mol_node.children) == 0:      # No valid expansion results
            assert mol_node.init_values(no_child=True)[0] == np.inf
            if mol_node.parent:
                mol_node.parent.backup(np.inf, np.inf, from_mol=mol_node.mol)
            return self.succ

        v_delta, v_plan_delta = mol_node.init_values()
        if mol_node.parent:
            mol_node.parent.backup(v_delta, v_plan_delta, from_mol=mol_node.mol)

        if not self.succ and self.root.succ:
            logging.info('Synthesis route found!')
            self.succ = True
        
        return self.succ

    def get_best_route(self):
        if not self.succ:
            return None

        syn_route = SynRoute(
            target_mol=self.root.mol,
            succ_value=self.root.succ_value,
            search_status=self.search_status
        )

        mol_queue = Queue()
        mol_queue.put(self.root)
        while not mol_queue.empty():
            mol = mol_queue.get()
            if mol.is_known:
                syn_route.set_value(mol.mol, mol.succ_value)
                continue

            best_reaction = None
            for reaction in mol.children:
                if reaction.succ:
                    if best_reaction is None or \
                            reaction.succ_value < best_reaction.succ_value:
                        best_reaction = reaction
            assert best_reaction.succ_value == mol.succ_value

            reactants = []
            for reactant in best_reaction.children:
                mol_queue.put(reactant)
                reactants.append(reactant.mol)

            syn_route.add_reaction(
                mol=mol.mol,
                value=mol.succ_value,
                template=best_reaction.template,
                reactants=reactants,
                cost=best_reaction.cost
            )

            # print('succ value: %.2f | total cost: %.2f | length: %d' %
            #       (syn_route.succ_value, syn_route.total_cost, syn_route.length))

        return syn_route

    def get_routes(self):
        if not self.succ:
            return None

        routes = []
        syn_route = SynRoute(
            target_mol=self.root.mol,
            succ_value=self.root.succ_value,
            search_status=self.search_status
        )
        routes.append(syn_route)

        mol_queue = Queue()
        mol_queue.put((syn_route, self.root))
        while not mol_queue.empty():
            syn_route, mol = mol_queue.get()
            if mol.is_known:
                syn_route.set_value(mol.mol, mol.succ_value)
                continue

            best_reaction = None
            all_reactions = []
            for reaction in mol.children:
                if reaction.succ:
                    all_reactions.append(reaction)
                    if best_reaction is None or \
                            reaction.succ_value < best_reaction.succ_value:
                        best_reaction = reaction
            # assert best_reaction.succ_value == mol.succ_value

            syn_route_template = None
            if len(all_reactions) > 1:
                syn_route_template = deepcopy(syn_route)

            # best reaction
            reactants = []
            for reactant in best_reaction.children:
                mol_queue.put((syn_route, reactant))
                reactants.append(reactant.mol)

            syn_route.add_reaction(
                mol=mol.mol,
                value=mol.succ_value,
                template=best_reaction.template,
                reactants=reactants,
                cost=best_reaction.cost
            )

            # other reactions
            if len(all_reactions) > 1:
                for reaction in all_reactions:
                    if reaction == best_reaction:
                        continue

                    syn_route = deepcopy(syn_route_template)
                    routes.append(syn_route)

                    reactants = []
                    for reactant in reaction.children:
                        mol_queue.put((syn_route, reactant))
                        reactants.append(reactant.mol)

                    syn_route.add_reaction(
                        mol=mol.mol,
                        value=mol.succ_value,   # might be incorrect
                        template=reaction.template,
                        reactants=reactants,
                        cost=reaction.cost
                    )

            # print('succ value: %.2f | total cost: %.2f | length: %d' %
            #       (syn_route.succ_value, syn_route.total_cost, syn_route.length))

        return routes

    def viz_search_tree(self, viz_file, topk=3):
        # pass
        G = Digraph('G', filename=viz_file)
        G.attr(rankdir='LR')
        G.attr('node', shape='box')
        G.format = 'pdf'

        node_queue = Queue()
        node_queue.put((self.root, None))
        while not node_queue.empty():
            node, parent = node_queue.get()

            if node.open:
                color = 'lightgrey'
            else:
                color = 'aquamarine'

            if hasattr(node, 'mol'):
                shape = 'box'
            else:
                shape = 'rarrow'

            if node.succ:
                color = 'lightblue'
                if hasattr(node, 'mol') and node.is_known:
                    color = 'lightyellow'

            G.node(node.serialize(), shape=shape, color=color, style='filled')

            label = ''
            if hasattr(parent, 'mol'):
                label = '%.2f' % np.exp(-node.cost)
            if parent is not None:
                G.edge(parent.serialize(), node.serialize(), label=label)

            if node.children is not None:
                for c in node.children:
                    node_queue.put((c, node))

        G.render()

    def viz_search_progress(self, route, viz_file):
        target = route[0].split('>')[0]
        assert target == self.target_mol

        reaction_dict = {}

        G = nx.DiGraph()

        for reaction in route:
            parent = reaction.split('>')[0]
            reactants = reaction.split('>')[2].split('.')
            reaction_dict[parent] = set(reactants)

            if parent not in list(G.nodes):
                G.add_node(parent)

            for reactant in reactants:
                G.add_node(reactant)
                G.add_edge(parent, reactant)

        # match
        mapping = {}
        unable_to_find = False
        match_queue = Queue()
        match_queue.put(self.root)
        while not match_queue.empty():
            node = match_queue.get()
            if node.mol not in reaction_dict:
                # starting molecule
                mapping[node.mol] = '%s | %f | CLOSE' % (node.mol, node.v_target())
                continue
            route_reactants = reaction_dict[node.mol]

            if node.open:
                mapping[node.mol] = '%s | %f | OPEN' % (node.mol, node.v_target())
                continue

            mapping[node.mol] = '%s | %f | CLOSE' % (node.mol, node.v_target())

            found_match = False
            for c in node.children:
                reactants_c = set()
                for mol_node in c.children:
                    reactants_c.add(mol_node.mol)

                if reactants_c.issubset(route_reactants):
                    for mol_node in c.children:
                        match_queue.put(mol_node)
                    found_match = True
                    continue
            if not found_match:
                unable_to_find = True

        G = nx.relabel_nodes(G, mapping, copy=False)
        G.graph['node'] = {'shape': 'rect'}
        A = nx.nx_agraph.to_agraph(G)
        A.layout('dot')
        A.draw(viz_file)

        return unable_to_find
