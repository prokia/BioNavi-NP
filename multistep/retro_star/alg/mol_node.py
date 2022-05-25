import numpy as np
import logging


class MolNode:
    def __init__(self, mol, init_value, parent=None, is_known=False,
                 zero_known_value=True):
        self.mol = mol
        self.pred_value = init_value
        self.value = init_value
        self.succ_value = np.inf      # self value for existing solution
        self.parent = parent

        self.id = -1
        if self.parent is None:
            self.depth = 0
        else:
            self.depth = self.parent.depth

        self.is_known = is_known
        self.children = []
        self.succ = is_known
        self.open = True  # before expansion: True, after expansion: False
        if is_known:
            self.open = False
            if zero_known_value:
                self.value = 0
            self.succ_value = self.value

        # value for setting mol value to 0
        self.pure_value_self = 0

        if parent is not None:
            parent.children.append(self)

    def v_pure_self(self):
        return self.pure_value_self

    def v_pure_target(self):
        if self.parent is None:
            return self.v_pure_self()
        else:
            return self.parent.v_pure_target()

    def v_self(self):
        """
        :return: V(self | subtree)
        """
        return self.value

    def v_target(self):
        """
        :return: V(target | this molecule, search tree)
        """
        if self.parent is None:
            return self.value
        else:
            return self.parent.v_target()

    def init_values(self, no_child=False):
        assert self.open
        assert self.open and (no_child or self.children)

        new_value = np.inf
        new_pure_value = np.inf
        self.succ = False
        for reaction in self.children:
            new_value = np.min((new_value, reaction.v_self()))
            new_pure_value = np.min((new_pure_value, reaction.v_pure_self()))
            self.succ |= reaction.succ

        v_delta = new_value - self.value
        self.value = new_value

        v_pure_delta = new_pure_value - self.pure_value_self
        self.pure_value_self = new_pure_value

        if self.succ:
            for reaction in self.children:
                self.succ_value = np.min((self.succ_value, reaction.succ_value))

        self.open = False

        return v_delta, v_pure_delta

    def backup(self, succ):
        assert not self.is_known

        new_value = np.inf
        new_pure_value = np.inf
        for reaction in self.children:
            new_value = np.min((new_value, reaction.v_self()))
            new_pure_value = np.min((new_pure_value, reaction.v_pure_self()))
        new_succ = self.succ | succ
        updated = (self.value != new_value) or (self.succ != new_succ) or (self.pure_value_self != new_pure_value)

        new_succ_value = np.inf
        if new_succ:
            for reaction in self.children:
                new_succ_value = np.min((new_succ_value, reaction.succ_value))
            updated = updated or (self.succ_value != new_succ_value)

        v_delta = new_value - self.value
        v_pure_delta = new_pure_value - self.pure_value_self
        self.value = new_value
        self.pure_value_self = new_pure_value
        self.succ = new_succ
        self.succ_value = new_succ_value

        if updated and self.parent:
            return self.parent.backup(v_delta, v_pure_delta, from_mol=self.mol)

    def serialize(self):
        # text = '%d | %s | pred %.2f | value %.2f | target %.2f | pure %.2f' % \
        #        (self.id, self.mol, self.pred_value, self.v_self(), self.v_target(), self.v_pure_self())
        text = '%d | %s' % (self.id, self.mol)
        # text = '%d | %s | pred %.2f | value %.2f | target %.2f' % \
        #        (self.id, self.mol, self.pred_value, self.v_self(), self.v_target())
        # if self.is_known:
        #     text += ' | known'
        # if self.open:
        #     text += ' | open'
        return text

    def get_ancestors(self):
        if self.parent is None:
            return {self.mol}

        ancestors = self.parent.parent.get_ancestors()
        ancestors.add(self.mol)
        return ancestors