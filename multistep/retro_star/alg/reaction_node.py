import numpy as np
import logging


class ReactionNode:
    def __init__(self, parent, cost, template):
        self.parent = parent
        
        self.depth = self.parent.depth + 1
        self.id = -1

        self.cost = cost
        self.template = template
        self.children = []
        self.value = None   # [V(m | subtree_m) for m in children].sum() + cost
        self.succ_value = np.inf      # self value for existing solution
        self.target_value = None    # V(target | this reaction, search tree)
        self.succ = None    # successfully found a valid synthesis route
        self.open = True    # before expansion: True, after expansion: False
        parent.children.append(self)

        # value for quickly finding one solution
        self.pure_value_self = self.value
        self.pure_value_target = self.target_value

    def v_pure_self(self):
        return self.pure_value_self

    def v_pure_target(self):
        return self.pure_value_target

    def v_self(self):
        """
        :return: [V(m | subtree_m) for m in children].sum() + cost
        """
        return self.value

    def v_target(self):
        """
        :return: V(target | this reaction, search tree)
        """
        return self.target_value

    def init_values(self):
        assert self.open

        self.value = self.cost
        self.succ = True
        for c in self.children:
            self.value += c.value
            self.succ &= c.succ

        if self.succ:
            self.succ_value = self.cost
            for mol in self.children:
                self.succ_value += mol.succ_value

        self.target_value = self.parent.v_target() - self.parent.v_self() + \
                            self.value
        self.open = False

        # planning values
        self.pure_value_self = self.cost
        self.pure_value_target = self.parent.v_pure_target() - self.parent.v_pure_self() + self.pure_value_self

    def backup(self, v_delta, v_pure_delta, from_mol=None):
        self.value += v_delta
        self.target_value += v_delta

        self.pure_value_self += v_pure_delta
        self.pure_value_target += v_pure_delta

        self.succ = True
        for mol in self.children:
            self.succ &= mol.succ

        if self.succ:
            self.succ_value = self.cost
            for mol in self.children:
                self.succ_value += mol.succ_value

        if v_delta != 0:
            assert from_mol
            self.propagate(v_delta, exclude=from_mol)

        return self.parent.backup(self.succ)

    def propagate(self, v_delta, exclude=None):
        if exclude is None:
            self.target_value += v_delta

        check = True
        if exclude is not None:
            check = False
        for child in self.children:
            if exclude is None or child.mol != exclude:
                for grandchild in child.children:
                    grandchild.propagate(v_delta)
            else:
                check = True
        assert check

    def serialize(self):
        # return '%d | value %.2f | target %.2f | pure %.2f | pure target %.2f' % \
        #        (self.id, self.v_self(), self.v_target(), self.v_pure_self(), self.v_pure_target())
        # return '%d | value %.2f | target %.2f' % \
        #        (self.id, self.v_self(), self.v_target())
        return '%d' % (self.id)