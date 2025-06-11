#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UC Irvine CS274B

Handout version:  Some unimplemented functionality indicated by TODO
"""

import numpy as np
from copy import deepcopy
from pgmpy.models import FactorGraph
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.factors import factor_product
from pgmpy.readwrite import BIFReader
from pgmpy.inference.ExactInference import BeliefPropagation


class SimpleFactorGraph(FactorGraph):
    # simplify some methods and error-checking in FactorGraph for efficiency
    def __init__(self):
        super(SimpleFactorGraph, self).__init__()
        self.card_dict = {}

    def get_variable_nodes(self):
        variable_nodes = set([x for factor in self.factors for x in factor.scope()])
        return list(variable_nodes)

    def add_factors(self, *factors):
        for factor in factors:
            self.factors.append(factor)
            for var in factor.scope():
                self.card_dict[var] = factor.get_cardinality([var])[var]

    def remove_node(self, f):
        super().remove_node(f)
        if f in self.card_dict.keys():
            del self.card_dict[f]

    def get_cardinality(self, variable_name=None):
        """
        Returns the cardinality of the node or a dict over all nodes if not specified.
        """
        if variable_name is not None:
            return self.card_dict[variable_name]
        else:
            return self.card_dict

class FastDiscreteFactor(DiscreteFactor):
    # define faster hash function, depending only on variable names
    def __hash__(self):
        variable_hashes = [hash(variable) for variable in self.variables]
        return hash(sum(variable_hashes))
        # return hash(str(sorted(variable_hashes)))

def belief_diff(b1, b2):
    """
    BELIEF_DIFF - Computes the symmetric L1 distance between belief's 'b1'
    and 'b2'. Inputs are N-entry dictionaries, where each dictionary stores
    the marginal distribution of each variable.  L1 distances are returned
    in an N-dim numpy array.
    """

    num_b = len(b1)
    D = np.zeros(num_b)
    for i, key in enumerate(b1.keys()):
        D[i] = 0.5 * np.sum(abs(b1[key] - b2[key]))

    return D


def get_beliefs(G, M, variable_nodes=None):
    """
    Return a dictionary of node marginals (beliefs) after message passing.

    Parameters
    ----------
    G : factor-graph instance, exposing
        • neighbors(var)              – iterable of adjacent factor nodes
        • get_variable_nodes()        – iterable of all variable nodes
        • card_dict[var] (or .get_cardinality(var))
    M : dict  {(factor , var) : np.ndarray or scalar}
        Factor-to-variable messages produced by BP.
    variable_nodes : iterable, optional
        List of variables for which to compute beliefs.  If None, all
        variable nodes are processed.

    Returns
    -------
    node_marg : dict {var : np.ndarray(shape=(card,1)) }
        Normalised belief for each requested variable.
    """

    if variable_nodes is None:
        variable_nodes = G.get_variable_nodes()

    node_marg = {}

    for var in variable_nodes:
        k = 3                          # domain size
        belief = np.ones(k, dtype=float)           # start with uniform 1’s

        for fac in G.neighbors(var):
            msg = M.get((fac, var), 1.0)           # default to neutral 1
            arr = np.asarray(msg, dtype=float).ravel()

            if arr.size == 1:                      # scalar → broadcast
                arr = np.full(k, arr.item())
            elif arr.size != k:
                raise ValueError(
                    f"Message {fac}->{var} has size {arr.size}, "
                    f"expected {k}"
                )

            belief *= arr                          # element-wise product

        tot = belief.sum()
        if tot == 0.0:
            # all incoming messages zero → fall back to uniform
            belief.fill(1.0 / k)
        else:
            belief /= tot                          # normalise

        node_marg[var] = belief[:, None]           # column vector (k,1)

    return node_marg


import numpy as np

DAMP = 0.5
EPS  = 1e-12


# ------------------------------------------------------------------
# variable  ➜  factor
# ------------------------------------------------------------------
def get_var_to_factor_messages(G,
                               factor_to_variable_messages,
                               variable_to_factor_messages):
    """
    Update every (var → factor) message using current (factor → var) messages.
    Logic, shapes and damping unchanged – only faster.
    """
    for var in G.get_variable_nodes():
        neigh = tuple(G.neighbors(var))
        if not neigh:                  # isolated variable
            continue

        # stack incoming messages once: shape (deg, card)
        incoming = np.stack(
            [np.squeeze(factor_to_variable_messages[(f, var)])
             for f in neigh],
            axis=0,
        )

        all_prod = incoming.prod(axis=0)     # product of *all* incoming
        for idx, f in enumerate(neigh):
            msg = all_prod / incoming[idx]   # leave-one-out product
            msg = np.maximum(msg, EPS)
            msg /= msg.sum()

            old = variable_to_factor_messages.get(
                (var, f), np.full_like(msg, 1.0 / msg.size)
            )
            variable_to_factor_messages[(var, f)] = (
                DAMP * msg + (1.0 - DAMP) * old
            )


# ------------------------------------------------------------------
# factor  ➜  variable
# ------------------------------------------------------------------
def get_factor_to_var_messages(G,
                               variable_to_factor_messages,
                               factor_to_variable_messages):
    """
    Update every (factor → var) message using current (var → factor) messages.
    """
    for factor in G.get_factors():
        scope = factor.scope()         # tuple of variables in this factor
        k = len(scope)
        if k == 0:                     # empty scope
            continue

        msgs = [np.squeeze(variable_to_factor_messages[(v, factor)])
                for v in scope]

        # reshape each message exactly as before
        shaped = [m.reshape([1]*i + [-1] + [1]*(k-i-1))
                  for i, m in enumerate(msgs)]

        prod_all = factor.values.copy()
        for s in shaped:
            prod_all *= s              # multiply all messages into table

        for i, var in enumerate(scope):
            msg = prod_all / shaped[i]                        # remove self-msg
            msg = msg.sum(axis=tuple(j for j in range(k) if j != i))
            msg = np.maximum(msg, EPS)
            msg /= msg.sum()

            old = factor_to_variable_messages.get(
                (factor, var), np.full_like(msg, 1.0 / msg.size)
            )
            factor_to_variable_messages[(factor, var)] = (
                DAMP * msg + (1.0 - DAMP) * old
            )

def run_loopy_bp_parallel(G, max_iters, conv_tol):
    """
    RUN_LOOPY_BP - Runs Loopy Belief Propagation (Sum-Product) on a factor
    Graph given by 'G'. This implements a "parallel" updating scheme in
    which all factor-to-variable messages are updated in a single clock
    cycle, and then all variable-to-factor messages are updated.

    Parameters
    ----------
    G : Factor Graph
    max_iters : max iterations
    conv_tol : convergence tolerance

    Returns
    -------
    nodeMargs : list keeping track of node marginals at each iteration.
                NodeMargs[i] is a dictionary containing beliefs for each node 
                with the same format as nodeMarg = marg_brute_force(G)
    """

    nodeMargs = []
    factor_to_var_messages = {}
    var_to_factor_messages = {}

    # Initialize by setting the variable-to-factor messages to equal 1 for all states
    for variable in G.get_variable_nodes():
        neighbors = G.neighbors(variable)
        for factor in neighbors:
            var_to_factor_messages[(variable,factor)] = np.ones(G.get_cardinality(variable))
    for i in range(max_iters):
        # Update factor-to-variable messages given the current variable-to-factor messages
        get_factor_to_var_messages(G,var_to_factor_messages,factor_to_var_messages)
        # ...then update all variable-to-factor messages given the current factor-to-variable messages
        get_var_to_factor_messages(G,factor_to_var_messages,var_to_factor_messages)
        beliefs = get_beliefs(G,factor_to_var_messages)
        nodeMargs.append(beliefs)
        if i==0:
            continue
        beliefs_diff = belief_diff(nodeMargs[i],nodeMargs[i-1])
        if np.max(beliefs_diff) < conv_tol:
            return nodeMargs
    
    return nodeMargs
