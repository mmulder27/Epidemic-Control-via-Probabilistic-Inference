# brew install gsl swig make gcc
import numpy as np
from pgmpy.models import FactorGraph
import math
from potential_utils import compute_obs_likelihood_table, compute_recovery_table
from pgmpy.factors.discrete import DiscreteFactor
import math
from bp_numba import _compute_message, build_prod_tables

class StubFactor:
    def __init__(self, scope, card, i):
        """
        scope:   list of variable node names, e.g. ["t_0","r_0","auto_0","s_pair_0_1"]
        card:    list of same length, each entry = cardinality of that variable
        """
        self.id = i
        self._scope = list(scope)
        self._card  = list(card)

    def get_scope(self):
        return list(self._scope)
    
    def add_to_scope(self,var):
        self.scope.append(var)

    def add_cardinality(self,x):
        self._card.append()

    def get_cardinality(self, variables):
        """
        Given a list of variable names, return a dict mapping each to its cardinality.
        """
        result = {}
        for var in variables:
            if var not in self._scope:
                raise ValueError(f"{var} is not in factor scope {self._scope}")
            idx = self._scope.index(var)
            result[var] = self._card[idx]
        return result


    

class SimpleFactorGraph(FactorGraph):
    # simplify some methods and error-checking in FactorGraph for efficiency
    def __init__(self):
        super(SimpleFactorGraph, self).__init__()
        self.card_dict = {}

    def get_variable_nodes(self):
        variable_nodes = set([x for factor in self.factors for x in factor.get_scope()])
        return list(variable_nodes)

    def add_factors(self, *factors):
        for factor in factors:
            self.factors.append(factor)
            for var in factor.get_scope():
                self.card_dict[var] = factor.get_cardinality([var])[var]

    def remove_node(self, f):
        super().remove_node(f)
        if f in self.card_dict.keys():
            del self.card_dict[f]


def belief_diff(b1, b2):
    """
    BELIEF_DIFF - Computes the symmetric L1 distance between beliefs 'b1'
    and 'b2'. 
    
    Parameters
    ----------
    b1, b2 : N-entry dictionaries of variables and their corresponding marginals.

    Returns
    -------
    D : N-dimensional numpy array of L1 distances.
    """

    num_b = len(b1)
    D = np.zeros(num_b)
    for i, key in enumerate(b1.keys()):
        D[i] = 0.5 * np.sum(abs(b1[key] - b2[key]))

    return D


import numpy as np

def get_beliefs(G, M):
    """
    Return a dictionary   node_marg[var]  →  belief vector.

    Parameters
    ----------
    G : SimpleFactorGraph (or pgmpy FactorGraph)
        Must expose
            • get_variable_nodes()            – iterable of variable names
            • neighbors(var)                  – iterable of neighbouring factors
            • card_dict[var]                  – cardinality of the variable
    M : dict
        Factor-to-variable messages.
        Key   : (factor_obj , variable_name)
        Value : 1-D NumPy array of length card(var)   or the scalar 1.0
                (scalar means “all-ones” to save space).

    Returns
    -------
    node_marg : dict
        belief vectors *normalised to sum to 1*.
        One entry for every variable node in the graph.
    """

    node_marg = {}

    for var in G.get_variable_nodes():

        card = G.card_dict[var]
        belief = np.ones(card, dtype=float)          # start with ones

        for factor in G.neighbors(var):

            msg = M.get((factor, var), 1.0)          # default → scalar 1.0

            if isinstance(msg, float):               # placeholder → all-ones
                # multiplying by 1 leaves 'belief' unchanged
                continue

            # msg is a NumPy vector of the correct length
            belief *= msg

        # normalise (fallback to uniform if it happened to be all zeros)
        s = belief.sum()
        if s == 0.0:
            belief[:] = 1.0 / card
        else:
            belief /= s

        node_marg[var] = belief          # 1-D vector length = card

    return node_marg



def get_var_to_factor_messages(G,
                               factor_to_var_messages,
                               var_to_factor_messages,
                               cache):
    """
    Fast forward pass: for each variable node var, send ψ→var messages back out.

    - Degree-1 var: send a pre-computed uniform vector.
    - Degree-2 var: swap the two incoming ψ→var messages.

    `cache` must be the tuple (var_deg1, var_deg2, uniform_by_card) produced by build_var_to_factor_cache().
    """
    var_deg1, var_deg2, uniform_by_card = cache

    # 1) Degree-1: var→sole neighbour gets uniform
    for var, fA in var_deg1:
        card = G.card_dict[var]  # O(1) dict lookup, no G.neighbors() needed
        var_to_factor_messages[(var, fA)] = uniform_by_card[card]

    # 2) Degree-2: swap the two incoming ψ→var messages
    get = factor_to_var_messages.__getitem__
    put = var_to_factor_messages.__setitem__

    for var, fA, fB in var_deg2:
        # retrieve the two incoming messages (no squeeze needed if stored as 1D)
        msg_A = get((fA, var))
        msg_B = get((fB, var))
        put((var, fA), msg_B)
        put((var, fB), msg_A)



def make_G_tables(i, k,
                  Xi_vals,            # (K,)
                  Xik_vals,           # (Ne,)
                  Sik,                # (Ne, K, K)
                  msg_vec):           # 1-D size Ne² or scalar 1.0
    """
    Return two (K, K) arrays: G0 (≥) and G1 (>).
    No Python loops, no dict look-ups.
    """
    Ne = Xik_vals.size
    # broadcast Xi_vals so we can compare against all edge times
    Xi_grid = Xi_vals[None, None, :]               # (1,1,K)
    Xik_g   = Xik_vals[:, None, None]              # (Ne,1,1)

    # mask: edge time >= / > t_i
    GE_mask = (Xik_g >= Xi_grid)
    GT_mask = (Xik_g >  Xi_grid)

    if isinstance(msg_vec, float):                 # placeholder 1.0
        # rows = GE_mask or GT_mask; cols = GT_mask
        G0 = (Sik * GT_mask).sum(0) * GE_mask.sum(0)
        G1 = (Sik * GT_mask).sum(0) * GT_mask.sum(0)
        return G0, G1

    msg_mat = msg_vec.reshape(Ne, Ne)
    # row/col selection via masks and einsum
    prod_GE = np.einsum('rc, r -> c', msg_mat, GE_mask[:, 0, 0])
    prod_GT = np.einsum('rc, r -> c', msg_mat, GT_mask[:, 0, 0])
    G0 = (Sik * prod_GE[:, None, None] * GT_mask).sum(0)
    G1 = (Sik * prod_GT[:, None, None] * GT_mask).sum(0)
    return G0, G1



def get_factor_to_var_messages(G,
                                var_to_factor_messages,
                                factor_to_var_messages,
                                X_i,
                                X_edges,
                                R_i,
                                pO_i,
                                S_ij,
                                person_factors,
                                gamma):

    inf = np.inf
    K_max = max(len(lst) for lst in X_i.values())

    # γ-dependent scalars ----------------------------------------------------
    pref_prod           = np.ones(K_max, dtype=np.float64)
    one_minus_gamma_pow = np.empty(K_max, dtype=np.float64)
    running = 1.0
    for t in range(K_max):
        pref_prod[t]           = running
        one_minus_gamma_pow[t] = 1.0 - gamma ** t
        running *= (1.0 - gamma ** t)

    # -----------------------------------------------------------------------
    for i, psi_i in person_factors.items():

        # Xi ----------------------------------------------------------------
        Xi_list = X_i[i]
        Kp1     = len(Xi_list)
        Xi_vals = np.array([x if x is not None else inf for x in Xi_list],
                           dtype=np.float64)

        if i not in R_i:
            R_i[i] = compute_recovery_table(Xi_list)
        if i not in pO_i:
            pO_i[i] = np.ones((Kp1, Kp1), dtype=float)

        R_tbl = R_i[i].astype(np.float64)
        pO    = pO_i[i].astype(np.float64)

        # neighbours --------------------------------------------------------
        neigh = sorted({b if a == i else a
                        for (a, b) in X_edges
                        if a == i or b == i})
        nN = len(neigh)

        # G0/G1 tensors -----------------------------------------------------
        G0_all = np.empty((Kp1, Kp1, nN), dtype=np.float64)
        G1_all = np.empty_like(G0_all)

        for k_pos, k in enumerate(neigh):
            edge_key  = (min(i, k), max(i, k))
            Xik_vals  = np.array([v if v is not None else inf
                                  for v in X_edges[edge_key]],
                                  dtype=np.float64)
            Sik       = S_ij[(i, k)].astype(np.float64)
            edge_var  = f"s_pair_{edge_key[0]}_{edge_key[1]}"
            msg_vec   = var_to_factor_messages.get((edge_var, psi_i), 1.0)

            G0_all[:, :, k_pos], G1_all[:, :, k_pos] = make_G_tables(
                i, k, Xi_vals, Xik_vals, Sik, msg_vec
            )

        # prefix/suffix products -------------------------------------------
        prod0_excl, prod1_excl = build_prod_tables(G0_all, G1_all)

        # unary messages ----------------------------------------------------
        uniform = np.full(Kp1, 1.0 / Kp1, dtype=np.float64)
        for vname in (f"t_{i}", f"r_{i}", f"auto_{i}"):
            factor_to_var_messages[(psi_i, vname)] = uniform.copy()

        # edge messages -----------------------------------------------------
        for idx_j, j in enumerate(neigh):
            edge_key  = (min(i, j), max(i, j))
            edge_var  = f"s_pair_{edge_key[0]}_{edge_key[1]}"
            Xij_vals  = np.array([x if x is not None else inf
                                  for x in X_edges[edge_key]],
                                  dtype=np.float64)
            M         = Xij_vals.size
            S_table   = S_ij[(i, j)].astype(np.float64)

            msg_vec = _compute_message(      # numba-JIT
                Kp1, M,
                Xi_vals, Xij_vals, S_table,
                pO, R_tbl,
                pref_prod, one_minus_gamma_pow,
                prod0_excl, prod1_excl,
                idx_j
            )
            factor_to_var_messages[(psi_i, edge_var)] = msg_vec



def run_loopy_bp_parallel(G, max_iters, conv_tol, X_i, X_edges, R_i, pO_i, S_ij, person_factors, gamma, cache):
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
    """

    nodeMargs = []
    factor_to_var_messages = {}
    var_to_factor_messages = {}
    # Initialize by setting the variable-to-factor messages to equal 1 for all states
    for variable in G.get_variable_nodes():
        neighbors = G.neighbors(variable)
        for factor in neighbors:
            var_to_factor_messages[(variable,factor)] = 1.0
            #var_to_factor_messages[(variable,factor)] = np.ones(G.get_cardinality(variable))
    for i in range(max_iters):
        print(i)
        import time
        t0 = time.perf_counter()
        # Update factor-to-variable messages given the current variable-to-factor messages
        get_factor_to_var_messages(G,
                               var_to_factor_messages,
                               factor_to_var_messages,
                               X_i,           # dict: i -> list (len K+1)
                               X_edges,       # dict: (a,b) -> list (len M+1)
                               R_i,           # dict: i -> (K+1,K+1) array
                               pO_i,          # dict: i -> (K+1,K+1) array
                               S_ij,          # dict: (a,b) -> (M+1,K+1,K+1)
                               person_factors, # dict: i -> StubFactor object
                               gamma)
        # ...then update all variable-to-factor messages given the current factor-to-variable messages
        get_var_to_factor_messages(G,factor_to_var_messages,var_to_factor_messages,cache)
        print("elapsed time for this message passing", time.perf_counter() - t0, "seconds")
        beliefs = get_beliefs(G,factor_to_var_messages)
        nodeMargs.append(beliefs)
        if i==0:
            continue
        beliefs_diff = belief_diff(nodeMargs[i],nodeMargs[i-1])
        if np.max(beliefs_diff) < conv_tol:
            return nodeMargs
        
    
    return nodeMargs