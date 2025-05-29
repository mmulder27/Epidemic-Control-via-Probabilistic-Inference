
import numpy as np

def get_beliefs(G, M, variable_nodes=[]):
    """
    GET_BELIEFS - Returns dictionary containing beliefs for each node and
    each clique, respectively.

    Parameters
    ----------
    G : Factor Graph.
    M : factor-to-variable message dictionary
    variable_nodes : optional, precomputed G.get_variable_nodes()

    Returns
    -------
    node_marg : dictionary containing beliefs for each node
                same format as nodeMarg = marg_brute_force(G)

    """

    # node beliefs
    node_marg = {}
    
    if not variable_nodes:
        variable_nodes = G.get_variable_nodes()

    for var in variable_nodes:
        incoming_factor_messages = [M[(factor,var)] for factor in G.neighbors(var)]
        belief = incoming_factor_messages[0]
        for message in incoming_factor_messages[1:]:
            belief = belief * message
        # Normalize
        belief = belief/np.sum(belief)
        # Reshape to column vector to match formatting of nodeMarg = marg_brute_force(G)
        belief = belief.reshape(len(belief), 1)
        node_marg[var] = belief
    
    return node_marg


def get_var_to_factor_messages(G,factor_to_var_messages,var_to_factor_messages):
    """
    GET_VAR_TO_FACTOR_MESSAGES - Updates variable-to-factor messages using factor-to-variable messages

    Parameters
    ----------
    G: Factor Graph
    factor_to_var_messages: dictionary of messages for (factor,var) tuples
    var_to_factor_messages: dictionary of messages for (var,factor) tuples
    
    Returns
    -------
    None
    """
    for var in G.get_variable_nodes():
        fac_neighbors = list(G.neighbors(var))
        for f_idx, f in enumerate(fac_neighbors):
            msg_prod = np.ones(G.get_cardinality(var))

            for i, other in enumerate(fac_neighbors):
                if i == f_idx:
                    continue
                msg_prod *= np.squeeze(factor_to_var_messages[(other, var)])

            # Normalize messages to avoid numerical underflow
            msg_prod /= msg_prod.sum()
            
            var_to_factor_messages[(var, f)] = msg_prod


def get_factor_to_var_messages(G,var_to_factor_messages,factor_to_var_messages):
    """
    GET_FACTOR_TO_VAR_MESSAGES - Updates factor-to-variable messages using variable-to-factor messages

    Parameters
    ----------
    G: Factor Graph
    var_to_factor_messages: dictionary of messages for (var,factor) tuples
    factor_to_var_messages: dictionary of messages for (factor,var) tuples
    
    Returns
    -------
    None
    """
    # Compute factor-to-variable messages
    for factor in G.get_factors():
        var_neighbors = factor.scope()
        k = len(var_neighbors)
        for v_idx, var in enumerate(var_neighbors):
            # start from raw factor table every time
            prod = factor.values.copy()
    
            for i, other in enumerate(var_neighbors):
                if i == v_idx:
                    continue
                msg = var_to_factor_messages[(other, factor)].squeeze()   # 1-D
                prod *= msg.reshape([1]*i + [-1] + [1]*(k-i-1))           # multiply message along appropriate axis of the potential array
            # Sum out all variables except the one receiving this message
            msg_to_var = prod.sum(axis=tuple(j for j in range(k) if j != v_idx))
            
            # Normalize messages to avoid numerical underflow
            msg_to_var /= msg_to_var.sum()
        
            factor_to_var_messages[(factor, var)] = msg_to_var


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
