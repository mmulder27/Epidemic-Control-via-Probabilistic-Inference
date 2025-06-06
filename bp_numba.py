import numpy as np
from numba import njit, prange

# ----------------------------------------------------------------------
# NUMBA HELPERS
# ----------------------------------------------------------------------

@njit(fastmath=True, cache=True)
def _compute_message(K, M,
                     Xi_vals, Xij_vals, S_table,
                     pO, R_tbl,
                     pref_prod, one_minus_gamma_pow,
                     prod0_excl, prod1_excl, idx_j):
    """
    Core double-loop for a single edge (i,j).
    Returns the flattened, normalised message vector (shape M*M → (M*M,)).
    """
    msg = np.zeros((M, M), dtype=np.float64)

    for col in range(M):              # s_ij
        for row in prange(M):         # s_ji
            acc = 0.0
            sji_val = Xij_vals[row]

            # ---------------- Part A : t_i < s_ji -------------------
            if not np.isinf(sji_val):
                for ti_idx in range(K):
                    t_val = Xi_vals[ti_idx]
                    if np.isinf(t_val) or t_val >= sji_val:
                        continue

                    pref = pref_prod[ti_idx]

                    for ri_idx in range(K):
                        r_val = Xi_vals[ri_idx]
                        if (not np.isinf(r_val)) and r_val < sji_val:
                            continue

                        Sval = S_table[col, ti_idx, ri_idx]
                        if Sval == 0.0:
                            continue

                        obs  = pO[ti_idx, ri_idx]
                        rec  = R_tbl[ri_idx, ti_idx]
                        prod0 = prod0_excl[ti_idx, ri_idx, idx_j]
                        prod1 = prod1_excl[ti_idx, ri_idx, idx_j]
                        acc  += obs * rec * Sval * pref * (
                                   prod0 - one_minus_gamma_pow[ti_idx] * prod1)

            # ---------------- Part B : t_i == s_ji ------------------
            for ti_idx in range(K):
                if Xi_vals[ti_idx] == Xij_vals[row]:
                    for ri_idx in range(K):
                        r_val = Xi_vals[ri_idx]
                        if (not np.isinf(r_val)) and r_val < Xij_vals[row]:
                            continue
                        obs  = pO[ti_idx, ri_idx]
                        rec  = R_tbl[ri_idx, ti_idx]
                        prod0 = prod0_excl[ti_idx, ri_idx, idx_j]
                        acc  += obs * rec * prod0
                    break   # only one ti_idx can match

            msg[row, col] = acc

    s = msg.sum()
    return (msg if s == 0.0 else msg / s).ravel()

@njit(fastmath=True, cache=True)
def build_prod_tables(G0_all, G1_all):
    """
    G0_all, G1_all : (K, K, nN) float64
    Returns prod0_excl, prod1_excl with the same shape.
    """
    K, _, nN = G0_all.shape
    prod0 = np.empty((K, K, nN), dtype=np.float64)
    prod1 = np.empty_like(prod0)

    for ti in prange(K):
        for ri in range(K):
            # --------- prod0 (from G0_all) ----------
            pref = 1.0
            for s in range(nN):              # prefix pass
                prod0[ti, ri, s] = pref
                pref *= G0_all[ti, ri, s]
            pref = 1.0
            for s in range(nN - 1, -1, -1):  # suffix pass
                prod0[ti, ri, s] *= pref
                pref *= G0_all[ti, ri, s]

            # --------- prod1 (from G1_all) ----------
            pref = 1.0
            for s in range(nN):
                prod1[ti, ri, s] = pref
                pref *= G1_all[ti, ri, s]
            pref = 1.0
            for s in range(nN - 1, -1, -1):
                prod1[ti, ri, s] *= pref
                pref *= G1_all[ti, ri, s]

    return prod0, prod1
