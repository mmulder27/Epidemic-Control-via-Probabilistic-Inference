import numpy as np
import math
from scipy.special import gammainc


def compute_obs_likelihood_table(i_node, X_i_list,test_results,fnr,fpr):
    #`test results` is a list of (person_idx, observed_state, obs_time)
    my_obs = [(s_obs, t_obs) for (j, s_obs, t_obs) in test_results if j == i_node]
    neg_probs = {"S": 1 - fnr, "I": fnr, "R": 1 - fnr}
    pos_probs = {"S": fpr, "I": 1 - fpr, "R": fpr}

    # Create np.array of shape (|X_i|,|X_i|)
    pO_i = np.zeros((len(X_i_list), len(X_i_list)), dtype=float)

    for k, t_i_val in enumerate(X_i_list): 
        for l, r_i_val in enumerate(X_i_list):
            if (t_i_val is not None) and (r_i_val is not None) and (r_i_val < t_i_val):
                # Cannot have “recovery” before “infection”
                pO_i[k,l] = 0.0
                continue
            # Compute the likelihood of all of i’s observations, given (t_i_val, r_i_val)
            likelihood = 1.0
            for (s_obs, t_obs) in my_obs:
                if t_i_val is None:
                    latent = "S"
                elif t_obs < t_i_val:
                    latent = "S"
                elif (r_i_val is None) or (t_obs < r_i_val):
                    latent = "I"
                else:
                    latent = "R"
                
                if s_obs != 1:
                   likelihood *= neg_probs[latent]
                else:
                    likelihood *= pos_probs[latent]

                if likelihood == 0.0:
                    break

            pO_i[k, l] = likelihood

    return pO_i


def lambda_lookup(sij, ti):
    """
    Returns λ(sij | ti) = p_I(sij - ti) as a rescaled Gamma PDF with shape k = 5.76, “mean” μ = 0.96, and overall multiplier 0.25
    """
    k = 5.76
    μ = 0.96
    theta = μ / k

    diff = sij - ti
    if diff <= 0:
        return 0.0

    # Gamma PDF: f(sij_ti) = (sij-ti)^(k−1) * exp(−sij+ti)/θ) / (gamma(k) * θ^k)
    raw_pdf = (diff ** (k - 1) * math.exp(-diff / theta)/(math.gamma(k) * theta**k))

    return 0.25 * raw_pdf

def compute_S_table(i, j, Xij_list, Xi_list):
    """
    Build the truncated geometric transmission table
        S_ij(s_ij | t_i, r_i)
    over all discrete values s_ij in Xij_list, t_i in Xi_list, and r_i in Xi_list

    Parameters
    ----------
    i : int
        Person index for the infectious individual.
    j : int
        Person index for the susceptible individual.

    Xij_list : list of length M+1 (sorted, ascending), where the last entry is None to represent ∞.
        The finite contact times between i and j are Xij_list[0..M-1].
        Xij_list[M] == None stands for “∞” (= no transmission ever).

    Xi_list : list of length K+1 (sorted, ascending), where the last entry is None to represent ∞.
        The finite infection times for person i are Xi_list[0..K-1].
        Xi_list[K] == None stands for “∞” (i never infected or never recovered in window).

    Returns
    -------
    S : np.ndarray of shape (M+1, K+1, K+1)
        S[s_idx, ti_idx, ri_idx] = P(s_{ij} = Xij_list[s_idx]  |  t_i = Xi_list[ti_idx],  r_i = Xi_list[ri_idx]).
    """
    M = len(Xij_list)
    K = len(Xi_list)

    # Allocate the 3D array
    S = np.zeros((M, K, K), dtype=float)

    # Precompute all lambda and (1 - lambda) for each (s_idx, ti_idx)
    lam_array = np.zeros((M, K), dtype=float)
    one_minus_lam = np.ones((M, K), dtype=float)

    for s_idx, s_val in enumerate(Xij_list):
        for ti_idx, ti_val in enumerate(Xi_list):
            if (s_val is None) or (ti_val is None):
                lam_val = 0.0
            else:
                lam_val = lambda_lookup(s_val,ti_val)
            lam_array[s_idx, ti_idx]     = lam_val
            one_minus_lam[s_idx, ti_idx] = 1.0 - lam_val


    for ti_idx, ti_val in enumerate(Xi_list):
        for ri_idx, ri_val in enumerate(Xi_list):
            # Impossible to recover before becoming infected:
            if (ti_val is not None) and (ri_val is not None) and (ri_val < ti_val):
                continue

            # For all finite s_idx . . .
            for s_idx, s_val in enumerate(Xij_list):
                if s_val is None:
                    continue

                # Check the condition ti < s < ri; 
                if (ti_val is None) or (s_val <= ti_val):
                    # no chance of first‐transmission at s (either i isn't infected yet or s is too early)
                    continue
                if (ri_val is not None) and (s_val >= ri_val):
                    # s is at or after recovery ⇒ no first‐transmission at this s
                    continue

                # Now ti < s < ri
                prod_no_transmit = 1.0
                for earlier_idx in range(s_idx):
                    s_earlier = Xij_list[earlier_idx]
                    if (s_earlier is None) or (ti_val is None) or (s_earlier <= ti_val):
                        continue
                    # s_earlier is a finite contact time, ti_val < s_earlier < s_val
                    prod_no_transmit *= one_minus_lam[earlier_idx, ti_idx]

                # The probability of first transmission exactly at s_idx is:
                S[s_idx, ti_idx, ri_idx] = lam_array[s_idx, ti_idx] * prod_no_transmit

            # For the infinite index s_idx = M-1 . . .
            inf_idx = M - 1
            prod = 1.0
            for idx, s_val in enumerate(Xij_list):
                if s_val is None:
                    continue
                if ri_val is None:
                    # “never recovered” so include only s > ti_val
                    if ti_val is None or s_val <= ti_val:
                        continue
                else:
                    # “ri finite” case so include only s ≥ ri_val
                    if s_val < ri_val:
                        continue

                prod *= one_minus_lam[idx, ti_idx]
            S[inf_idx, ti_idx, ri_idx] = prod

    return S


def compute_recovery_table(X_i_list, k=10, mu=0.57):
    """
    Build discrete recovery tables R_hat[ri_idx, ti_idx]
    using a Gamma(k, rate=mu) delay distribution.
    """
    K = len(X_i_list)
    R_hat = np.zeros((K, K), dtype=float)

    # Regularized lower‐incomplete gamma CDF: F(x) = P(X ≤ x) for X ~ Gamma(k, scale=1/mu)
    gamma_cdf = lambda x: gammainc(k, mu * x)

    for ti_idx, t_i in enumerate(X_i_list):
        if t_i is None:
            continue

        for ri_idx in range(K):
            if ri_idx < K - 1:
                lower = X_i_list[ri_idx]
            else:
                lower = X_i_list[-2]

            # Determine the “upper” bound of the interval:
            upper = X_i_list[ri_idx + 1] if (ri_idx + 1 < K) else None

            # Shift by t_i and clip lower to zero
            x_low = (lower - t_i) if (lower is not None) else np.inf
            x_low = max(x_low, 0)

            if upper is None:
                # Integrate from x_low → ∞
                if x_low <= 0:
                    R_hat[ri_idx, ti_idx] = 1.0
                else:
                    R_hat[ri_idx, ti_idx] = 1.0 - gamma_cdf(x_low)
            else:
                x_high = upper - t_i
                if x_high <= 0:
                    R_hat[ri_idx, ti_idx] = 0.0
                else:
                    # Interval [x_low, x_high):
                    R_hat[ri_idx, ti_idx] = gamma_cdf(x_high) - gamma_cdf(x_low)

    return R_hat



def build_var_to_factor_cache(G):
    """
    Fast cache builder for get_var_to_factor_messages.

    Steps:
      1) Build var_to_factors: a dict mapping each variable node → list of its factor neighbors.
      2) Gather the set of all cardinals; pre‐allocate one uniform vector per distinct cardinality.
      3) Fill var_deg1 and var_deg2 by inspecting var_to_factors.

    Returns:
      (var_deg1, var_deg2, uniform_by_card)

    - var_deg1  = [(var, fA), …] for every variable with degree 1
    - var_deg2  = [(var, fA, fB), …] for every variable with degree 2
    - uniform_by_card = {card: np.ndarray of length=card, normalized to sum=1.0}
    """
    import numpy as np

    # 1) Build var_to_factors (variable → list of factors) in one pass through G.factors
    var_to_factors = {}
    for factor in G.factors:
        for var in factor.get_scope():
            if var not in var_to_factors:
                var_to_factors[var] = [factor]
            else:
                var_to_factors[var].append(factor)

    # 2) Collect all distinct cardinalities and pre‐allocate uniform vectors
    #    We know every variable in var_to_factors appears in G.card_dict
    uniform_by_card = {}
    for var, card in G.card_dict.items():
        # card_dict stores all variables’ cardinalities (including edge‐vars and t_i, r_i, auto_i)
        if card not in uniform_by_card:
            u = np.full(card, 1.0 / card, dtype=float)
            uniform_by_card[card] = u

    # 3) Build var_deg1 and var_deg2 by checking the length of var_to_factors[var]
    var_deg1 = []
    var_deg2 = []
    for var, factor_list in var_to_factors.items():
        deg = len(factor_list)
        if deg == 1:
            fA = factor_list[0]
            var_deg1.append((var, fA))
        elif deg == 2:
            fA, fB = factor_list
            var_deg2.append((var, fA, fB))
        else:
            raise RuntimeError(f"Variable node {var!r} has unexpected degree {deg} ≥ 3")

    return (var_deg1, var_deg2, uniform_by_card)


def mf_step(pS, pI, pR, lam, mu):
        """
        One mean‐field step when λ is a scalar constant (same for every k→j).

        pS, pI, pR : numpy arrays of shape (N,)
        """
        # total infected probability across all nodes at time t
        total_infected = pI.sum()  # scalar = Σ_k pI[k]
        inf_pressure = lam * (total_infected - pI)  # vector of length N

        pS_next = pS * (1.0 - inf_pressure)
        pR_next = pR + mu * pI
        pI_next = pI + pS * inf_pressure - mu * pI

        return pS_next, pI_next, pR_next


def apply_all_observations(pS, pI, pR, test_results, τ, t_MF):
    """
    Apply a batch of observations to the MF probability arrays.

    Arguments:
      pS, pI, pR : numpy arrays of shape (N, T_max+1)
          pS[j, t] = P_MF[q_j(t)=S], etc.
      test_results : list of tuples (person_idx, s_obs, obs_time)
          s_obs ∈ {0,1,2}  (0→S, 1→I, 2→R), obs_time is an integer time index.
      τ : int
          If s_obs==1 (I), enforce I from [obs_time-τ .. obs_time].
      t_MF : int
          If s_obs==0 (S), enforce S from [obs_time-t_MF .. obs_time].

    This updates pS, pI, pR in place.
    """
    _, T_all = pS.shape
    T_max = T_all - 1

    for (j, s_obs, t_obs) in test_results:
        if s_obs == 0:
            # Observed S at time t_obs
            start = max(0, t_obs - t_MF)
            end   = t_obs
            pS[j, start:end+1] = 1.0
            pI[j, start:end+1] = 0.0
            pR[j, start:end+1] = 0.0

        elif s_obs == 1:
            # Observed I at time t_obs
            start = max(0, t_obs - τ)
            end   = t_obs
            pS[j, start:end+1] = 0.0
            pI[j, start:end+1] = 1.0
            pR[j, start:end+1] = 0.0

        elif s_obs == 2:
            # Observed R at time t_obs
            start = t_obs
            end   = T_max
            pS[j, start:end+1] = 0.0
            pI[j, start:end+1] = 0.0
            pR[j, start:end+1] = 1.0
