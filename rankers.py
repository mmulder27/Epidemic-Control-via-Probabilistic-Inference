import numpy as np
import pandas as pd
import abc
from copy import deepcopy
from collections import defaultdict
from BP_utils import StubFactor, SimpleFactorGraph, run_loopy_bp_parallel
from potential_utils import compute_S_table, compute_recovery_table, compute_obs_likelihood_table, build_var_to_factor_cache, mf_step, apply_all_observations
from collections import deque

class BaseRanker(abc.ABC):
    def __init__(self):
        # now use deque so popleft() is O(1)
        self.contact_buffer = deque()   # holds (i, j, contact_time, weight)
        self.obs_buffer     = deque()   # holds (i, observed_state, t)
        self.window_size    = None

    def init(self, N, T, seed=None, **hyperparams):
        self.N = N
        self.T = T
        self.rng = np.random.default_rng(seed)
        self.contact_buffer.clear()
        self.obs_buffer.clear()

    def update_history(self, weighted_contacts, observations, t):
        # 1) append today’s events to the right
        for ev in weighted_contacts:
            self.contact_buffer.append(ev)
        for obs in observations:
            self.obs_buffer.append(obs)

        if self.window_size is not None:
            cutoff = t - (self.window_size - 1)

            # Pop off old contacts from the left until the contact_time ≥ cutoff
            while self.contact_buffer and self.contact_buffer[0][2] < cutoff:
                self.contact_buffer.popleft()

            # Same for observation buffer (assuming obs[2] is the time)
            while self.obs_buffer and self.obs_buffer[0][2] < cutoff:
                self.obs_buffer.popleft()

    @abc.abstractmethod
    def rank(self, t, data):
        """
        Compute a “score” for each of N individuals at time t.  Return a list
        of (i,score) for i in [0..N-1], sorted in descending order
        
        - t:    current day index
        - data: the dictionary of simulation counters.
        """
        raise NotImplementedError("Subclasses must implement .rank()")

class RandomRanker(BaseRanker):
    def init(self, N, T, seed=None):
        # Call the parent init so it sets up N, T, rng, and clears buffers
        super().init(N, T, seed)
        self._indices = np.arange(N, dtype=int)


    def rank(self, t, data):
        self.rng.shuffle(self._indices)
        return self._indices


class CTRanker(BaseRanker):
    """
    Scores individuals by counting how many times they have contacted
    someone who tested positive in the last [tau] days. 
    """

    def init(self, N, T, seed=None, tau=5, lamb=0.014, **kwargs):
        """
        Called once at the start of loop_abm.
        """
        super().init(N, T, seed)
        self.tau = tau
        self.lamb = lamb
        self.window_size = tau

    def rank(self, t, data):
        """
        For each person i, score them as the total (weighted) number of contacts
        with individuals who tested positive in [t - tau, t).  Return [(i,score)].
        """

        # If we are in the first few days before tau, we don’t have a full window:
        if t < self.tau:
            # Return a random ranking for early days
            return np.random.permutation(self.N).tolist()

        # Find “positive tests” in last tau days
        recent_positives = { i for (i, s, _) in self.obs_buffer if s == 1}

        # Count how many times each person i has “contacted” someone in recent_positives
        counts = np.zeros(self.N, dtype=float)

        for (i, j, _, w) in self.contact_buffer:
            # We only care about “i contacted j” if j ∈ recent_positives.
            if j in recent_positives:
                counts[i] += 1

        # 3) Return sorted (i, score) by descending count
        ranking = list(enumerate(counts))
        ranking.sort(key=lambda pair: pair[1], reverse=True)
        return [pair[0] for pair in ranking]
    

class BPRanker(BaseRanker):
    # ───────────────────────────── init ───────────────────────────── #
    def init(
        self, N, T, *, seed=None, window_size=21,
        bp_iters=10, conv_tol=1e-3, fnr=0.0, fpr=0.0, **kwargs
    ):
        super().init(N, T, seed)
        self.window_size = window_size
        self.bp_iters   = bp_iters
        self.conv_tol   = conv_tol
        self.gamma      = 1 / N
        self.fnr        = fnr
        self.fpr        = fpr
        self.delta      = 10          # risk-look-back window

        # persistent data structures
        self.G   = SimpleFactorGraph()          # stays alive for the whole run
        self.X_i = {i: [None] for i in range(N)}
        self.X_edges = {}                       # (a,b) -> [τ_list + None]

        self.R_i, self.pO_i, self.S_ij = {}, {}, {}
        self.person_factors = {}                # i -> factor object
        self.edge_vars      = set()             # names of s_pair variables

        # 1/ create three variable-nodes per person (card = 1 initially)
        for i in range(N):
            for v in ("t", "r", "auto"):
                name = f"{v}_{i}"
                self.G.add_node(name)
                self.G.card_dict[name] = 1

        # 2/ stub Ψᵢ factors (scope only the three personal variables for now)
        for i in range(N):
            scope = [f"t_{i}", f"r_{i}", f"auto_{i}"]
            phi   = StubFactor(scope, [1, 1, 1], i)
            self.person_factors[i] = phi
            self.G.add_factors(phi)
            for v in scope:
                self.G.add_edge(v, phi)

        self._vf_cache = build_var_to_factor_cache(self.G)  # initial cache

    # ──────────────────────── helper utilities ────────────────────── #
    def _resize_var(self, name, new_card):
        """Change a variable node’s cardinality if it grew."""
        if self.G.card_dict[name] != new_card:
            self.G.card_dict[name] = new_card
            return True
        return False

    def _touch_person_tables(self, i):
        """Recompute Rᵢ and pOᵢ after |Xᵢ| changed or new obs arrived."""
        Xi = self.X_i[i]
        self.R_i[i]  = compute_recovery_table(Xi)
        self.pO_i[i] = compute_obs_likelihood_table(
            i, Xi, self.obs_buffer, self.fnr, self.fpr
        )

    def _ensure_edge_var(self, a, b, τ_list):
        """Create or resize s_pair_{a,b} and its S_ij tables."""
        var = f"s_pair_{a}_{b}"
        card = len(τ_list) ** 2
        if var not in self.edge_vars:
            # new edge
            self.G.add_node(var)
            self.edge_vars.add(var)
            self.G.card_dict[var] = card
            # hook into the two Ψ factors
            for i in (a, b):
                phi = self.person_factors[i]
                phi._scope.append(var)
                phi._card.append(card)
                self.G.add_edge(var, phi)
        else:
            self._resize_var(var, card)

        # refresh S tables for both directions
        self.S_ij[(a, b)] = compute_S_table(a, b, τ_list, self.X_i[a])
        self.S_ij[(b, a)] = compute_S_table(b, a, τ_list, self.X_i[b])

    # ───────────────────────── update_history ─────────────────────── #
    def update_history(self, weighted_contacts, observations, t):
        super().update_history(weighted_contacts, observations, t)  # prunes buffers

        # ---- rebuild timestamp sets ------------------------------------------------
        new_edge_times = defaultdict(set)
        for i, j, τ, w in self.contact_buffer:
            a, b = (i, j) if i < j else (j, i)
            new_edge_times[(a, b)].add(τ)

        # add trailing “None” sentinel once
        new_X_edges = {
            edge: sorted(times) + [None]
            for edge, times in new_edge_times.items()
        }

        # ---- detect per-person timestamp changes -----------------------------------
        touched_people = set()
        new_X_i = {p: set() for p in range(self.N)}
        for (a, b), τs in new_X_edges.items():
            for τ in τs[:-1]:
                new_X_i[a].add(τ)
                new_X_i[b].add(τ)
        new_X_i = {p: sorted(ts) + [None] for p, ts in new_X_i.items()}

        # ---- update Xᵢ and person variables ----------------------------------------
        for i in range(self.N):
            if new_X_i[i] != self.X_i[i]:
                self.X_i[i] = new_X_i[i]
                card = len(self.X_i[i])
                for v in ("t", "r", "auto"):
                    self._resize_var(f"{v}_{i}", card)
                self._touch_person_tables(i)
                touched_people.add(i)

        # ---- update / add s_pair variables -----------------------------------------
        # removed edges first
        removed_edges = set(self.X_edges) - set(new_X_edges)
        for (a, b) in removed_edges:
            var = f"s_pair_{a}_{b}"
            # detach from Ψ factors
            for i in (a, b):
                phi = self.person_factors[i]
                if var in phi._scope:
                    idx = phi._scope.index(var)
                    phi._scope.pop(idx)
                    phi._card.pop(idx)
                    self.G.remove_edge(var, phi)
            self.G.remove_node(var)
            self.edge_vars.discard(var)
            self.S_ij.pop((a, b), None)
            self.S_ij.pop((b, a), None)

        # new / resized edges
        for (a, b), τ_list in new_X_edges.items():
            self._ensure_edge_var(a, b, τ_list)
            if (a, b) not in self.X_edges or τ_list != self.X_edges[(a, b)]:
                touched_people.update((a, b))
        self.X_edges = new_X_edges  # now official

        # ---- refresh caches if structure changed -----------------------------------
        if touched_people or removed_edges:
            self._vf_cache = build_var_to_factor_cache(self.G)


    def rank(self, t, data):
        
        print(f"time: {t}")
        #import time
        #t0 = time.perf_counter()
        # 1) Run BP and get final marginals
        nodeMargs = run_loopy_bp_parallel(self.G, self.bp_iters, self.conv_tol, self.X_i, self.X_edges, self.R_i, self.pO_i, self.S_ij, self.person_factors, self.gamma, self._vf_cache)
        final_beliefs = nodeMargs[-1]
        #print("elapsed time for loopy BP all iterations", time.perf_counter() - t0, "seconds")

        # 2) Compute each node's risk = sum of b_i(t_i) over t_i ∈ [t - δrank, t]
        risk_scores = np.zeros(self.N, dtype=float)
        for i in range(self.N):
            # full marginal vector over the t_i grid (last entry corresponds to None)
            marginal = final_beliefs[f"t_{i}"].flatten()  # shape = (K+1,)

            # drop the final “None” entry from both times and marginal
            times = np.array(self.X_i[i][:-1])   # shape = (K,)
            probs = marginal[:-1]                # shape = (K,)

            # select indices where t_i is in [t - δrank, t]
            mask = (times >= (t - self.delta)) & (times <= t)
            risk_scores[i] = probs[mask].sum()

        # 3) Sort nodes descending by risk score
        ranking = sorted(range(self.N), key=lambda idx: risk_scores[idx], reverse=True)
        return ranking

class MFRanker(BaseRanker):
    """
    Simple scalar-λ implementation with one pass per day.
    """

    # ─────────────── public API ────────────────────────────────────────────
    def init(
        self,
        N,
        T,
        *,
        seed=None,
        tau=5,          # τ  (I enforced on [t_obs-τ, t_obs])
        t_MF=10,        # integration window for S constraints
        mu=1 / 12,      # recovery rate
        lam=0.02,       # scalar λ
        **kwargs,
    ):
        super().init(N, T, seed)
        self.N, self.T = N, T
        self.tau, self.t_MF, self.mu, self.lam = tau, t_MF, mu, lam

        self.pS = np.ones((N, T))
        self.pI = np.zeros_like(self.pS)
        self.pR = np.zeros_like(self.pS)
        self.last_t = 0

    def rank(self, t, data=None):
        self._simulate_until(t)
        return np.argsort(self.pI[:, t])[::-1]

    def _simulate_until(self, t_target):
        if t_target <= self.last_t:
            return
        for t in range(self.last_t, t_target):
            self._enforce_constraints(t)   # before MF step
            self._mf_step(t)               # write column t+1
            self._renorm(t + 1)
        self._enforce_constraints(t_target)  # final slice
        self.last_t = t_target

    def _mf_step(self, t):
        pS, pI, pR = self.pS[:, t], self.pI[:, t], self.pR[:, t]
        total_I = pI.sum()
        infected_fraction = total_I / self.N                # <= 1
        inf_press = np.clip(self.lam * (infected_fraction - pI), 0.0, 1.0)


        self.pS[:, t+1] = pS * (1.0 - inf_press)
        self.pR[:, t+1] = pR + self.mu * pI
        self.pI[:, t+1] = pI + pS * inf_press - self.mu * pI

    # constraints & state forcing combined
    def _enforce_constraints(self, t_day):
        for j, s, t_obs in self.obs_buffer:
            if s == 0 and (t_obs - self.t_MF <= t_day <= t_obs):
                self.pS[j, t_day], self.pI[j, t_day], self.pR[j, t_day] = 1.0, 0.0, 0.0
            elif s == 1 and (t_obs - self.tau <= t_day <= t_obs):
                self.pS[j, t_day], self.pI[j, t_day], self.pR[j, t_day] = 0.0, 1.0, 0.0
            elif s == 2 and (t_day >= t_obs):
                self.pS[j, t_day], self.pI[j, t_day], self.pR[j, t_day] = 0.0, 0.0, 1.0

    def _renorm(self, t_idx):
        row_sum = self.pS[:, t_idx] + self.pI[:, t_idx] + self.pR[:, t_idx]
        valid = row_sum > 0
        self.pS[valid, t_idx] /= row_sum[valid]
        self.pI[valid, t_idx] /= row_sum[valid]
        self.pR[valid, t_idx] /= row_sum[valid]
