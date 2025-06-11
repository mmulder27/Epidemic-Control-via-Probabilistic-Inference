import numpy as np
import networkx as nx
import abc
from collections import defaultdict
from BP_utils import FastDiscreteFactor, SimpleFactorGraph, run_loopy_bp_parallel
from collections import deque
from scipy.sparse import csr_matrix


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
        # append today’s events
        self.contact_buffer.extend(weighted_contacts)
        self.obs_buffer.extend(observations)
        if self.window_size is None:
            return  # nothing to trim
        cutoff = t - self.window_size
        while self.contact_buffer and self.contact_buffer[0][2] < cutoff:
            self.contact_buffer.popleft()
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


    def rank(self, t, weighted_contacts, daily_obs, data=None):
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

    def rank(self, t, weighted_contacts, daily_obs, data):
        """
        For each person i, score them as the total (weighted) number of contacts
        with individuals who tested positive in [t - tau, t).  Return [(i,score)].
        """
        self.update_history(weighted_contacts, daily_obs, t)
        # If we are in the first few days before tau, we don’t have a full window:
        if t < self.tau:
            # Return a random ranking for early days
            return np.random.permutation(self.N).tolist()

        # Find “positive tests” in last tau days
        recent_positives = { i for (i, s, _) in self.obs_buffer if s == 1}

        # Count how many times each person i has “contacted” someone in recent_positives
        counts = np.zeros(self.N, dtype=float)

        for (i, j, _, w) in self.contact_buffer:
            # We only care about “i contacted j” if j in recent_positives.
            if j in recent_positives:
                counts[i] += 1

        # 3) Return sorted (i, score) by descending count
        ranking = list(enumerate(counts))
        ranking.sort(key=lambda pair: pair[1], reverse=True)
        return [pair[0] for pair in ranking]


def mf_step_linear(P, contacts, lam, mu, N):
        """
        One-step linearised mean-field update:
        - contacts: list[(i, j, weight)] with i = receiver, j = donor
        - P: np.ndarray (N, 3) current slice (S, I, R)
        Returns new array (N, 3) for P(t+1)
        """
        pS, pI, pR = P[:, 0], P[:, 1], P[:, 2]
        inf_press = np.zeros(N)
        for i, j, w in contacts:
            inf_press[i] += lam * w * pI[j]
        np.clip(inf_press, 0.0, 1.0, out=inf_press)

        S_to_I = pS * inf_press
        I_to_R = pI * mu

        P_next = np.zeros_like(P)
        P_next[:, 0] = pS - S_to_I
        P_next[:, 1] = pI + S_to_I - I_to_R
        P_next[:, 2] = pR + I_to_R
        return P_next

class SMFRanker(BaseRanker):
    """Linear mean‑field ranker that needs *only* NumPy.

    Parameters
    ----------
    tau   : int   – delay between infection and positive test (days)
    delta : int   – size of the back‑tracking window (days)
    mu    : float – daily recovery probability
    lam   : float – contact‑to‑transmission scaling (λ)
    seed  : int or None – RNG seed for tie‑breaking
    """
    def __init__(self, tau=5, delta=10, mu=1/12, lam=0.02, seed=None):
        super().__init__()
        self.tau   = int(tau)
        self.delta = int(delta)
        self.mu    = float(mu)
        self.lam   = float(lam)
        self.window_size = self.delta  # used by BaseRanker trimming
        self.rng   = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # mandatory API
    # ------------------------------------------------------------------
    def init(self, N, T, **kwargs):
        self.N = int(N)
        self.T = int(T)
        return True


    def rank(self, t_day, weighted_contacts, observations, data=None):
        """Return indices sorted by decreasing P(I) at *t_day*.

        The method is self​-contained: no `csr_matrix`, no helper calls.
        Input formats are the same as in `update_history`:
        weighted_contacts : list[(i, j, t_day, weight)]
        observations      : list[(i, s, t_day)]
        """
        self.update_history(weighted_contacts, observations, t_day)

        start_day = max(0, t_day - self.delta)
        contacts_by_day = defaultdict(list)
        for i, j, t_evt, w in self.contact_buffer:
            contacts_by_day[t_evt].append((i, j, w))
        obs_by_day = defaultdict(list)
        for i, s, t_obs in self.obs_buffer:
            obs_by_day[t_obs].append((i, s))

        P = np.zeros((self.N, 3), dtype=float)
        P[:, 0] = 1.0
        P_history_I = None

        for day in range(start_day, t_day + 1):
            if day in obs_by_day:
                for i, s in obs_by_day[day]:
                    if s == 0:
                        P[i] = [1., 0., 0.]
                    elif s == 1:
                        t_I = day - self.tau
                        if t_I <= day:
                            P[i] = [0., 1., 0.]
                    elif s == 2:
                        P[i] = [0., 0., 1.]

            if day == t_day:
                P_history_I = P[:, 1].copy()
                break

            contacts = contacts_by_day.get(day, [])
            P = mf_step_linear(P, contacts, self.lam, self.mu, self.N)

        scores = P_history_I if P_history_I is not None else self.rng.random(self.N)
        return np.argsort(scores)[::-1]



# ────────────────────────────────────────────────────────────────────
#  “Robust” naïve-mean-field (energy form) update
#     • protects against duplicated edges
#     • caps λ·w at λ_max  (default 0.2)
#     • works entirely in log-space, avoids underflow
#     • uses SciPy CSR for O(E) time
# ────────────────────────────────────────────────────────────────────
import numpy as np
from collections import deque, defaultdict
from scipy.sparse import csr_matrix
from math import gamma, exp

# ───────────────────────── Gamma-based per-contact λ ─────────────────────
def lambda_lookup(sij, ti, alpha=0.25, k=5.76, μ=0.96):
    """
    λ(sij | t_i)  = alpha · GammaPDF(Δt)      with  Δt = sij − t_i   (days)
    The PDF is   Γ(k,θ)  where θ = μ / k   so mean = μ.
    Result is capped strictly below 1 to stay a valid probability.
    """
    if ti is None:                 # donor never infected in the window
        return 0.0
    diff = sij - ti
    if diff <= 0:
        return 0.0
    θ   = μ / k
    pdf = (diff**(k-1) * exp(-diff/θ)) / (gamma(k) * θ**k)
    return min(alpha * pdf, 1 - 1e-12)

# ───────────────────────── contacts → CSR helper ────────────────────────
def contacts_to_csr(N,
                    contacts_this_day,
                    day,
                    lam,
                    λ_max=0.50):
    """
    Build Λ_t  (receiver i, donor j)  for one simulation day.

    Parameters
    ----------
    N              : int
        Population size.
    contacts_iter  : iterable of (i_receiver, j_donor, s_day, weight)
        *s_day* is the (integer) day index when this contact occurred.
        *weight* aggregates how many raw calls you folded into this record.
    t_inf_est      : 1-D array-like length N (or dict)
        Best current estimate of each donor’s infection time t_i.
        Use None (or -np.inf) if donor j is still believed susceptible.
    λ_max          : float, optional
        Hard ceiling for per-edge probability after weighting.

    Returns
    -------
    csr_matrix  shape (N,N) with Λ_t[i,j] = λ_{j→i}(t)
    """
    # ---- deduplicate edges ------------------------------------------------
    best_λ = defaultdict(float)                 # (i,j) -> λ    (max over w)
    for i, j, w in contacts_this_day:
        #λ_ij = lambda_lookup(day, t_inf_est[j]) * w
        λ_ij = lam * w
        #λ_ij = min(λ_ij, λ_max)                 # enforce global cap
        if λ_ij > best_λ[(i, j)]:               # keep the strongest edge
            best_λ[(i, j)] = λ_ij

    # ---- build CSR arrays -------------------------------------------------
    if not best_λ:                              # no contacts today
        return csr_matrix((N, N), dtype=float)

    rows, cols, data = zip(*((i, j, λ) for (i, j), λ in best_λ.items()))
    return csr_matrix((data, (rows, cols)), shape=(N, N))


# ---------- one-day step -------------------------------------------------
def mf_step_energy_safe(P, Λ_t, mu, λ_max=0.20, min_log=-700.0):
    """
    Parameters
    ----------
    P      : (N,3) array  -- today’s [S,I,R] probabilities
    Λ_t    : csr_matrix   -- Λ_t[i, j] = *uncapped* (λ_base * weight) for day t
    mu     : float        -- daily recovery probability
    λ_max  : float        -- hard ceiling for λ after weighting
    min_log: float        -- clamp to avoid exp(–inf) underflow (~-745)

    Returns
    -------
    P_next : (N,3) array for day t+1 (energy-form naïve MF)
    """
    pS, pI, pR = P.T                        # views, no copy

    # -------- prepare λ data -------------------------------------------------
    # cap every edge at λ_max and convert to log(1-λ)
    capped = np.minimum(Λ_t.data, λ_max)
    log1m  = np.log1p(-capped)              # safe for λ∈[0,1)

    # multiply by donor’s P_I  (vectorised through CSR “indices”)
    weighted_data = log1m * pI[Λ_t.indices]

    # build a CSR matrix with the same (row, col) positions but new data
    log_mat = csr_matrix((weighted_data, Λ_t.indices, Λ_t.indptr),
                         shape=Λ_t.shape)

    # Σ_k P_I^k log(1-λ_{k→i})
    log_prod = np.asarray(log_mat.sum(axis=1)).ravel()
    log_prod = np.maximum(log_prod, min_log)   # clamp for exp

    # -------- compartment updates -------------------------------------------
    pS_next = pS * np.exp(log_prod)
    pR_next = pR + mu * pI
    pI_next = 1.0 - pS_next - pR_next

    return np.column_stack((pS_next, pI_next, pR_next))

# ───────────────────────── ranker skeleton ───────────────────────────────
class MFRanker(BaseRanker):
    """
    Non-linear naïve-MF ranker with the safe energy-form update above.
    Only dependency beyond NumPy is SciPy CSR (for speed).
    """
    def __init__(self, tau=5, delta=10, mu=1/12, lam=0.02, λ_max=0.50, seed=None, fnr=0.1, fpr=0.1):
        super().__init__()
        self.tau, self.delta, self.mu = int(tau), int(delta), float(mu)
        self.lam, self.λ_max = float(lam), float(λ_max)
        self.window_size = self.delta
        self.rng = np.random.default_rng(seed)
        self.fnr = fnr
        self.fpr = fpr

    def init(self, N, T, **kw):
        self.N, self.T = int(N), int(T)
        return True

    # ------------ private: apply obs ---------------------------------------
    def _apply_obs(self, P, obs):
        for i, s in obs:
            if s == 0:                 # negative
                P[i] = [1., 0., 0.]
            elif s == 1:               # positive
                P[i] = [0., 1., 0.]
            else:                      # recovered
                P[i] = [0., 0., 1.]

    def _apply_obs_soft(self, P, obs_list):
        neg_like = np.array([1-self.fnr, self.fnr , 1.0])
        pos_like = np.array([self.fpr, 1-self.fpr, 0.0])

        for i, z in obs_list:           # z = 0 / 1 / 2
            if z == 0:                  # negative
                P[i] = neg_like
            elif z == 1:                # positive
                P[i] = pos_like
            else:
                P[i] = [0.,0., 1.]
            tot = P[i].sum()
            if tot == 0.0:
                P[i] = [0.,0.,0.]
            else:
                P[i] /= tot         # renormalise rows


    # ------------ main entry -----------------------------------------------
    def rank(self, t_day, weighted_contacts, observations, data=None):
        self.update_history(weighted_contacts, observations, t_day)

        start = max(0, t_day - self.delta)

        # group history by day
        contacts_by_day   = defaultdict(list)
        for i, j, d, w in self.contact_buffer:
            contacts_by_day[d].append((i, j, w))
        obs_by_day = defaultdict(list)
        for i, s, d in self.obs_buffer:
            obs_by_day[d].append((i, s))

        # initialize slice
        P = np.zeros((self.N, 3)); P[:, 0] = 1.
        #t_inf_est = np.full(self.N, None, dtype=object)

        for day in range(start, t_day + 1):

            # impose today’s obs *before* contacts act
            if day in obs_by_day:
                self._apply_obs(P, obs_by_day[day])

            #thresh = day/self.T
            # 2) record first crossing of 0.5   (do it *after* obs are applied)
            #newly_crossed = (P[:, 1] > 0.8) & (t_inf_est == None)

            #t_inf_est[newly_crossed] = day          # day is absolute calendar day
            if day == t_day:           # stop at target morning
                break

            # build Λ_t CSR for this day
            Λ_t = contacts_to_csr(self.N,
                                  contacts_by_day.get(day, []),
                                  day,
                                  lam=self.lam)

            # propagate
            P = mf_step_energy_safe(P, Λ_t, self.mu, λ_max=self.λ_max)

        return np.argsort(P[:, 1])[::-1]
    

from math import gamma
from scipy.special import gammainc      # regularised lower Γ
import numpy as np

def recovery_likelihood(delta_t, k=10, mu=0.57):
    """Return [1, 1-h, h] given Δt = days since infection."""
    if delta_t < 0:                      # not infected yet
        return np.array([1., 0., 0.])    # S=1, I=R=0  (you may prefer [1,1,1])
    θ  = mu / k
    F0 = gammainc(k,  delta_t      / θ)
    F1 = gammainc(k, (delta_t + 1) / θ)
    S  = 1.0 - F0
    if S <= 1e-12:                       # almost sure recovered already
        h = 1.0
    else:
        h = (F1 - F0) / S               # hazard for the coming day
    return np.array([1.0, 1.0 - h, h])

# ---------------------------------------------------------------------
#  Belief–Propagation ranker with bounded sliding window
#     • one pair-factor per unordered pair per day
#     • observation factors handle FP / FN
#     • outdated factors removed after <window_size> days
# ---------------------------------------------------------------------

from collections import defaultdict
import numpy as np

class BPRanker(BaseRanker):
    # --------------- configure static hyper-parameters ---------------
    def __init__(self,
                 tau: int  = 5,
                 window_size: int = 10,
                 bp_iters: int = 10,
                 conv_tol: float = 1e-3,
                 fnr: float = 0.1,
                 fpr: float = 0.1,
                 mu=1/12,
                 lam=0.02,
                 λ_max: float = 0.2,
                 seed=None):
        super().__init__()
        self.tau         = tau                 # delay pos-test – infection
        self.window_size = window_size         # sliding window length (days)
        self.bp_iters    = bp_iters            # max loopy-BP sweeps
        self.conv_tol    = conv_tol            # convergence threshold
        self.fnr, self.fpr = fnr, fpr          # test errors
        self.lam         = lam
        self.mu          = mu
        self.λ_max       = λ_max               # edge-probability ceiling
        self.rng         = np.random.default_rng(seed)

    # ------------------------ graph set-up ---------------------------
    def init(self, N, T, **kw):
        self.N, self.T = int(N), int(T)

        # self.G = SimpleFactorGraph()
        # self.G.add_nodes_from([f"q{i}" for i in range(self.N)])
        prior = np.array([1.0, 1.0, 1.0])     # [S, I, R] for every person
        self.beliefs = np.tile(prior, (self.N, 1))
        # Neutral unary factors (will be overwritten when observations arrive)
        self.unary_factors = {}
        # for i in range(self.N):
        #     phi = FastDiscreteFactor([f"q{i}"], [3], np.array([1.0,0.01,0.01]))
        #     self.G.add_factors(phi)
        #     self.G.add_edge(f"q{i}", phi)
        #     self.obs_factor[i] = phi

        # Pair factors organised by day:  {day: {(i,j): phi}}
        # self.edge_factors = defaultdict(dict)

        # first-infection estimate per person (None = never crossed 0.5 yet)
        #self.t_inf_est = np.full(self.N, None, dtype=object)

        # ------------------------ update_history -------------------------
    def update_history(self, weighted_contacts, observations, t_day):
        """
        Rebuild a fresh factor graph for the current day.
        Nothing from previous days is kept inside the graph itself;
        any temporal logic (e.g. recovery prior) is injected through the
        unary factors below.
        """
        super().update_history(weighted_contacts, observations, t_day)

        # 0. start a brand-new graph
        self.G = SimpleFactorGraph()
        self.G.add_nodes_from([f"q{i}" for i in range(self.N)])

        # 1. initial neutral unary factors for every person
        self.unary_factors = {}
        for i in range(self.N):
            phi = FastDiscreteFactor([f"q{i}"], [3], np.array([1.0, 1.0, 1.0]))
            self.G.add_factors(phi)
            self.G.add_edge(f"q{i}", phi)
            self.unary_factors[i] = phi

        # 2. inject today’s test results
        for i, s, _ in self.obs_buffer:
            i = int(i)
            if s == 0:   like = np.array([1 - self.fpr, self.fnr, 1.0])   # negative test
            elif s == 1: like = np.array([self.fpr , 1 - self.fnr, 0.0])   # positive test
            else:        like = np.array([0.0, 0.0, 1.0])                  # “recovered”
            self.unary_factors[i].values[:] *= like

        # 3. apply recovery prior for those estimated infected earlier
        like = np.array([1.0,1.0 - self.mu,1.0 + self.mu])

        # Boolean mask: who was infected yesterday?
        infected_mask = self.beliefs[:, 1] > 0.5

        for i in np.flatnonzero(infected_mask):
            phi = self.unary_factors[i]

            # multiply in the recovery likelihood
            phi.values[:] *= like

            # renormalise
            tot = phi.values.sum()
            if tot == 0.0:                      # all-zero guard (should not happen)
                phi.values[:] = 1.0 / 3
            else:
                phi.values[:] /= tot
            

        # 4. pair factors for *today’s* contacts
        #print(self.contact_buffer[-1])
        Λ_t = contacts_to_csr(
            self.N,
            [(i, j, w) for i, j, _, w in self.contact_buffer],
            day=t_day,
            lam=self.lam,
            λ_max=self.λ_max
        )

        rows, cols = Λ_t.nonzero()
        seen = set()
        for i, j in zip(rows, cols):
            if (i, j) in seen or (j, i) in seen:       # avoid duplicates
                continue
            seen.add((i, j))

            λ_ji = float(Λ_t[i, j])
            λ_ij = float(Λ_t[j, i])

            table = np.array([
                1.0, 1.0 - λ_ji, 1.0,   # x_i = S
                1.0 - λ_ij, 1.0, 1.0,   # x_i = I
                1.0, 1.0, 1.0    # x_i = R
            ])

            phi = FastDiscreteFactor([f"q{i}", f"q{j}"], [3, 3], table)
            self.G.add_factors(phi)
            self.G.add_edges_from([(f"q{i}", phi), (f"q{j}", phi)])


    # ----------------------------- rank ------------------------------
    def rank(self, t_day, weighted_contacts, observations, data=None):
        # push today’s events into the graph
        self.update_history(weighted_contacts, observations, t_day)

        import time
        start = time.perf_counter()
        # run loopy BP
        beliefs = run_loopy_bp_parallel(
            self.G,
            max_iters=self.bp_iters,
            conv_tol=self.conv_tol
        )[-1]                                  # last iteration dict
        end = time.perf_counter()
        print("Duration: ", end - start)
        #print(beliefs)

        self.beliefs = np.vstack([beliefs[f"q{i}"].ravel() for i in range(self.N)])    # → shape (N,3)
        # extract P_I for ranking
        pI = np.array([beliefs[f"q{i}"][1] for i in range(self.N)]).ravel()

        # update first-infection estimate
        #crossed = (pI > 0.5) & (self.t_inf_est == None)
        #print(len(crossed))
        #self.t_inf_est[crossed] = t_day

        return np.argsort(pI)[::-1]
