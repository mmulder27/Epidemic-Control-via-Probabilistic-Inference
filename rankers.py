import numpy as np
import pandas as pd

class RandomRanker:

    def init(self, N, T, seed=None):
        self.rng = np.random.default_rng(seed)
        self.N = N
        self.T = T

    def rank(self, t, weighted_contacts, observations, data):
        return [(i, self.rng.random()) for i in range(self.N)]


class CTRanker:

    def init(self, N, T, seed=None):
        self.rng = np.random.default_rng(seed)
        self.tau = 5
        self.lamb = 0.014
        self.transmissions = []
        self.observations = []
        self.N = N
        self.T = T
    

    def rank(self, t, weighted_contacts, observations, data):
        """
        Scores individuals based on how often they contacted recently positive cases
        in the time window [t - tau, t).

        Inputs:
            - t: current time step
            - weighted_contacts: list of (i, j, time, weight)
            - observations: list of (i, s, t_test)

        Returns:
            - List of (i, score) pairs, sorted by score descending
        """

        self.observations += [
            dict(i=i, s=s, t_test=t_test) for i, s, t_test in observations
        ]

        if t < self.tau:
            scores = self.rng.random(self.N)
        else:
            # Identify individuals who tested positive in [t - tau, t)
            last_tested = {
                obs["i"] for obs in self.observations
                if obs["s"] == 1 and (t - self.tau <= obs["t_test"] < t)
            }

            # Count how often each person contacted a recently tested-positive individual
            contact_rows = [
                (i, j) for i, j, _, weight in weighted_contacts
                if j in last_tested and weight > 0
            ]

            counts = pd.Series(0, index=np.arange(self.N), dtype=int)
            if contact_rows:
                df = pd.DataFrame(contact_rows, columns=["i", "j"])
                counts.update(df["i"].value_counts())

            scores = counts.values

        return sorted(enumerate(scores), key=lambda t: t[1], reverse=True)
