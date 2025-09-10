"""
Ray-актор для хранения рейтингов TrueSkill оппонентов и выборки соперника с весами.
"""

import numpy as np
import trueskill as ts
import ray

TS_ENV = ts.TrueSkill(draw_probability=0.0)

@ray.remote
class LeagueState:
    def __init__(self, opponent_ids):
        self.opponent_ids = list(opponent_ids)
        self.ratings = {"main": TS_ENV.create_rating()}
        for pid in self.opponent_ids:
            self.ratings[pid] = TS_ENV.create_rating()

    def get_opponent_weighted(self, top_k: int = 3) -> str:
        scores = np.array([self._skill(self.ratings[p]) for p in self.opponent_ids], dtype=np.float32)
        if top_k < len(self.opponent_ids):
            idxs = np.argsort(scores)[-top_k:]
            scores = scores[idxs]
            ids = [self.opponent_ids[i] for i in idxs]
        else:
            ids = self.opponent_ids
        w = np.exp(scores - scores.max()); w /= w.sum() + 1e-8
        return str(np.random.choice(ids, p=w))

    def update_pair_result(self, wins_main: int, wins_opp: int, opp_id: str):
        r_main = self.ratings["main"]; r_opp = self.ratings[opp_id]
        for _ in range(wins_main):
            r_main, r_opp = ts.rate_1vs1(r_main, r_opp, env=TS_ENV)
        for _ in range(wins_opp):
            r_opp, r_main = ts.rate_1vs1(r_opp, r_main, env=TS_ENV)
        self.ratings["main"], self.ratings[opp_id] = r_main, r_opp

    def clone_main_into(self, opp_id: str):
        # Сброс рейтинга у склонированного соперника
        self.ratings[opp_id] = TS_ENV.create_rating()

    def get_all_scores(self):
        return {k: (v.mu, v.sigma) for k, v in self.ratings.items()}

    @staticmethod
    def _skill(r: ts.Rating) -> float:
        return r.mu - 3.0 * r.sigma
