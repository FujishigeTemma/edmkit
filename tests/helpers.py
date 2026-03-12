import numpy as np


def make_seeded_sampler(seed: int):
    """テスト独立な RNG を持つサンプラーを生成。"""
    rng = np.random.default_rng(seed)

    def sampler(pool, size):
        return rng.choice(pool, size=size, replace=True)

    return sampler
