import numpy as np


class RunningNorm:
    """
    Standard dynamic data normalization.
    """

    def __init__(self, shape=1, eps=1e-8):
        self.mean = np.zeros(shape)
        self.var = np.ones(shape)
        self.count = eps

    def update(self, x):
        x = np.array(x)
        if x.ndim == 0:
            x = x.reshape(1)

        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0] if x.ndim > 0 else 1

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / total_count

        self.mean = new_mean
        self.var = M2 / total_count
        self.count = total_count

    def normalize(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)

    def get_state(self):
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def set_state(self, state):
        self.mean = np.array(state["mean"])
        self.var = np.array(state["var"])
        self.count = state["count"]
