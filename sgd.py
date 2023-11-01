import functools
import numpy as np
from autograd import grad
from tqdm import tqdm
from types import SimpleNamespace


class ValueTracker:
    def __init__(self, f):
        functools.update_wrapper(self, f)
        self.f = f
        self.f_history = []

    def __call__(self, *args, **kwargs):
        values = self.f(*args, **kwargs)
        self.f_history.append(values[1])
        return values

    def history(self):
        return np.array(self.f_history)

    def restart(self):
        self.f_history = []


def batch_index_generator(n_samples, batch_size):
    rg = np.random.default_rng()
    batch = rg.permutation(n_samples)
    start, stop = 0, batch_size
    while stop < n_samples:
        yield batch[start:stop]
        start += batch_size
        stop += batch_size
    stop = min(stop, n_samples)
    yield batch[start:stop]


@ValueTracker
def step(fct, fct_grad, z, alpha):
    z_prime = z - alpha * fct_grad(z)
    f_prime = fct(z_prime)
    return z_prime, f_prime


def sgd(fct, z, samples, batch_size, alpha_range, max_epochs, min_step=1.e-4, **kwargs):
    n_samples = len(samples.y)
    assert batch_size <= n_samples, 'mini-batch size cannot exceed batch size'
    z_shape = z.shape

    def unravel(arg):
        return arg.reshape(z_shape)

    z = z.ravel()

    step.restart()

    alpha = alpha_range[0]
    for _ in tqdm(range(max_epochs)):
        z_before_epoch = z
        for batch_indices in batch_index_generator(n_samples, batch_size):
            batch = SimpleNamespace(x=samples.x[batch_indices], y=samples.y[batch_indices])

            def fct_batch(v):
                return fct(unravel(v), batch, **kwargs)

            z, f_batch = step(fct_batch,  grad(fct_batch), z, alpha)
        if np.linalg.norm(z - z_before_epoch) < min_step * np.linalg.norm(z):
            break
        alpha = alpha_range[1] if alpha <= alpha_range[1] else alpha * 0.99

    return unravel(z)
