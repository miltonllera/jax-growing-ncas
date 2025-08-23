import numpy as np
import jax.numpy as jnp
import jax.random as jr


def init_central_seed(shape, key):
    _, H, W = shape
    x_i, y_i = W // 2, H // 2

    init = np.zeros(shape)
    init[3:, y_i - 1: y_i + 1, x_i -1: x_i + 1] = 1.0

    return init


def init_random(shape, key):
    return jr.uniform(key, shape, minval=-1, maxval=1)


def init_central_steerable_seed(shape, key):
    _, H, W = shape
    x_i, y_i = W // 2, H // 2

    init = jnp.zeros(shape)
    init = init.at[3:-1, y_i - 1: y_i + 1, x_i -1: x_i + 1].set(1.0)
    init = init.at[-1:, y_i - 1: y_i + 1, x_i -1: x_i + 1].set(
        jr.uniform(key, (3, 3), minval=0, maxval=2*np.pi)
    )

    return init

