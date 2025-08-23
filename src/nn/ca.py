import jax
import jax.numpy as jnp
import jax.random as jr
import jax.lax as lax
import equinox as eqx
from typing import Callable
from jaxtyping import Array, Float, Int


State = Float[Array, "..."]
Carry = tuple[State, jax.Array]


class CellularAutomaton(eqx.Module):
    perception_fn: Callable
    update_fn: Callable
    conditioning_fn: Callable | None

    def __init__(
            self,
            perception_fn: Callable,
            update_fn: Callable,
            conditioning_fn: Callable | None = None
    ):
        super().__init__()
        self.perception_fn = perception_fn
        self.update_fn = update_fn
        self.conditioning_fn = conditioning_fn

    def __call__(
            self,
            init_state: State,
            goal: Float[Array, "..."] | None,
            n_steps: int | tuple[int, int],
            key: jax.Array
    ):
        if goal is not None:
            assert self.conditioning_fn is not None

        carry_key, sample_key = jr.split(key)
        num_dev_steps, max_steps = self.sample_num_steps(n_steps, sample_key)

        def f(carry: Carry, step: Int) -> tuple[Carry, State]:
            cell_states, key = carry
            p_key, u_key, key = jr.split(key, 3)

            if goal is not None:
                perception_in = self.conditioning_fn(cell_states, goal)  # type: ignore
            else:
                perception_in = cell_states

            perception = self.perception_fn(perception_in, key=p_key)
            updated_states = self.update_fn(cell_states, perception, key=u_key)

            cell_states = lax.select(step >= num_dev_steps, cell_states, updated_states)
            return (cell_states, key), cell_states

        _, dev_path = lax.scan(f, (init_state, carry_key), jnp.arange(max_steps))
        dev_path = jnp.concat([init_state[None], dev_path], axis=0)

        return dev_path[num_dev_steps], dev_path

    def sample_num_steps(self, n_steps: int | tuple[int, int] , key: jax.Array):
        if isinstance(n_steps, int):
            return n_steps, n_steps
        return jr.randint(key, (1,), *n_steps).squeeze(), n_steps[1]

