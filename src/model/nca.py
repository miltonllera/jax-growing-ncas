from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import equinox.nn as nn
from typing import Literal
from jaxtyping import Float, Array

from src.nn.ca import CellularAutomaton
from src.nn.perception import sobel_perception, laplace_perception
from src.nn.update import GrowingUpdate, SimpleUpdate
from src.nn.morphogens import (
    gaussian_field,
    directional_fields,
    sinusoidal_fields,
    mix_fields,
)


#----------------------------------------- GrowingNCA --------------------------------------------

class GrowingNCA(eqx.Module):
    state_size: int
    hidden_size: int
    ca: CellularAutomaton
    num_dev_steps: tuple[int, int]

    def __init__(
        self,
        hidden_size = 12,
        perception_type: Literal['sobel', 'laplace', 'sobel-with-laplace', 'learned'] = 'sobel',
        update_width = 128,
        update_depth = 1,
        update_prob = 0.5,
        alive_threshold = 0.1,
        alive_index = 3,
        num_dev_steps = (48, 96),
        bound_updates = False,
        *,
        key
    ) -> None:
        super().__init__()

        state_size = hidden_size + 4
        conv_key, update_key = jr.split(key)

        # Perception function
        if perception_type == 'laplace':
            perception_fn = laplace_perception
        elif perception_type == 'sobel':
            perception_fn = partial(sobel_perception, use_laplace=False)
        elif perception_type == 'sobel-with-laplace':
            perception_fn = partial(sobel_perception, use_laplace=True)
        else:
            perception_fn = nn.Conv2d(
                in_channels=state_size,
                out_channels=state_size,
                kernel_size=3,
                padding=1,
                padding_mode='wrap',
                groups=state_size,
                key=conv_key
            )

        # Update function
        dummy_state = jnp.zeros((state_size, 8, 8))
        perception_out_size = perception_fn(dummy_state, key=conv_key).shape[0]

        layer_input_size, layers = perception_out_size, []
        for _ in range(update_depth):
            update_depth, conv_key = jr.split(update_key)
            layers.extend([
                nn.Conv2d(layer_input_size, update_width, kernel_size=1, key=conv_key),
                nn.Lambda(jax.nn.relu),
            ])
            layer_input_size = update_width

        layers.append(nn.Conv2d(layer_input_size, state_size, kernel_size=1, key=update_key))
        if bound_updates:
            layers.append(nn.Lambda(jax.nn.tanh))

        update_fn = GrowingUpdate(
            nn.Sequential(layers),
            alive_threshold,
            alive_index,
            update_prob
        )

        # used to mask goals in growth-baed goal-guided models, otherwise it will be compiled away
        def mask_goal(cell_states, goal):
            alive = cell_states[alive_index:alive_index+1] > alive_threshold
            return cell_states + goal[..., None, None] * alive

        self.state_size = state_size
        self.hidden_size = hidden_size
        self.ca = CellularAutomaton(perception_fn, update_fn, mask_goal)
        self.num_dev_steps = num_dev_steps

    def __call__(
        self,
        init_state: Float[Array, "C H W"],
        goal: Float[Array, "..."] | None = None,
        steps=None,
        *,
        key: jax.Array,
    ):
        if steps is None:
            steps = self.num_dev_steps
        cell_states, dev_path = self.ca(init_state, goal, steps, key=key)
        return cell_states[:4], dev_path


#----------------------------------------- NoiseNCA ----------------------------------------------

class NoiseNCA(eqx.Module):
    state_size: int
    hidden_size: int
    ca: CellularAutomaton
    num_dev_steps: tuple[int, int]

    def __init__(
        self,
        hidden_size = 12,
        perception_type: Literal['sobel', 'sobel-with-laplace', 'laplace', 'learned'] = 'sobel',
        update_width = 128,
        update_depth = 1,
        update_prob = 0.5,
        num_dev_steps = (48, 96),
        bound_updates = False,
        *,
        key
    ) -> None:
        super().__init__()

        state_size = hidden_size + 4
        conv_key, update_key = jr.split(key)

        # Perception function
        if perception_type == 'sobel':
            perception_fn = partial(sobel_perception, use_laplace=False)
        elif perception_type == 'sobel-with-laplace':
            perception_fn = partial(sobel_perception, use_laplace=True)
        elif perception_type == 'laplace':
            perception_fn = laplace_perception
        else:
            perception_fn = nn.Conv2d(
                in_channels=state_size,
                out_channels=state_size,
                kernel_size=3,
                padding=1,
                padding_mode='wrap',
                groups=state_size,
                key=conv_key
            )

        # Update function
        dummy_state = jnp.zeros((state_size, 8, 8))
        perception_out_size = perception_fn(dummy_state, key=conv_key).shape[0]

        layer_input_size, layers = perception_out_size, []
        for _ in range(update_depth):
            update_depth, conv_key = jr.split(update_key)
            layers.extend([
                nn.Conv2d(layer_input_size, update_width, kernel_size=1, key=conv_key),
                nn.Lambda(jax.nn.relu),
            ])
            layer_input_size = update_width

        layers.append(nn.Conv2d(layer_input_size, state_size, kernel_size=1, key=update_key))
        if bound_updates:
            layers.append(nn.Lambda(jax.nn.sigmoid))

        update_fn = SimpleUpdate(nn.Sequential(layers), update_prob)

        self.state_size = state_size
        self.hidden_size = hidden_size
        self.ca = CellularAutomaton(perception_fn, update_fn, lambda i, g: i + g[:, None, None])
        self.num_dev_steps = num_dev_steps

    def __call__(
        self,
        init_state: Float[Array, "C H W"],
        goal: Float[Array, "..."] | None = None,
        steps=None,
        *,
        key: jax.Array,
    ):
        if steps is None:
            steps = self.num_dev_steps
        cell_states, dev_path = self.ca(init_state, goal, steps, key=key)
        return cell_states[:4], dev_path


#----------------------------------------- mNCA --------------------------------------------------

class mNCA(eqx.Module):
    state_size: int
    hidden_size: int
    ca: CellularAutomaton
    num_dev_steps: tuple[int, int]

    def __init__(
        self,
        hidden_size = 12,
        perception_type: Literal['sobel', 'laplace', 'sobel-with-laplace', 'learned'] = 'laplace',
        morphogen_type: Literal['gaussian', 'directional', 'sinusoidal', 'mixed'] = 'directional',
        update_width = 128,
        update_depth = 1,
        update_prob = 0.5,
        alive_threshold = 0.1,
        alive_index = 3,
        num_dev_steps = (48, 96),
        *,
        key
    ) -> None:
        super().__init__()

        state_size = hidden_size + 4
        conv_key, update_key = jr.split(key)

        # Perception function
        if perception_type == 'sobel':
            perception_fn = sobel_perception
        elif perception_type == 'sobel-with-laplace':
            perception_fn = partial(sobel_perception, use_laplace=True)
        elif perception_type == 'laplace':
            perception_fn = laplace_perception
        else:
            perception_fn = nn.Conv2d(
                in_channels=state_size,
                out_channels=state_size,
                kernel_size=3,
                padding=1,
                padding_mode='wrap',
                groups=state_size,
                key=conv_key
            )

        # Mophogen init
        if morphogen_type == 'gaussian':
            morphogen_fn = partial(gaussian_field, sigma=1.0)
        elif  morphogen_type == 'directional':
            morphogen_fn = partial(directional_fields, n=2)
        elif morphogen_type == 'sinusoidal':
            morphogen_fn = partial(sinusoidal_fields, channels=4)
        else:
            morphogen_fn = partial(
                mix_fields,
                n=4,
                gaussian_sigma=5.0,
                sin_freq_min=0.5,
                sin_freq_max=1.0,
            )

        morphogen_size = morphogen_fn(8, 8)[0].shape[0]
        morphogen_concat = lambda x: jnp.concat([x, morphogen_fn(*x.shape[1:])[0]])

        # Update function
        dummy_state = jnp.zeros((state_size, 8, 8))
        perception_out_size = perception_fn(dummy_state, key=conv_key).shape[0]

        layer_input_size = perception_out_size + morphogen_size
        layers: list[eqx.Module] = [ nn.Lambda(morphogen_concat)]
        for _ in range(update_depth):
            update_depth, conv_key = jr.split(update_key)
            layers.extend([
                nn.Conv2d(layer_input_size, update_width, kernel_size=1, key=conv_key),
                nn.Lambda(jax.nn.relu),
            ])
            layer_input_size = update_width
        layers.append(
            nn.Conv2d(layer_input_size, state_size, kernel_size=1, key=update_key)
        )

        update_fn = GrowingUpdate(
            nn.Sequential(layers), # type: ignore
            alive_threshold,
            alive_index,
            update_prob
        )

        # used to mask goals in growth-baed goal-guided models, otherwise it will be compiled away
        def mask_goal(cell_states, goal):
            alive = cell_states[alive_index:alive_index+1] > alive_threshold
            return cell_states + goal[..., None, None] * alive

        self.state_size = state_size
        self.hidden_size = hidden_size
        self.ca = CellularAutomaton(perception_fn, update_fn, mask_goal)
        self.num_dev_steps = num_dev_steps

    def __call__(
        self,
        init_state: Float[Array, "C H W"],
        goal: Float[Array, "..."] | None = None,
        steps=None,
        *,
        key: jax.Array,
    ):
        if steps is None:
            steps = self.num_dev_steps
        cell_states, dev_path = self.ca(init_state, goal, steps, key=key)
        return cell_states[:4], dev_path

