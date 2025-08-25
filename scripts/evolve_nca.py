import os
import argparse
import logging
from pathlib import Path
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
# import optax
import evosax.algorithms as es
from tqdm import tqdm
from typing import Callable
from jaxtyping import Array, Float, PyTree

from src.model.nca import GrowingNCA, NoiseNCA
from src.nn.seeding import init_central_seed, init_random
from src.dataset.emoji import load_emoji, Emoji
from src.visualisation.dev import plot_state, plot_dev_path
from src.utils import  seed_everything, save_pytree


logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def main(
    model: str,
    perception_type: str,
    target,
    target_size: int,
    target_pad: int,
    evo_iters: int,
    pop_size: float,
    save_folder: Path,
    seed: int | None,
):
    _, key = seed_everything(seed)

    # Set training data
    _logger.info("Loading target example...")
    target = load_emoji(Emoji[target.upper()].value, target_size).transpose(2, 0, 1)  # to CHW
    target = np.pad(target, ((0, 0), (target_pad, target_pad), (target_pad, target_pad)))

    # Init model
    _logger.info("Done.\nInitialising model...")

    if model == 'NCA':
        nca = GrowingNCA(
            hidden_size=12,
            perception_type=perception_type,
            bound_updates=True,
            key=key
        )
        init_fn = partial(init_central_seed, shape=(nca.state_size, *target.shape[-2:]))

    elif model == 'noiseNCA':
        nca = NoiseNCA(
            hidden_size=12,
            perception_type=perception_type,
            bound_updates=True,
            key=key
        )
        init_fn = partial(init_random, shape=(nca.state_size, *target.shape[-2:]))

    else:
        raise NotImplementedError("Model not recognized")

    # Setup training
    _logger.info("Done.\nSetting up training...")

    params, statics = eqx.partition(nca, eqx.is_array)

    # CMA
    strat = es.CMA_ES(population_size=256, solution=params)
    strat_params = strat.default_params.replace(std_init=0.01)  # type: ignore

    # OpenES
    # lr_schedule = optax.exponential_decay(
    #     init_value=0.001,
    #     transition_steps=evo_iters,
    #     decay_rate=0.01,
    # )
    # std_schedule = optax.exponential_decay(
    #     init_value=0.01,
    #     transition_steps=evo_iters,
    #     decay_rate=1.0,
    # )

    # algo = Open_ES(
    #     population_size=pop_size,
    #     solution=params,
    #     optimizer=optax.adam(learning_rate=lr_schedule),
    #     std_schedule=optax.constant_schedule(0.01),
    # )
    # strat_params = strat.default_params

    evo_state = strat.init(key, params, strat_params)

    def compute_fitness(
        params: Callable,
        target: Float[Array, "CHW"],
        key: jax.Array
    ):
        init_key, eval_key = jr.split(key)
        init_state = init_fn(key=init_key)
        preds, _ = eqx.combine(params, statics)(init_state, key=eval_key)
        return jnp.sum((jnp.clip(preds, min=0.0, max=1.0) - target) ** 2)

    @eqx.filter_jit
    def evo_step(
        evo_state: PyTree,
        target: Float[Array, "CHW"],
        key: jax.Array,
    ):
        ask_key, eval_key, tell_key = jr.split(key, 3)
        candidates, evo_state = strat.ask(ask_key, evo_state, strat_params)
        fitness = jax.vmap(compute_fitness, in_axes=(0, None, 0))(
            candidates, target, jr.split(eval_key, pop_size)
        )
        evo_state, _ = strat.tell(tell_key, candidates, fitness, evo_state, strat_params)
        metrics= {
            'mean_fitness': fitness.mean(),
            'best_fitness': fitness.min(),
            'best_fitnees_ever': evo_state.best_fitness,
        }
        return evo_state, metrics

    # Train
    _logger.info("Done.\nEvolution starting...")

    for i in (pbar := tqdm(range(evo_iters))):
        key, step_key = jr.split(key)
        evo_state, metrics = evo_step(evo_state, target, step_key)
        pbar.set_postfix_str( f"iter: {i}; mean fitneess: {float(metrics['mean_fitness'])}")

    # Evaluating the best individual
    _logger.info("Done.\nEvaluating the best individual...")
    nca = eqx.combine(evo_state.best_solution, statics)
    output, dev_path = nca(init_fn(key=key), steps=96, key=key)

    # Save results
    _logger.info("Saving results...")

    save_folder = Path(save_folder)
    os.makedirs(save_folder, exist_ok=True)

    plot_state(output).savefig(save_folder / "example.png")
    plot_dev_path(dev_path[:, :4]).save(save_folder / "growth.gif")
    save_pytree(nca, save_folder / "checkpoint.eqx")

    _logger.info("Done.\nScript terminating.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser. add_argument(
        "--model",
        choices=['NCA', 'noiseNCA', 'mNCA'],
        default='NCA',
    )
    parser.add_argument(
        "--perception_type",
        choices=['sobel', 'sobel-with-laplace'],
        default='sobel',
    )
    parser.add_argument(
        "--target",
        type=str,
        default="salamander",  # 1984
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--evo_iters",
        type=int,
        default=5000,
    )
    parser.add_argument(
        "--pop_size",
        type=float,
        default=256,
    )
    parser.add_argument(
        "--save_folder",
        type=str,
        default="data/logs/temp"
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        default=None,
    )

    args = vars(parser.parse_args())
    main(**args)

