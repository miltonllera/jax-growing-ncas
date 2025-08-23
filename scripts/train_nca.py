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
import optax
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
    train_iters: int,
    lr: float,
    save_folder: Path,
    seed: int | None,
):
    _, key = seed_everything(seed)

    # Set training data

    _logger.info("Loading dataset...")

    target = load_emoji(Emoji[target.upper()].value, 64)
    target = target[None].repeat(8, axis=0).transpose(0, 3, 1, 2)

    # Init model
    _logger.info("Done.\nInitialising model...")

    if model == 'NCA':
        nca = GrowingNCA(
            hidden_size=12,
            perception_type=perception_type,
            key=key
        )
        init_fn = partial(init_central_seed, shape=(nca.state_size, *target.shape[-2:]))

    elif model == 'noiseNCA':
        nca = NoiseNCA(
            hidden_size=12,
            perception_type=perception_type,
            key=key
        )

        init_fn = partial(init_random, shape=(nca.state_size, *target.shape[-2:]))

    else:
        raise NotImplementedError("Model not recognized")

    # Setup training
    _logger.info("Done.\nSetting up training...")

    optim = optax.chain(
        optax.clip_by_global_norm(1.0),
        # optax.clip_by_block_rms(1.0),
        optax.adamw(lr, b1=0.95, b2=0.9995),
    )
    opt_state = optim.init(eqx.filter(nca, eqx.is_array))

    def compute_loss(
        model: Callable,
        batch: tuple[jax.Array, jax.Array],
        key: jax.Array
    ):
        inputs, targets = batch
        batch_key = jr.split(key, targets.shape[0])
        preds, _ = jax.vmap(model)(inputs, key=batch_key)
        return jnp.sum(optax.l2_loss(preds, targets)) / len(targets)


    @eqx.filter_jit
    def train_step(
        model: PyTree,
        batch: tuple[Float[Array, "NCHW"], Float[Array, "NCHW"]],
        opt_state: PyTree,
        key: jax.Array,
    ):
        loss_value, grads = eqx.filter_value_and_grad(compute_loss)(model, batch, key)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return loss_value, model, opt_state

    # Train
    _logger.info("Done.\nTraining starting...")

    for i in (pbar := tqdm(range(train_iters))):
        key, step_key = jr.split(key)
        # NOTE: Use lambda to vmap over keys since these are keyword parameters.
        init_state = jax.vmap(lambda k: init_fn(key=k))(jr.split(step_key, len(target)))
        train_loss, nca, opt_state = train_step(nca, (init_state, target), opt_state, step_key)

        pbar.set_postfix_str(f"iter: {i}; loss: {np.asarray(train_loss)}")

    # evaluating the model
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
    def may_be_str_or_int(x: str):
        if x.isdecimal():
            return int(x)
        return x

    parser = argparse.ArgumentParser()

    parser. add_argument(
            "--model",
            choices=['NCA', 'noiseNCA', 'mNCA'],
            default='NCA',
        )
    parser.add_argument(
        "--perception_type",
        choices=['sobel', 'laplace', 'sobel-with-laplace', 'steerable', 'steerable-with-laplace'],
        default='sobel',
    )
    parser.add_argument(
        "--target",
        type=may_be_str_or_int,
        default=0,  # 1984
    )
    parser.add_argument(
        "--train_iters",
        type=int,
        default=5000,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3
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

