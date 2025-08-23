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
import jax_dataloader as jdl
from tqdm import tqdm
from typing import Callable
from jaxtyping import Array, Float, PyTree

from src.model.nca import GrowingNCA, NoiseNCA, mNCA
from src.nn.seeding import init_central_seed, init_random
from src.dataset.emoji import EmojiDataset
# from src.visualisation.dev import plot_dev_path
from src.visualisation.utils import plot_examples
from src.utils import  cycle, seed_everything, save_pytree


logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def main(
    model: str,
    perception_type: str,
    morphogen_type: str,
    train_iters: int,
    lr: float,
    batch_size: int,
    save_folder: Path,
    seed: int | None,
    run,
):
    rng, key = seed_everything(seed)

    # Set training data
    _logger.info("Loading dataset...")
    data = EmojiDataset(target_size=40, pad=16, return_one_hot=False)
    assert batch_size < len(data)

    dataloader = jdl.DataLoader(
        data,
        backend='jax',
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        generator=rng,
    )

    # Init model
    _logger.info("Done. Initialising model...")
    if model == 'NCA':
        nca = GrowingNCA(
            hidden_size=12,
            perception_type=perception_type,
            key=key,
        )
        init_fn = partial(init_central_seed, shape=(nca.state_size, *data.image_size))

    elif model == 'noiseNCA':
        nca = NoiseNCA(
            hidden_size=12,
            perception_type=perception_type,
            key=key,
        )
        init_fn = partial(init_random, shape=(nca.state_size, *data.image_size))

    elif model == 'mNCA':
        nca = mNCA(
            hidden_size=12,
            perception_type=perception_type,
            morphogen_type=morphogen_type,
            key=key,
        )
        init_fn = partial(init_central_seed, shape=(nca.state_size, *data.image_size))

    else:
        raise NotImplementedError("Model not recognized")

    goal_embeddings = eqx.nn.Embedding(
        num_embeddings=len(data),
        embedding_size=nca.hidden_size + (model == 'steer-mNCA'),  # +1 for steerable models
        key=key
    )

    def apply_nca(model, inputs, key, steps=None):
        goal = goal_embeddings(inputs)
        goal = jnp.concat([jnp.zeros((3,)), jnp.ones((1,)), goal])
        init_state = init_fn(key=key)
        return model(init_state, goal, steps, key=key)

    # Setup training
    _logger.info("Done. Setting up training...")
    optim = optax.chain(
        optax.clip_by_global_norm(1.0),
        # optax.clip_by_block_rms(1.0),
        optax.adamw(lr),
    )
    opt_state = optim.init(eqx.filter(nca, eqx.is_array))

    def compute_loss(
        model: Callable,
        batch: tuple[jax.Array, jax.Array],
        key: jax.Array
    ):
        inputs, targets = batch
        batch_key = jr.split(key, targets.shape[0])
        preds, _ = jax.vmap(apply_nca, in_axes=(None, 0, 0))(model, inputs, batch_key)
        return jnp.sum(optax.l2_loss(preds, targets)) / len(targets)

    @eqx.filter_jit(donate='all')
    def train_step(
        model: PyTree,
        opt_state: PyTree,
        batch: tuple[Float[Array, "NCHW"], Float[Array, "NCHW"]],
        key: jax.Array,
    ):
        loss_value, grads = eqx.filter_value_and_grad(compute_loss)(model, batch, key)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value

    # Train
    _logger.info("Done. Training starting...")

    for i, batch in zip(pbar := tqdm(range(train_iters)), cycle(dataloader)):
        key, step_key = jr.split(key)
        nca, opt_state, train_loss = train_step(nca, opt_state, batch, step_key)
        pbar.set_postfix_str(f"iter: {i}; loss: {np.asarray(train_loss)}")
        run.log({"mse": float(train_loss)})

    _logger.info("Training completed.\nRunning evaluation...")
    # Run a simple evaluation
    output, _ = jax.vmap(apply_nca, in_axes=(None, 0, 0))(
        nca, data[np.arange(len(data))][0], jr.split(key, len(data))
    )

    # Save results
    _logger.info("Saving results...")

    save_folder = Path(save_folder)
    os.makedirs(save_folder, exist_ok=True)

    plot_examples(output, w=8, format='NCHW').savefig(save_folder / "examples.png")
    # plot_dev_path(dev_path[:, :4]).save(save_folder / "growth.gif")
    save_pytree((nca, goal_embeddings), save_folder / "checkpoint.eqx")
    _logger.info("Done. Script terminating.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=["NCA", "mNCA", "noiseNCA"],
        default='NCA',
    )
    parser.add_argument(
        "--perception_type",
        choices=['sobel', 'laplace', 'sobel-with-laplace'],
        default='sobel',
    )
    parser.add_argument(
        "--morphogen_type",
        choices=['directional', 'gaussian', 'sinusoidal', 'mixed'],
        default='mixed',
    )
    parser.add_argument(
        "--dataset_hash",
        type=str,
        default="5e88adc27d18deb1",
    )
    parser.add_argument(
        "--train_iters",
        type=int,
        default=5000,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
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
    parser.add_argument(
        "--debug",
        action='store_true',
        default=False,
    )

    args = vars(parser.parse_args())
    main(**args)

