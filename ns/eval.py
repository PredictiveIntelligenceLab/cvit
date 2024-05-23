import os

import einops


import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

import orbax.checkpoint as ocp

from torch.utils.data import TensorDataset, Dataset, DataLoader, Subset

from model import CVit
from utils import (
    create_optimizer,
    create_train_state,
    create_checkpoint_manager,
    rollout,
)
from data_pipeline import BaseDataset

from ns_pipeline import prepare_ns_dataset


def evaluate(config):
    # Initialize model
    model = CVit(**config.model)
    # Create learning rate schedule and optimizer
    lr, tx = create_optimizer(config)
    state = create_train_state(config, model, tx)

    # Create checkpoint manager
    ckpt_path = os.path.join(os.getcwd(), config.wandb.name, "ckpt")
    ckpt_mngr = create_checkpoint_manager(config.saving, ckpt_path)

    # Restore the model
    state = ckpt_mngr.restore(
        ckpt_mngr.latest_step(), args=ocp.args.StandardRestore(state)
    )

    flatten_params = ravel_pytree(state.params)[0]
    print("Total number of parameters: {:,}".format(len(flatten_params)))

    # One-step
    test_inputs, test_outputs = prepare_ns_dataset(
        directory=config.dataset.path,
        keys=config.dataset.components,
        prev_steps=config.dataset.prev_steps,
        pred_steps=config.dataset.pred_steps,
        mode="test",
        num_samples=1000,
    )

    test_dataset = BaseDataset(test_inputs, test_outputs)

    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, drop_last=True, num_workers=8
    )

    # Create a grid for cvit
    _, t, h, w, c = test_inputs.shape
    x_star = jnp.linspace(0, 1, h)
    y_star = jnp.linspace(0, 1, w)
    x_star, y_star = jnp.meshgrid(x_star, y_star, indexing="ij")
    coords = jnp.hstack([x_star.flatten()[:, None], y_star.flatten()[:, None]])

    l2_error_list = []
    for batch in test_loader:
        batch = jax.tree_map(lambda x: jnp.array(x), batch)
        x, y = batch
        pred = model.apply(state.params, x, coords)
        pred = pred.reshape(-1, 1, h, w, c)

        pred = einops.rearrange(pred, "B T H W C-> B (T H W) C")
        y = einops.rearrange(y, "B T H W C-> B (T H W) C")

        diff_norms = jnp.linalg.norm(pred - y, axis=1)
        y_norms = jnp.linalg.norm(y, axis=1)

        l2_error = (diff_norms / y_norms).mean(axis=1)
        l2_error_list.append(l2_error)

    l2_error = jnp.mean(jnp.array(l2_error_list))
    print("l2_error:", l2_error)

    # Multiple-step rollout
    test_inputs, test_outputs = prepare_ns_dataset(
        directory=config.dataset.path,
        keys=config.dataset.components,
        prev_steps=config.dataset.prev_steps,
        pred_steps=config.eval.rollout_steps,
        mode="test",
        num_samples=1000,
    )

    test_dataset = BaseDataset(test_inputs, test_outputs)

    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, drop_last=True, num_workers=8
    )

    l2_error_list = []
    for batch in test_loader:
        batch = jax.tree_map(lambda x: jnp.array(x), batch)
        x, y = batch

        pred = rollout(
            state,
            x,
            coords,
            prev_steps=config.dataset.prev_steps,
            pred_steps=config.dataset.pred_steps,
            rollout_steps=config.eval.rollout_steps,
        )

        pred = einops.rearrange(pred, "B T H W C-> B (T H W) C")
        y = einops.rearrange(y, "B T H W C-> B (T H W) C")

        diff_norms = jnp.linalg.norm(pred - y, axis=1)
        y_norms = jnp.linalg.norm(y, axis=1)

        l2_error = (diff_norms / y_norms).mean(axis=1)
        l2_error_list.append(l2_error)

    l2_error = jnp.mean(jnp.array(l2_error_list))
    print("l2_error:", l2_error)
