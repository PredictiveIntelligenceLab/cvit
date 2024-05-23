import os

import time
import ml_collections
import wandb

from einops import rearrange


import jax
from jax import random, jit
import jax.numpy as jnp
import orbax.checkpoint as ocp

from flax.training import train_state

from model import Vit
from utils import create_optimizer, create_checkpoint_manager
from data_pipeline import create_dataloaders, batch_parser

from swe_pipeline import create_swe_datasets


def create_train_state(config, model, tx):
    x = jnp.ones(config.x_dim)
    params = model.init(random.PRNGKey(config.seed), x=x)

    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    return state


class PatchHandler:
    def __init__(self, inputs, patch_size):
        self.patch_size = patch_size

        _, self.time, self.height, self.width, self.channel = inputs.shape

        self.patch_height, self.patch_width = (
            self.height // self.patch_size[0],
            self.width // self.patch_size[1],
        )

    def merge_patches(self, x):
        batch, _, _ = x.shape
        x = jnp.reshape(
            x,
            (
                batch,
                self.patch_height,
                self.patch_width,
                self.patch_size[0],
                self.patch_size[1],
                -1,
            ),
        )
        x = jnp.swapaxes(x, 2, 3)
        x = jnp.reshape(
            x,
            (
                batch,
                self.patch_height * self.patch_size[0],
                self.patch_width * self.patch_size[1],
                -1,
            ),
        )
        return x


def create_train_step(model, patch_handler):
    # TODO: Support multi-GPU training
    @jit
    def train_step(state, batch):
        def loss_fn(params):
            x, y = batch
            pred = model.apply(params, x)

            pred = patch_handler.merge_patches(pred)  # （B, H, W, C）
            loss = jnp.mean((jnp.squeeze(y) - jnp.squeeze(pred)) ** 2)

            return loss

        grad_fn = jax.value_and_grad(loss_fn, has_aux=False)
        loss, grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss

    return train_step


def create_eval_step(model, patch_handler):
    # TODO: Support multi-GPU training
    @jit
    def eval_step(state, batch):
        x, y = batch
        pred = model.apply(state.params, x)

        pred = patch_handler.merge_patches(pred)  # （B, H, W, C）

        pred = jnp.squeeze(pred)
        y = jnp.squeeze(y)

        pred = rearrange(jnp.squeeze(pred), "b h w c -> b (h w) c")
        y = rearrange(jnp.squeeze(y), "b h w c -> b (h w) c")

        l2_error = jnp.linalg.norm(y - pred, axis=(1, 2)) / jnp.linalg.norm(
            y, axis=(1, 2)
        )
        smse = jnp.mean((pred - y) ** 2, axis=(0, 1)).sum()
        return l2_error, smse

    return eval_step


def train_and_evaluate(config: ml_collections.ConfigDict):
    # Initialize model
    model = Vit(**config.model)
    # Create learning rate schedule and optimizer
    lr, tx = create_optimizer(config)
    state = create_train_state(config, model, tx)

    # Create checkpoint manager
    ckpt_path = os.path.join(os.getcwd(), config.wandb.name, "ckpt")
    ckpt_mngr = create_checkpoint_manager(config.saving, ckpt_path)

    # Create patch handler
    patch_handler = PatchHandler(jnp.ones(config.x_dim), config.model.patch_size[1:])

    train_step_fn = create_train_step(model, patch_handler)
    eval_step_fn = create_eval_step(model, patch_handler)

    # Create dataloaders
    train_dataset, test_dataset = create_swe_datasets(config.dataset)
    train_iter, test_iter = create_dataloaders(
        config.dataset, train_dataset, test_dataset
    )

    # Initialize W&B
    wandb_config = config.wandb
    wandb.init(project=wandb_config.project, name=wandb_config.name)

    # Train
    start_time = time.time()
    last_loss = 1.0
    for step in range(config.training.num_steps):
        batch = next(train_iter)
        batch = jax.tree_map(lambda x: jnp.squeeze(x), batch)
        state, loss = train_step_fn(state, batch)

        # Evaluate model
        if step % config.logging.log_interval == 0:
            l2_error_list = []
            smse_list = []
            for _ in range(config.logging.eval_steps):
                batch = next(test_iter)
                batch = jax.tree_map(lambda x: jnp.squeeze(x), batch)

                l2_error, smse = eval_step_fn(state, batch)
                l2_error_list.append(l2_error)
                smse_list.append(smse)

            l2_error = jnp.array(l2_error_list).mean()
            smse = jnp.array(smse_list).mean()

            log_dict = {
                "loss": loss,
                "l2_error": l2_error,
                "smse": smse,
                "lr": lr(state.step),
            }
            wandb.log(log_dict, step)
            end_time = time.time()
            print(
                "step: {}, loss: {:.3e}, test error: {:.3e}, test smse: {:.3e}, time: {:.3e}".format(
                    step, loss, l2_error, smse, end_time - start_time
                )
            )
            start_time = end_time

            # if loss blowup, restart training from the last checkpoint
            if loss >= last_loss * 5:
                print("Loss blowup detected, reverting to last checkpoint")
                state = ckpt_mngr.restore(
                    ckpt_mngr.latest_step(), args=ocp.args.StandardRestore(state)
                )
                # if revert to last checkpoint, skip the rest of the loop
                continue

        # Save checkpoints
        if step % config.saving.save_interval == 0 and loss < last_loss:
            ckpt_mngr.save(step, args=ocp.args.StandardSave(state))
            last_loss = loss
