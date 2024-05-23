import os

import jax
import jax.numpy as jnp
from jax import lax, jit, vmap, pmap, random, tree_map, jacfwd

from flax.training import train_state

import optax
import orbax.checkpoint as ocp


def create_optimizer(config):
    lr = optax.warmup_exponential_decay_schedule(init_value=config.lr.init_value,
                                                 peak_value=config.lr.peak_value,
                                                 end_value=config.lr.end_value,
                                                 warmup_steps=config.lr.warmup_steps,
                                                 transition_steps=config.lr.transition_steps,
                                                 decay_rate=config.lr.decay_rate)

    tx = optax.chain(
        optax.clip_by_global_norm(config.optim.clip_norm),
        optax.adamw(lr, weight_decay=config.optim.weight_decay)
        )

    return lr, tx


def create_train_state(config, model, tx):

    x = jnp.ones(config.x_dim)
    coords = jnp.ones(config.coords_dim)
    params = model.init(random.PRNGKey(config.seed), x=x, coords=coords)

    state = train_state.TrainState.create(apply_fn=model.apply,
                                          params=params,
                                          tx=tx)

    return state


def create_checkpoint_manager(config, ckpt_path):
    ckpt_options = ocp.CheckpointManagerOptions(max_to_keep=config.num_keep_ckpts)

    if not os.path.isdir(ckpt_path):
        os.makedirs(ckpt_path)

    ckpt_mngr = ocp.CheckpointManager(
        ckpt_path,
        options=ckpt_options
        )

    return ckpt_mngr


def create_train_step(model):
    # TODO: Support multi-GPU training
    @jit
    def train_step(state, batch):
        def loss_fn(params):
            coords, x, y = batch
            pred = vmap(model.apply, (None, None, 0), out_axes=2)(params, x, coords[:, None, :])
            loss = jnp.mean((jnp.squeeze(y) - jnp.squeeze(pred)) ** 2)
            return loss

        grad_fn = jax.value_and_grad(loss_fn, has_aux=False)
        loss, grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss

    return train_step


def create_eval_step(model):
    # TODO: Support multi-GPU training
    @jit
    def eval_step(state, batch):
        coords, x, y = batch
        pred = vmap(model.apply, (None, None, 0), out_axes=2)(state.params, x, coords[:, None, :])

        y = jnp.squeeze(y)
        pred = jnp.squeeze(pred)

        l2_error = jnp.linalg.norm(y - pred, axis=(1, 2)) / jnp.linalg.norm(y, axis=(1, 2))
        smse = jnp.mean((pred - y) ** 2, axis=(0, 1)).sum()
        return l2_error, smse

    return eval_step


def rollout(state, x, coords, prev_steps=2, pred_steps=1, rollout_steps=5):
    b, t, h, w, c = x.shape
    pred_list = []
    for k in range(rollout_steps):
        pred = vmap(state.apply_fn, (None, None, 0), out_axes=2)(state.params, x, coords[:, None, :])
        pred = pred.reshape(b, pred_steps, h, w, c)

        x = jnp.concatenate([x, pred], axis=1)
        x = x[:, -prev_steps:]
        pred_list.append(pred)

    pred = jnp.concatenate(pred_list, axis=1)
    return pred