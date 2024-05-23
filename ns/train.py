import os

import time
import ml_collections
import wandb

import jax
from jax import random, vmap, jit
import jax.numpy as jnp
import orbax.checkpoint as ocp


from model import CVit
from utils import create_optimizer, create_train_state, create_checkpoint_manager, create_train_step, create_eval_step
from data_pipeline import create_dataloaders, batch_parser

from ns_pipeline import create_ns_datasets


def train_and_evaluate(config: ml_collections.ConfigDict):
    # Initialize model
    model = CVit(**config.model)
    # Create learning rate schedule and optimizer
    lr, tx = create_optimizer(config)
    state = create_train_state(config, model, tx)

    # Create checkpoint manager
    ckpt_path = os.path.join(os.getcwd(), config.wandb.name, "ckpt")
    ckpt_mngr = create_checkpoint_manager(config.saving, ckpt_path)

    # Create train and eval step functions
    train_step_fn = create_train_step(model)
    eval_step_fn = create_eval_step(model)

    # Create dataloaders
    train_dataset, test_dataset = create_ns_datasets(config.dataset)
    train_iter, test_iter = create_dataloaders(config.dataset, train_dataset, test_dataset)

    # Initialize W&B
    wandb_config = config.wandb
    wandb.init(project=wandb_config.project, name=wandb_config.name)

    # Train
    rng = random.PRNGKey(config.seed + 1)
    start_time = time.time()
    last_loss = 1.0
    for step in range(config.training.num_steps):
        rng, _ = random.split(rng)

        batch = next(train_iter)
        batch = batch_parser(batch, rng, config.dataset.num_query_points)
        state, loss = train_step_fn(state, batch)

        # Evaluate model
        if step % config.logging.log_interval == 0:

            l2_error_list = []
            smse_list = []
            for _ in range(config.logging.eval_steps):
                batch = next(test_iter)
                batch = batch_parser(batch)
                l2_error, smse = eval_step_fn(state, batch)
                l2_error_list.append(l2_error)
                smse_list.append(smse)

            l2_error = jnp.array(l2_error_list).mean()
            smse = jnp.array(smse_list).mean()

            log_dict = {'loss': loss, 'l2_error': l2_error, 'smse': smse, 'lr': lr(state.step)}
            wandb.log(log_dict, step)
            end_time = time.time()
            print("step: {}, loss: {:.3e}, test error: {:.3e}, test smse: {:.3e}, time: {:.3e}".format(step, loss,
                                                                                                       l2_error, smse,
                                                                                                       end_time - start_time))
            start_time = end_time

            # if loss blowup, restart training from the last checkpoint
            if loss >= last_loss * 5:
                print("Loss blowup detected, reverting to last checkpoint")
                state = ckpt_mngr.restore(ckpt_mngr.latest_step(), args=ocp.args.StandardRestore(state))
                # if revert to last checkpoint, skip the rest of the loop
                continue

        # Save checkpoints
        if (step % config.saving.save_interval == 0 and loss < last_loss) or step == config.training.num_steps - 1:
            ckpt_mngr.save(step, args=ocp.args.StandardSave(state))
            last_loss = loss










