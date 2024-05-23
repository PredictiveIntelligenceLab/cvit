import os
from myutils import gpu_utils

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_utils.get_next_available_gpu())

import jax
from models import OperatorModel
from loaders import get_train_val_test_loaders


batch_size = 256
grid_size = 128
n_iterations = 200000

train_dataloader, val_loader, _ = get_train_val_test_loaders(batch_size, grid_size)
model = OperatorModel(jax.random.PRNGKey(0), "nomad")
model.train(iter(train_dataloader), iter(val_loader), n_iterations)
