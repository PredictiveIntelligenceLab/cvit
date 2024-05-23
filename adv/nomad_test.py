import os
from myutils import gpu_utils

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_utils.get_next_available_gpu())

import jax
import h5py
import jax.numpy as jnp
from models import OperatorModel
from loaders import get_train_val_test_loaders


batch_size = 256
grid_size = 200
n_iterations = 200000

_, _, test_loader = get_train_val_test_loaders(batch_size, grid_size)
model = OperatorModel(jax.random.PRNGKey(0), "nomad")
model.load_model()

inputs_test, grid_test, outputs_test = test_loader.u, test_loader.y, test_loader.s

s_pred = []
for i in range(10):
    u, y, s = (
        inputs_test[i * 1000 : (i + 1) * 1000],
        grid_test[i * 1000 : (i + 1) * 1000],
        outputs_test[i * 1000 : (i + 1) * 1000],
    )
    # for nomad / deeponet:
    s_pred.append(
        jnp.squeeze(
            jax.vmap(jax.vmap(model.s_net, (None, None, 0)), (None, 0, 0))(
                model.avg_params, jnp.squeeze(u), y
            )
        )
    )

s_pred = jnp.array(s_pred).reshape(10000, 200)
error_test = jnp.linalg.norm(outputs_test - s_pred, 2, axis=-1) / jnp.linalg.norm(
    outputs_test, 2, axis=-1
)

print(
    "Relative L2 Error (Min, Median, Mean, Max):, ",
    jnp.min(error_test),
    jnp.median(error_test),
    jnp.mean(error_test),
    jnp.max(error_test),
)

h5f = h5py.File("nomad.h5", "w")
h5f.create_dataset("error_test", data=error_test)
h5f.create_dataset("s_pred", data=s_pred)
h5f.create_dataset("outputs_test", data=outputs_test)
h5f.close()
