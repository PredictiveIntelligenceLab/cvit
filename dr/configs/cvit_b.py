import ml_collections



def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.mode = "train"

    # Input shape for initializing Flax models
    config.x_dim = [10, 2, 128, 128, 2]
    config.coords_dim = [1024, 2]

    # Integer for PRNG random seed.
    config.seed = 42

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "dr"
    wandb.name = "dr_cvit_b_8x8_"
    wandb.tag = None

    # Model
    config.model = model = ml_collections.ConfigDict()
    model.patch_size = (1, 8, 8)
    model.grid_size = (128, 128)
    model.latent_dim = 512
    model.emb_dim = 512
    model.depth = 10
    model.num_heads = 8
    model.dec_emb_dim = 512
    model.dec_num_heads = 16
    model.dec_depth = 1
    model.num_mlp_layers = 1
    model.mlp_ratio = 1
    model.out_dim = 2
    model.eps = 1e4
    model.embedding_type = "grid"

    # Dataset
    config.dataset = dataset = ml_collections.ConfigDict()
    dataset.path = '/scratch/PDEDatasets/pdebench/2d_reacdiff/2D/diffusion-reaction/2D_diff-react_NA_NA.h5'
    dataset.prev_steps = 2
    dataset.pred_steps = 1
    dataset.downsample_factor = 10
    dataset.train_samples = 900
    dataset.test_samples = 50
    dataset.batch_size = 128
    dataset.num_query_points = 1024
    dataset.num_workers = 8

    # Learning rate
    config.lr = lr = ml_collections.ConfigDict()
    lr.init_value = 0.0
    lr.end_value = 1e-6
    lr.peak_value = 1e-3
    lr.decay_rate = 0.9
    lr.transition_steps = 5000
    lr.warmup_steps = 5000

    # Optim
    config.optim = optim = ml_collections.ConfigDict()
    optim.beta1 = 0.9
    optim.beta2 = 0.999
    optim.eps = 1e-8
    optim.weight_decay = 1e-5
    optim.clip_norm = 1.0

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.num_steps = 3 * 10 ** 5

    # Logging
    config.logging = logging = ml_collections.ConfigDict()
    logging.log_interval = 200
    logging.eval_steps = 10

    # Saving
    config.saving = saving = ml_collections.ConfigDict()
    saving.save_interval = 200
    saving.num_keep_ckpts = 5

    # Evaluation
    config.eval = eval = ml_collections.ConfigDict()
    eval.rollout_steps = 5

    return config