import sys
import os
import jax

import sbgm

sys.path.append(os.path.dirname(__file__))

from configs import *


def main():
    """
        Fit a score-based diffusion model.
    """

    # Load (data) and save (model, samples & optimiser state) directories 
    datasets_path = "../datasets/"
    root_dir = "./"

    # Config file for architecture and optimisation
    config = dict(
        mnist=mnist_config(), 
        grfs=grfs_config(),
        flowers=flowers_config(),
        cifar10=cifar10_config(),
        quijote=quijote_config() 
    )["mnist"]

    key = jax.random.key(config.seed)
    data_key, model_key, train_key = jax.random.split(key, 3)

    # Dataset object of training data and loaders
    dataset = sbgm.data.get_dataset(datasets_path, data_key, config)

    # Multiple GPU training if you are so inclined
    sharding, replicated_sharding = sbgm.shard.get_shardings()

    # Restart training or not
    reload_opt_state = False 
        
    # Diffusion model 
    model = sbgm.models.get_model(
        model_key, 
        config.model.model_type, 
        config,
        dataset.data_shape, 
        dataset.context_shape, 
        dataset.parameter_dim
    )

    # Stochastic differential equation (SDE)
    sde = sbgm.sde.get_sde(config.sde)

    # Fit model to dataset
    model = sbgm.train.train_from_config(
        train_key,
        model,
        sde,
        dataset,
        config,
        reload_opt_state=reload_opt_state,
        plot_train_data=True,
        sharding=sharding,
        replicated_sharding=replicated_sharding,
        save_dir=root_dir
    )


if __name__ == "__main__":
    main()