import jax.random as jr

import sbgm
import data 
import configs 


def main():
    """
        Fit a score-based diffusion model.
    """

    # Load (data) and save (model, samples & optimiser state) directories 
    datasets_path = "../datasets/"
    root_dir = "./"

    # Config file for architecture and optimisation
    config = [
        configs.mnist_config(), 
        configs.grfs_config(),
        configs.flowers_config(),
        configs.cifar10_config(),
        configs.quijote_config() 
    ][3]

    key = jr.key(config.seed)
    data_key, model_key, train_key = jr.split(key, 3)

    # Dataset object of training data and loaders
    dataset = data.get_dataset(datasets_path, data_key, config)

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