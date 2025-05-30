from typing import Tuple, Optional
import jax


def get_shardings() -> Tuple[
    Optional[jax.sharding.NamedSharding], Optional[jax.sharding.NamedSharding]
]:
    devices = jax.devices()
    n_devices = len(devices)

    print(f"Running on {n_devices} local devices: \n\t{devices}")

    if n_devices > 1:

        mesh = jax.sharding.Mesh(devices, "x")

        replicated = jax.sharding.NamedSharding(
            mesh, jax.sharding.PartitionSpec()
        )
        sharding = jax.sharding.NamedSharding(
            mesh, jax.sharding.PartitionSpec("x")
        )

    else:
        sharding = replicated = None

    return sharding, replicated