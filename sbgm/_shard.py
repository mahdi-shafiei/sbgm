from typing import Tuple
import jax
from jax.experimental import mesh_utils


def get_shardings() -> Tuple[jax.sharding.PositionalSharding | None, jax.sharding.NamedSharding | None]:
    devices = jax.local_devices()
    n_devices = len(devices)
    print(f"Running on {n_devices} local devices: \n\t{devices}")

    if n_devices > 1:
        mesh = jax.sharding.Mesh(devices, ('x',))
        sharding = jax.sharding.NamedSharding(
            mesh, spec=jax.sharding.PartitionSpec('x')
        )

        devices = mesh_utils.create_device_mesh((n_devices, 1))
        replicated = jax.sharding.PositionalSharding(devices).replicate()
    else:
        sharding = replicated = None

    return sharding, replicated