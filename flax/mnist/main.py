import flax
import optax
import numpy as np
import jax.numpy as jnp
from absl import logging
import tensorflow_datasets as tfds


import train
import default as config_lib

logging.set_verbosity(logging.INFO)


def main():
    config = config_lib.get_config()
    state = train.train_and_evaluate(config, workdir=f'./model/')
    return


if __name__=="__main__":
    main()


