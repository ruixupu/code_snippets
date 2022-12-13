from absl import logging
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.training import train_state
import ml_collections
import tensorflow_dataset as tfds


def load_datasets():
    ds_builder = tfds.builder('mnist')
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
    train_ds['image'] = jnp.float32(train_ds['image']) / 255.0
    test_ds['image'] = jnp.float32(test_ds['image']) / 255.0
    return train_ds, test_ds

class CNN(nn.Module):

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3,3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2,2), strides=(2,2))
        x = nn.Conv(features=64, kernel_size=(3,3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2,2), strides=(2,2))
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return x




def train_and_evaluate(config: ml_collections.FrozenConfigDict(), workdir:
        str) -> train_state.TrainState:
    train_ds, test_ds = load_datasets()
    rng = jax.random.PRNGKey(0)

    summary_writer = tensorboard.SummaryWriter(workdir)
    summary_writer.hparams(dict(config))

    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, config)

    for epoch in range(1, config.num_epochs+1):
        rng, epoch_rng = jax.random.split(rng)
        state, loss, accuracy = train_epoch(state, train_ds, config.batch_size,
                epoch_rng)
        _, test_loss, test_accuracy = apply_model(state, test_ds)

        logging.info("epoch: %3d, loss: %.2f, test_loss: %.2f, test_accuracy:
        %.2f" % (loss, test_loss, test_accuracy))

        summary_writer.scalar("train_loss", loss, epoch)

    summary_writer.flush()
    return state
