from absl import logging
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.metrics import tensorboard
from flax import linen as nn
from flax.training import train_state
import ml_collections
import tensorflow_datasets as tfds


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


def create_train_state(rng, config):
    cnn = CNN()
    params = cnn.init(rng, jnp.ones((1,28,28,1)))['params']
    tx = optax.sgd(config.learning_rate, config.momentum)
    return train_state.TrainState.create(apply_fn=cnn.apply, params=params,
            tx=tx)


@jax.jit
def apply_model(state, images, labels):
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, images)
        one_hot = jax.nn.one_hot(labels, 10)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits,
            labels=one_hot))
        return loss, logits

    (loss, logits), grads = jax.value_and_grad(loss_fn,
            has_aux=True)(state.params)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return grads, loss, accuracy


def train_epoch(state, train_ds, batch_size, rng):
    train_ds_size = len(train_ds['image'])
    steps = train_ds_size // batch_size

    perms = jax.random.permutation(rng, len(train_ds['image']))
    perms = perms[:steps*batch_size]
    perms = perms.reshape((steps, batch_size))

    epoch_loss = []
    epoch_accuracy = []
    for perm in perms:
        batch_images = train_ds['image'][perm, ...]
        batch_labels = train_ds['label'][perm, ...]
        grads, loss, accuracy = apply_model(state, batch_images, batch_labels)
        state = state.apply_gradients(grads=grads)
        epoch_loss.append(loss)
        epoch_accuracy.append(accuracy)
    train_loss = np.mean(epoch_loss)
    train_accuracy = np.mean(epoch_accuracy)

    return state, train_loss, train_accuracy

def train_and_evaluate(config: ml_collections.ConfigDict(), workdir:
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
        _, test_loss, test_accuracy = apply_model(state, test_ds['image'],
                test_ds['label'])

        logging.info("epoch: %3d, loss: %.3f, test_loss: %.3f, test_accuracy: \
        %.3f" % (epoch, loss, test_loss, test_accuracy))

        summary_writer.scalar("train_loss", loss, epoch)

    summary_writer.flush()
    return state
