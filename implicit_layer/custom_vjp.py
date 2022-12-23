import jax
import numpy as np
import jax.numpy as jnp
from typing import Callable

def fn_with_estimator(forward_fn: Callable[..., jnp.ndarray],
        estimator_fn: Callable[..., jnp.ndarray]) -> Callable[..., jnp.ndarray]:

    @jax.custom_vjp
    def f(*args):
        return forward_fn(*args)

    def f_fwd(*args):
        _, vjp_fn = jax.vjp(estimator_fn, *args)
        return f(*args), vjp_fn

    def f_rev(vjp_fn, x_dot):
        return vjp_fn(x_dot)

    f.defvjp(f_fwd, f_rev)
    return f


def main():
    # sign_fn_tanh_estimator
    forward_fn = jnp.sign
    estimator_fn = jnp.tanh
    transformed_fn = fn_with_estimator(forward_fn, estimator_fn)
    for x in jnp.linspace(-2, 2, 4):
        expected_value = forward_fn(x)
        expected_grad = jax.grad(estimator_fn)(x)
        print('value', expected_value, transformed_fn(x))
        grad = jax.grad(transformed_fn)(x)
        print('grad', expected_grad, grad)


if __name__=="__main__":
    main()

