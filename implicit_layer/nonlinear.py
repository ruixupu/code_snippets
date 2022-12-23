import jax
import numpy as np
import jax.numpy as jnp
import scipy.optimize as opt

_X_INIT = jnp.array([2.0])

def solver(fn, p):
    return opt.root(fn, p, _X_INIT, method='hybr').x


def jax_wrapper(fn, *args):
    @jax.custom_vjp
    def f(*args):
        return solver(fn, *args)

    def f_fwd(*args):
        x_star = solver(fn, *args)
        return x_star, (*args, x_star)

    def f_rev(res, l_x):
        *args, x_star = res
        f_p, f_x = jax.jacrev(fn, argnums=(0, 1))(*args, x_star)
        r = np.linalg.solve(f_x.T, -l_x.T)
        return (jnp.dot(r.T, f_p), )

    f.defvjp(f_fwd, f_rev)

    return f

def loss(fn, p):
    x_star = fn(p)
    return jnp.sum(x_star**2)

def main():
    fn = lambda p, x: x - p / x**2
    grad_fn = lambda p,x : 2.0*x/(1.0+2.0*p/x**3)/x**2

    p = 4.1
    x_star = solver(fn, p)
    print('expected_value_grad:', x_star, grad_fn(p, x_star))

    new_fn = jax_wrapper(fn, p)
    new_grad = jax.grad(loss, argnums=(1,))(new_fn, p)
    print('actual_value_grad:', new_fn(p), new_grad)

    return


if __name__ == '__main__':
    main()
