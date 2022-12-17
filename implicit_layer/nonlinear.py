import jax
import numpy as np
import jax.numpy as jnp
import scipy.optimize as opt

def jax_wrapper(fn, p, x_init):
    @jax.custom_vjp
    def root(p):
        return opt.root(fn, p, x_init, method='hybr').x

    def root_fwd(p):
        x_star = opt.root(fn, p, x_init, method='hybr').x
        return x_star, (p, x_star)

    def root_bwd(res, l_x):
        p, x_star = res
        f_p, f_x = jax.jacrev(fn, argnums=(0,1))(p, x_star)
        r = np.linalg.solve(f_x.T, -l_x.T)
        return (jnp.dot(r.T, f_p), )

    root.defvjp(root_fwd, root_bwd)
    return root

def loss(p, fn):
    x_star = fn(p)
    return jnp.sum(x_star)**2

def main():
    fn = lambda p, x: x - p / x**2
    grad_fn = lambda p,x : 2.0*x/(1.0+2.0*p/x**3)/x**2

    p = 4.1
    x_init = jnp.array([2.0])
    x_star = opt.root(fn, p, x_init).x
    print('expected_grad:', grad_fn(p, x_star))

    new_fn = jax_wrapper(fn, p, x_init)
    new_grad = jax.grad(loss, argnums=(0,))(p, new_fn)
    print('actual_grad:', new_grad)

    return


if __name__ == '__main__':
    main()
