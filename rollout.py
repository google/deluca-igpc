import os
import jax.numpy as jnp
import numpy as onp
import time
import jax, sys
from PIL import Image


def rollout(
    env,
    cost_func,
    U_old,
    k=None,
    K=None,
    X_old=None,
    alpha=1.0,
    D=None,
    F=None,
    H=None,
    start_state=None,
    render=False,
    render_dir=".",
):
    """
    Arg List
    env: The environment to do the rollout on. This is treated as a derstructible copy.
    U_old: A base open loop control sequence
    k: open loop gain (iLQR)
    K: closed loop gain (iLQR)
    X_old: Previous trajectory to compute gain (iLQR)
    alpha: Optional multplier to the open loop gain (iLQR)
    D: Optional noise vectors for the rollout (GPC)
    F: Linearization shift ??? (GPC)
    H: The horizon length to perform a rollout.
    render: Whether to render the rollout or not
    render_dir: Which directory to render the rollout.
    """
    if render and not os.path.exists(render_dir):
        os.makedirs(render_dir)
    if H == None:
        H = env.H
    X, U = [None for _ in range(H + 1)], [None for _ in range(H)]
    if start_state is None:
        start_state = env.init()
    X[0], cost = start_state, 0.0
    for h in range(H):
        if k is None:
            U[h] = U_old[h]
        else:
            X_flat = X[h].flatten()
            X_old_flat = X_old[h].flatten()
            U[h] = U_old[h] + alpha * k[h] + K[h] @ (X_flat - X_old_flat)

        if D is None:
            X[h + 1] = env(X[h], U[h])
        elif F is None:
            X_next = env(X[h],U[h])
            X[h + 1] = X_next.unflatten(X_next.flatten() + D[h])
        else:
            X[h + 1] = (
                X_old[h + 1] + F[h][0] @ (X[h] - X_old[h]) + F[h][1] @ (U[h] - U_old[h])
            )
        cost += cost_func(X[h], U[h], env)
        
        if render:
            img = env.render(mode="rgb_array")
            img = Image.fromarray(img)
            img.save("{}/{}.png".format(render_dir, h), "PNG")
    return X, U, cost


def hessian(f, arg):
    return jax.jacfwd(jax.jacrev(f, argnums=arg), argnums=arg)


def compute_ders(env, cost, X, U, H=None):
    # if H == None:
    #     H = env.H
    H = env.H
    D, F, C = (
        [None for _ in range(H)],
        [None for _ in range(H)],
        [None for _ in range(H)],
    )
    # flat_X = [jax.tree_util.tree_flatten(x) for x in X]
    # flat_X_data = [ for ]
    # def F_temp(x):
    #     print(x)
    #     return jax.tree_util.tree_flatten(jax.jacfwd(env, argnums=(0,1))(x,[2., 3.]))[0]
    # print(len(X), len(X[:-1]), len(U))
    # #print(X,U)
    # F_temp = jax.pmap(F_temp)(X[:-1])
    # print(F_temp.shape)
    # def loop_fun(carry, args):
    #     t = args
    #     F = jax.tree_util.tree_flatten(jax.jacfwd(env, argnums=(0,1))(carry[t][0],carry[t][1]))[0]
    #     return None, F

    # _, a = jax.lax.scan(loop_fun, list(zip(X[:-1], U)), jnp.arange(H))
    # print(len(a))

    for h in range(H):
        #s = time.time()
        D[h] = X[h+1].flatten() - env(X[h], U[h]).flatten()
        #ss = time.time()
        #print('aaaa', ss-s)
        #s = time.time()
        g = jax.jacfwd(env, argnums=(0,1))(X[h],U[h])
        #ss = time.time()
        #print('aa', ss-s)
        F[h] = jax.tree_util.tree_flatten(g)[0]
        #tt = time.time()
        #print('a', tt-ss)
        c_der = jax.tree_util.tree_flatten(jax.grad(cost, argnums=(0,1))(X[h],U[h], env))[0]
        #s = time.time()
        #print('b', s-tt)
        c_der_der = jax.tree_util.tree_flatten(hessian(cost, (0,1))(X[h],U[h], env))[0]
        #tt = time.time()
        #print('c', tt-s)
        C[h] = (c_der[0], c_der[1], c_der_der[0], c_der_der[3])
    return D,F,C
