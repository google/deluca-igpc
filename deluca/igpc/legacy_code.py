import time
import jax.numpy as jnp
from deluca.igpc.rollout import rollout, compute_ders
from deluca.igpc.lqr_solver import LQR
from PIL import Image
import jax

@partial(jax.jit, static_argnums=(1,))
def rollout_with_lin_shift(
    env,
    cost_func,
    U_old,
    D,
    k=None,
    K=None,
    X_old=None,
    alpha=1.0,
    H=None,
    start_state=None
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
    """
    ## problematic
    # if H == None:
    #     H = env.H
    H = env.H
    X, U = [None] * (H + 1), [None] * (H)
    if start_state is None:
        start_state = env.init()
    #start_state = env.init()
    X[0], cost = start_state, 0.0
    for h in range(H):
        if k is None:
            U[h] = U_old[h]
        else:
            X_flat = X[h].flatten()
            X_old_flat = X_old[h].flatten()
            U[h] = U_old[h] + alpha * k[h] + K[h] @ (X_flat - X_old_flat)

        X_next, _ = env(X[h], U[h])
        X[h + 1] = X_next.unflatten(X_next.flatten() + D[h])
        # else is perhaps not needed
        # else:
        #     X[h + 1] = X_old[h + 1] + F[h][0] @ (X[h] - X_old[h]) + F[h][1] @ (U[h] - U_old[h])
        cost += cost_func(X[h], U[h], env)

    return X, U, cost


## Some Legacy stuff - likely not needed
def iLC_open(
    env_true, env_sim, cost_func, U, T, rollin="non", backtracking=True, verbose=True, alpha=1.0
):
    r = 1
    X, U, c = rollout(env_true, cost_func, U)
    assert rollin in ["g", "lin", "non"]
    for t in range(T):
        D, F, C = compute_ders(env_sim, cost_func, X, U)
        k, K = LQR(F, C)
        if backtracking:
            for alphaC in alpha * 1.1 * 1.1 ** (-jnp.arange(10) ** 2):
                r += 1

                # The choice here is between rolling out on LIN(g)+OFF vs g+DEV vs g.
                if rollin == "lin":
                    _, UC, _ = rollout(env_sim, cost_func, U, k, K, X, alphaC, D, F)
                elif rollin == "non":
                    _, UC, _ = rollout(env_sim, cost_func, U, k, K, X, alphaC, D)
                else:
                    _, UC, _ = rollout(env_sim, cost_func, U, k, K, X, alphaC)

                XC, UC, cC = rollout(env_true, cost_func, UC)
                if cC < c:
                    X, U, c, alpha = XC, UC, cC, alphaC
                    if verbose:
                        print(f"iLC (open+{rollin}): t = {t}, r = {r}, c = {c}, alpha = {alpha}")
                    break
        else:
            r += 1

            # The choice here is between rolling out on LIN(g)+OFF vs g+DEV vs g.

            # Lin case needs to be omitted
            # if rollin == "lin":
            #     _, UC, _ = rollout(env_sim, cost_func, U, k, K, X, alpha, D, F)
            if rollin == "non":
                _, UC, _ = rollout_with_lin_shift(env_sim, cost_func, U, D, k, K, X, alpha)
            else:
                _, UC, _ = rollout(env_sim, cost_func, U, k, K, X, alpha)

            X, U, c = rollout(env_true, cost_func, UC)
    return X, U, None, None, c