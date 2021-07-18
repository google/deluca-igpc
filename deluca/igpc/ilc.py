import os
import time
import jax.numpy as jnp
from deluca.igpc.rollout import rollout, compute_ders
from deluca.igpc.lqr_solver import LQR
from PIL import Image
import jax


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
            if rollin == "lin":
                _, UC, _ = rollout(env_sim, cost_func, U, k, K, X, alpha, D, F)
            elif rollin == "non":
                _, UC, _ = rollout(env_sim, cost_func, U, k, K, X, alpha, D)
            else:
                _, UC, _ = rollout(env_sim, cost_func, U, k, K, X, alpha)

            X, U, c = rollout(env_true, cost_func, UC)
    return X, U, None, None, c


def iLC_closed(
    env_true,
    env_sim,
    cost_func,
    U,
    T,
    k=None,
    K=None,
    X=None,
    ref_alpha=1.0,
    verbose=True,
    backtracking=True,
):
    alpha, r = ref_alpha, 1
    X, U, c = rollout(env_true, cost_func, U, k, K, X)
    print(f"iLC: t = {-1}, r = {r}, c = {c}")
    for t in range(T):
        _, F, C = compute_ders(env_sim, cost_func, X, U)
        k, K = LQR(F, C)
        if backtracking:
            for alphaC in alpha * 1.1 * 1.1 ** (-jnp.arange(10) ** 2):
                r += 1
                XC, UC, cC = rollout(env_true, cost_func, U, k, K, X, alphaC)
                if cC <= c:
                    X, U, c, alpha = XC, UC, cC, alphaC
                    if verbose:
                        print(
                            f"iLC (closed+alpha={ref_alpha}): t = {t}, r = {r}, c = {c}, alpha = {alpha}"
                        )
                    break
        else:
            r += 1
            XC, UC, cC = rollout(env_true, cost_func, U, k, K, X, alpha)
            if verbose:
                print(f"iLC (closed+alpha={ref_alpha}): t = {t}, r = {r}, c = {c}, alpha = {alpha}")
            break
    return X, U, k, K, c

