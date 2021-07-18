import time
import jax.numpy as jnp
from deluca.igpc.rollout import rollout, compute_ders
from deluca.igpc.lqr_solver import LQR
import jax


def counterfact_loss(
    E, off, W, H, M, x, t, env_sim, cost_func, U_old, k, K, X_old, D, F, w_is, alpha, C
):
    y, cost = x, 0
    for h in range(H):
        u_delta = jnp.tensordot(E, W[h : h + M], axes=([0, 2], [0, 1])) + off
        u = (
            U_old[t + h]
            + alpha * k[t + h]
            + K[t + h] @ (y.flatten() - X_old[t + h].flatten())
            + C * u_delta
        )
        cost = cost_func(y, u, env_sim)

        new_state, _ = env_sim(y, u)
        if w_is == "de":
            y = y.unflatten(new_state.flatten() + W[h + M])
        elif w_is == "dede":
            y = y.unflatten(new_state.flatten() + D[h + M] + W[h + M])
        else:
            y = y.unflatten(
                X_old[h + M + 1].flatten()
                + F[h + M][0] @ (y.flatten() - X_old[h + M].flatten())
                + F[h + M][1] @ (u - U_old[h + M])
                + W[h + M]
            )
    return cost


grad_loss = jax.grad(counterfact_loss, (0, 1))


def GPC_rollout(env_true, env_sim, cost_func, U_old, k, K, X_old, D, F, alpha, w_is, ref_lr=0.1):
    # print(ref_lr)
    H, M, lr, C = 3, 3, ref_lr, 1.0
    X, U, U_delta = (
        [None for _ in range(env_sim.H + 1)],
        [None for _ in range(env_sim.H)],
        [None for _ in range(env_sim.H)],
    )
    W, E, off = (
        jnp.zeros((H + M, env_sim.state_dim)),
        jnp.zeros((H, env_sim.action_dim, env_sim.state_dim)),
        jnp.zeros(env_sim.action_dim),
    )
    X[0], cost = env_true.init(), 0
    for h in range(env_true.H):
        U_delta[h] = jnp.tensordot(E, W[-M:], axes=([0, 2], [0, 1])) + off
        U[h] = (
            U_old[h] + alpha * k[h] + K[h] @ (X[h].flatten() - X_old[h].flatten()) + C * U_delta[h]
        )
        X[h + 1], _ = env_true(X[h], U[h])
        cost += cost_func(X[h], U[h], env_true)

        # 3 choices for w are f-g, (f-g)-(f-g last time), (f- LIN(g) with OFF).
        new_state, _ = env_sim(X[h], U[h])
        if w_is == "de":
            w = X[h + 1].flatten() - new_state.flatten()
        elif w_is == "dede":
            w = (X[h + 1].flatten() - new_state.flatten()) - D[h]
        else:
            w = X[h + 1].flatten() - (
                X_old[h + 1].flatten()
                + F[h][0] @ (X[h].flatten() - X_old[h].flatten())
                + F[h][1] @ (U[h] - U_old[h])
            )

        W = jax.ops.index_update(W, 0, w)
        W = jnp.roll(W, -1, axis=0)
        if h >= H:
            delta_E, delta_off = grad_loss(
                E,
                off,
                W,
                H,
                M,
                X[h - H],
                h - H,
                env_sim,
                cost_func,
                U_old,
                k,
                K,
                X_old,
                D,
                F,
                w_is,
                alpha,
                C,
            )
            E -= lr / h * delta_E
            off -= lr / h * delta_off
    # print(norm_U_old, norm_other, norm_off, norm_U_delta, norm_U_vals)
    return X, U, cost


def iGPC_closed(
    env_true,
    env_sim,
    cost_func,
    U,
    T,
    k=None,
    K=None,
    X=None,
    w_is="de",
    lr=None,
    ref_alpha=1.0,
    backtracking=True,
    verbose=True,
):
    alpha, r = ref_alpha, 1
    X, U, c = rollout(env_true, cost_func, U, k, K, X)
    print(f"iGPC: t = {-1}, r = {r}, c = {c}")
    assert w_is in ["de", "dede", "delin"]
    prev_loop_fail = False
    for t in range(T):
        D, F, C = compute_ders(env_sim, cost_func, X, U)
        k, K = LQR(F, C)
        if backtracking:
            if prev_loop_fail:
                print("dropping LR and alpha")
                lr, alpha = lr / 100, alpha

            prev_loop_fail = True
            for alphaC in alpha * 1.1 * 1.1 ** (-jnp.arange(6) ** 2):
                print("Trying alpha ", alphaC)
                r += 1
                # if c < 500:
                #     print('cutting lr')
                #     lr = 0.0
                XC, UC, cC = GPC_rollout(
                    env_true, env_sim, cost_func, U, k, K, X, D, F, alphaC, w_is, lr
                )
                if cC < c:
                    X, U, c, alpha = XC, UC, cC, alphaC
                    if verbose:
                        print(f"iGPC: t = {t}, r = {r}, c = {c}, alpha = {alpha}, lr = {lr}")
                    prev_loop_fail = False
                    break
        else:
            r += 1
            X, U, c = GPC_rollout(env_true, env_sim, cost_func, U, k, K, X, D, F, alpha, w_is, lr)
            if verbose:
                print(f"iGPC: t = {t}, r = {r}, c = {c}")
    return X, U, k, K, c

