import numpy as np
import pandas as pd

def model(t, theta):
    A, lam, w, phi, c = theta
    return A * np.exp(-lam * t) * np.sin(w * t + phi) + c

def mse_fitness(theta, t, y):
    y_hat = model(t, theta)
    if np.isnan(y_hat).any() or np.isinf(y_hat).any():
        return 1e12
    return float(np.mean((y - y_hat) ** 2))

def ensure_bounds(vec, bounds):
    # bounds: list of (low, high)
    v = vec.copy()
    for j, (lo, hi) in enumerate(bounds):
        if v[j] < lo: v[j] = lo
        if v[j] > hi: v[j] = hi
    return v

def differential_evolution(fitness_fn, bounds, NP=50, G=300, F=0.8, CR=0.9, seed=42):
    """
    Standard DE/rand/1/bin for minimization.
    fitness_fn: callable(theta) -> float
    bounds: list[(lo, hi)]
    """
    rng = np.random.default_rng(seed)
    D = len(bounds)

    pop = np.array([[rng.uniform(lo, hi) for (lo, hi) in bounds] for _ in range(NP)], dtype=float)
    fitness = np.array([fitness_fn(ind) for ind in pop], dtype=float)

    best_idx = int(np.argmin(fitness))
    best = pop[best_idx].copy()
    best_fit = float(fitness[best_idx])
    history = [best_fit]

    for _ in range(G):
        for i in range(NP):
            idxs = [idx for idx in range(NP) if idx != i]
            r1, r2, r3 = rng.choice(idxs, size=3, replace=False)

            # mutation: rand/1
            v = pop[r1] + F * (pop[r2] - pop[r3])
            v = ensure_bounds(v, bounds)

            # bin crossover + forced j_rand
            j_rand = int(rng.integers(0, D))
            u = pop[i].copy()
            for j in range(D):
                if rng.random() < CR or j == j_rand:
                    u[j] = v[j]

            u = ensure_bounds(u, bounds)
            fu = float(fitness_fn(u))

            # selection (minimization)
            if fu <= fitness[i]:
                pop[i] = u
                fitness[i] = fu
                if fu < best_fit:
                    best_fit = fu
                    best = u.copy()

        history.append(best_fit)

    return best, best_fit, np.array(history, dtype=float)