import numpy as np
import pandas as pd

def model(t, theta):
    A, lam, w, phi, c = theta
    return A * np.exp(-lam * t) * np.sin(w * t + phi) + c

def mse_fitness(theta, t, y):
    y_hat = model(t, theta)
    return np.mean((y - y_hat) ** 2)

def ensure_bounds(vec, bounds):
    # bounds: list of (low, high)
    v = vec.copy()
    for j, (lo, hi) in enumerate(bounds):
        if v[j] < lo: v[j] = lo
        if v[j] > hi: v[j] = hi
    return v

def differential_evolution(t, y, bounds, NP=50, G=300, F=0.8, CR=0.9, seed=42):
    rng = np.random.default_rng(seed)
    D = len(bounds)

    # Init population uniformly in bounds (as in course)
    pop = np.array([
        [rng.uniform(lo, hi) for (lo, hi) in bounds]
        for _ in range(NP)
    ], dtype=float)

    fitness = np.array([mse_fitness(ind, t, y) for ind in pop])
    best_idx = np.argmin(fitness)
    best = pop[best_idx].copy()
    best_fit = fitness[best_idx]

    history = [best_fit]

    for _ in range(G):
        for i in range(NP):
            # pick r1, r2, r3 all different and != i
            idxs = [idx for idx in range(NP) if idx != i]
            r1, r2, r3 = rng.choice(idxs, size=3, replace=False)

            # Mutation: rand/1
            v = pop[r1] + F * (pop[r2] - pop[r3])
            v = ensure_bounds(v, bounds)

            # Binomial crossover with forced j_rand (so at least one gene from v)
            j_rand = rng.integers(0, D)
            u = pop[i].copy()
            for j in range(D):
                if rng.random() < CR or j == j_rand:
                    u[j] = v[j]

            u = ensure_bounds(u, bounds)
            fu = mse_fitness(u, t, y)

            # Selection (minimization)
            if fu <= fitness[i]:
                pop[i] = u
                fitness[i] = fu

                if fu < best_fit:
                    best_fit = fu
                    best = u.copy()

        history.append(best_fit)

    return best, best_fit, np.array(history)
