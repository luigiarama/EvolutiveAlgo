import numpy as np
import pandas as pd

"""
Defineste modelul matematic pentru oscilatia amortizata - P1
"""
def model(t, theta):
    A, lam, w, phi, c = theta
    return A * np.exp(-lam * t) * np.sin(w * t + phi) + c

"""
Calculeaza nota unei solutii folosind Eroarea Medie Patratica
cu cat eroarea este mai mica cu atat solutia este mai buna
"""
def mse_fitness(theta, t, y):
    y_hat = model(t, theta)
    if np.isnan(y_hat).any() or np.isinf(y_hat).any():
        return 1e12
    return float(np.mean((y - y_hat) ** 2))

"""
functie care asigura ca parametrii gasiti nu depasesc limitele fizice impuse
"""
def ensure_bounds(vec, bounds):
    v = vec.copy()
    for j, (lo, hi) in enumerate(bounds):
        if v[j] < lo: v[j] = lo
        if v[j] > hi: v[j] = hi
    return v


"""
Algoritmul de Evolutie Diferentiala

fitness_fn: Functia care evalueaza cat de buna este solutia
bounds: Intervalele de cautare pentru parametri
NP: Numarul de solutii testate simultan
G: Numarul de generatii.
F: Pasul de cautare
CR: Rata de crossover
"""
def differential_evolution(fitness_fn, bounds, NP=50, G=300, F=0.8, CR=0.9, seed=42):
    
    rng = np.random.default_rng(seed)
    D = len(bounds)

    # Initializam populatia cu valori aleatorii intre limite
    pop = np.array([[rng.uniform(lo, hi) for (lo, hi) in bounds] for _ in range(NP)], dtype=float)
    fitness = np.array([fitness_fn(ind) for ind in pop], dtype=float)

    # Identificam cea mai buna valoare initiala
    best_idx = int(np.argmin(fitness))
    best = pop[best_idx].copy()
    best_fit = float(fitness[best_idx])
    history = [best_fit]

    #procesul evolutiv
    for _ in range(G):
        for i in range(NP):
            #mutatia - alegem 3 indivizi aleatorii diferiti de cel curent
            idxs = [idx for idx in range(NP) if idx != i]
            r1, r2, r3 = rng.choice(idxs, size=3, replace=False)

            v = pop[r1] + F * (pop[r2] - pop[r3])
            v = ensure_bounds(v, bounds)

            #crossover, combinam individul curent cu mutantul
            j_rand = int(rng.integers(0, D))
            u = pop[i].copy()
            for j in range(D):
                if rng.random() < CR or j == j_rand:
                    u[j] = v[j]

            u = ensure_bounds(u, bounds)
            fu = float(fitness_fn(u))

            #selectia, pastram noul individ doar daca e mai bun decat cel vechi
            if fu <= fitness[i]:
                pop[i] = u
                fitness[i] = fu
                if fu < best_fit:
                    best_fit = fu
                    best = u.copy()

        history.append(best_fit)

    return best, best_fit, np.array(history, dtype=float)