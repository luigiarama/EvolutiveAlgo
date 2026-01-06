import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model import differential_evolution, model, mse_fitness
from problem2 import pid_fitness, simulate_mass_spring_damper_pid


def run_problem1(csv_path, NP, G, F, CR, seed, save_plots=False):
    df = pd.read_csv(csv_path)
    t = df["time"].to_numpy()
    y = df["x1"].to_numpy()

    y_max = np.max(np.abs(y))
    y_min = np.min(y)
    y_max_val = np.max(y)

    bounds = [
        (-2 * y_max,  2 * y_max),  # A
        (0.0, 5.0),                # lambda
        (0.0, 20.0),               # omega
        (-np.pi, np.pi),           # phi
        (y_min, y_max_val)         # c
    ]

    fitness_fn = lambda theta: mse_fitness(theta, t, y)

    best_theta, best_mse, hist = differential_evolution(
        fitness_fn=fitness_fn,
        bounds=bounds,
        NP=NP, G=G, F=F, CR=CR,
        seed=seed
    )

    print("=== Problema 1 ===")
    print("Best parameters:", best_theta)
    print("Best MSE:", best_mse)

    y_hat = model(t, best_theta)

    plt.figure(figsize=(10, 5))
    plt.plot(t, y, label="Date reale x1(t)")
    plt.plot(t, y_hat, "--", label="Aproximare DE")
    plt.legend()
    plt.xlabel("time")
    plt.ylabel("x1")
    plt.title("Problema 1: Aproximare oscilație amortizată (DE)")
    if save_plots:
        plt.savefig("p1_fit.png", dpi=150)
    plt.show()

    plt.figure()
    plt.plot(hist)
    plt.xlabel("Generație")
    plt.ylabel("Best MSE")
    plt.title("Problema 1: Convergența DE")
    if save_plots:
        plt.savefig("p1_convergence.png", dpi=150)
    plt.show()


def run_problem2(NP, G, F, CR, seed, save_plots=False):
    print(">>> ENTER run_problem2()")
    print(f">>> DE params: NP={NP}, G={G}, F={F}, CR={CR}, seed={seed}")
    bounds = [
        (0.0, 20.0),  # Kp
        (0.0, 50.0),   # Ki
        (0.0, 50.0)    # Kd
    ]

    fitness_fn = lambda theta: pid_fitness(
        theta,
        T=8.0, dt=0.01, x_ref=1.0,
        m=1.0, b=0.4, k=4.0, u_max=20.0
    )
    test_theta = np.array([50.0, 5.0, 5.0])
    print(">>> fitness test:", fitness_fn(test_theta))

    best_pid, bestJ, hist = differential_evolution(
        fitness_fn=fitness_fn,
        bounds=bounds,
        NP=NP, G=G, F=F, CR=CR,
        seed=seed
    )

    print("\n=== Problema 2 ===")
    print("Best PID:", best_pid)
    print("Best J:", bestJ)

    xs, us = simulate_mass_spring_damper_pid(
        best_pid, T=8.0, dt=0.01, x_ref=1.0,
        m=1.0, b=0.4, k=4.0, u_max=20.0
    )

    t = np.arange(len(xs)) * 0.01

    plt.figure(figsize=(10, 5))
    plt.plot(t, xs, label="x(t)")
    plt.plot(t, np.ones_like(t), "--", label="setpoint")
    plt.legend()
    plt.xlabel("t [s]")
    plt.ylabel("x")
    plt.title("Problema 2: Răspuns sistem cu PID optimizat (DE)")
    if save_plots:
        plt.savefig("p2_response.png", dpi=150)
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(t, us)
    plt.xlabel("t [s]")
    plt.ylabel("u(t)")
    plt.title("Problema 2: Semnal de control")
    if save_plots:
        plt.savefig("p2_control.png", dpi=150)
    plt.show()

    plt.figure()
    plt.plot(hist)
    plt.xlabel("Generație")
    plt.ylabel("Best J")
    plt.title("Problema 2: Convergența DE")
    if save_plots:
        plt.savefig("p2_convergence.png", dpi=150)
    plt.show()


def parse_args():
    p = argparse.ArgumentParser(description="Proiect DE - Problema 1 + Problema 2")
    p.add_argument("--problem", choices=["1", "2", "both"], default="both",
                   help="Ce problemă rulezi: 1, 2 sau both")
    p.add_argument("--csv", default="coupled_damped_oscillators.csv",
                   help="Calea către CSV pentru Problema 1")
    p.add_argument("--NP", type=int, default=50, help="Dimensiunea populației")
    p.add_argument("--G", type=int, default=150, help="Număr generații")
    p.add_argument("--F", type=float, default=0.8, help="Factor mutație")
    p.add_argument("--CR", type=float, default=0.9, help="Crossover rate")
    p.add_argument("--seed", type=int, default=42, help="Seed random")
    p.add_argument("--save-plots", action="store_true", help="Salvează graficele ca PNG")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.problem in ("1", "both"):
        run_problem1(args.csv, NP=args.NP, G=args.G, F=args.F, CR=args.CR, seed=args.seed, save_plots=args.save_plots)

    if args.problem in ("2", "both"):
        # mic tweak default: P2 poate merge mai bine cu NP=60, G=250,
        run_problem2(NP=20, G=50, F=args.F, CR=args.CR, seed=args.seed, save_plots=args.save_plots)

    input("Apasa Enter pentru iesire...")
