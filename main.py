import pandas as pd
import numpy as np
from model import differential_evolution, model
import matplotlib.pyplot as plt


# load data
df = pd.read_csv("coupled_damped_oscillators.csv")

t = df["time"].to_numpy()
y = df["x1"].to_numpy()

# basic stats
y_max = np.max(np.abs(y))
y_min = np.min(y)
y_max_val = np.max(y)

bounds = [
    (-2 * y_max,  2 * y_max),     # A
    (0.0, 5.0),                   # lambda
    (0.0, 20.0),                  # omega
    (-np.pi, np.pi),              # phi
    (y_min, y_max_val)            # c
]


best_theta, best_mse, history = differential_evolution(
    t=t,
    y=y,
    bounds=bounds,
    NP=50,
    G=300,
    F=0.8,
    CR=0.9
)

print("Best parameters:", best_theta)
print("Best MSE:", best_mse)

y_hat = model(t, best_theta)

plt.figure(figsize=(10,5))
plt.plot(t, y, label="Date reale x1(t)")
plt.plot(t, y_hat, label="Aproximare DE", linestyle="--")
plt.legend()
plt.xlabel("time")
plt.ylabel("x1")
plt.title("Aproximare oscilație amortizată folosind DE")
plt.show()

plt.figure()
plt.plot(history)
plt.xlabel("Generație")
plt.ylabel("Best MSE")
plt.title("Convergența algoritmului DE")
plt.show()
input('Apasa enter pentru iesire..')
