import numpy as np

def simulate_mass_spring_damper_pid(theta, T=8.0, dt=0.01, x_ref=1.0,
                                    m=1.0, b=0.4, k=4.0,
                                    u_max=20.0):
    Kp, Ki, Kd = theta

    steps = int(T / dt)
    x = 0.0
    v = 0.0
    integ = 0.0
    prev_e = x_ref - x

    xs = np.zeros(steps)
    us = np.zeros(steps)

    for i in range(steps):
        e = x_ref - x
        integ += e * dt
        deriv = (e - prev_e) / dt
        prev_e = e

        u = Kp * e + Ki * integ + Kd * deriv
        u = float(np.clip(u, -u_max, u_max))

        a = (u - b * v - k * x) / m
        v = v + a * dt
        x = x + v * dt

        xs[i] = x
        us[i] = u

        if not np.isfinite(x) or abs(x) > 1e6:
            return None, None

    return xs, us

def pid_fitness(theta, T=8.0, dt=0.01, x_ref=1.0,
                m=1.0, b=0.4, k=4.0, u_max=20.0):
    xs, us = simulate_mass_spring_damper_pid(theta, T=T, dt=dt, x_ref=x_ref, m=m, b=b, k=k, u_max=u_max)
    if xs is None:
        return 1e12

    e = x_ref - xs

    # error term (ISE)
    err_term = float(np.sum(e**2) * dt)

    # control effort penalty
    lam_u = 1e-3
    u_term = float(lam_u * np.sum(us**2) * dt)

    # overshoot penalty
    overshoot = float(max(0.0, np.max(xs) - x_ref))
    o_max = 0.1
    lam_o = 100.0
    o_term = float(lam_o * max(0.0, overshoot - o_max)**2)

    return err_term + u_term + o_term
