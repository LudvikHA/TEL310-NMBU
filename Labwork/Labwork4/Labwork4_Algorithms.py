import numpy as np
import Labwork1.Labwork1 as L1
import Labwork2.Labwork2_Algorithms as L2
from statistics import NormalDist
from math import erf, sqrt



rng = np.random.default_rng()

def grid_localization(p_prev, control_v, control_w, z_t, world_map, map_x, map_y, map_theta,
                      alphas, dt, sigma_r, sigma_phi):
    """
    Parameters
    ----------
    p_prev : np.ndarray, shape (Nx, Ny, Ntheta)
        Previous belief p_{k,t-1}, one probability per grid cell.
    control_v : float
        Control input v (m/s).
    control_w : float
        Control input w (rad/s).
    z_t : list of array-like
        Landmark measurements from get_landmark_measurements(),
        each [r, phi, j]: range (m), bearing (rad), landmark index.
    world_map : dict
        Known map m, same as passed to landmark_model_known_correspondance.
    map_x : np.ndarray, shape (Nx,)
        Cell centre x-coordinates (m).
    map_y : np.ndarray, shape (Ny,)
        Cell centre y-coordinates (m).
    map_theta : np.ndarray, shape (Ntheta,)
        Cell centre headings (rad).
    alphas : array-like, shape (6,)
        Motion noise parameters [a1, a2, a3, a4, a5, a6].
    dt : float
        Time since the last update (s).
    sigma_r : float
        Range measurement noise std used by the filter (m).
    sigma_phi : float
        Bearing measurement noise std used by the filter (rad).

    Returns
    -------
    p : np.ndarray, shape (Nx, Ny, Ntheta)
        Updated belief p_{k,t}, normalized.
    """
    u_t = np.array([control_v, control_w])
    robot_params = (dt, *alphas)

    standing_still = abs(control_v) < 1e-6 and abs(control_w) < 1e-6
    uniform = np.allclose(p_prev, p_prev.flat[0])

    if standing_still or uniform:
        p_bar = p_prev.copy()
    else:
        p_bar = np.zeros(p_prev.shape)
        skip_below = p_prev.max() * 1e-6

        cell_size = map_x[1] - map_x[0]
        reach = int(np.ceil(abs(control_v) * dt / cell_size)) + 1

        for ix, x_i in enumerate(map_x):
            for iy, y_i in enumerate(map_y):
                for ith, th_i in enumerate(map_theta):
                    p_i = p_prev[ix, iy, ith]
                    if p_i < skip_below:
                        continue

                    source_pose = np.array([x_i, y_i, th_i])

                    for kx in range(max(0, ix - reach), min(map_x.size, ix + reach + 1)):
                        for ky in range(max(0, iy - reach), min(map_y.size, iy + reach + 1)):
                            for kth, th_k in enumerate(map_theta):
                                th_k_unwrapped = th_i + np.arctan2(np.sin(th_k - th_i), np.cos(th_k - th_i))
                                target_pose = np.array([map_x[kx], map_y[ky], th_k_unwrapped])

                                p_bar[kx, ky, kth] += p_i * L1.motion_model_velocity(source_pose, target_pose, u_t, robot_params)

    likelihood = np.ones(p_prev.shape)
    if len(z_t) > 0:
        for ix, x in enumerate(map_x):
            for iy, y in enumerate(map_y):
                for ith, theta in enumerate(map_theta):
                    pose = np.array([x, y, theta])
                    for f in z_t:
                        likelihood[ix, iy, ith] *= L2.landmark_model_known_correspondance(f, pose, world_map, sigma_r, sigma_phi)

    p = p_bar * likelihood

    total = p.sum()
    if total > 0 and np.isfinite(total):
        return p / total

    print("Grid localization: belief became zero, relocalizing from measurements")
    total = likelihood.sum()
    if total > 0 and np.isfinite(total):
        return likelihood / total

    return np.ones(p_prev.shape) / p_prev.size

def MCL(X_prev, control_v, control_w, z_t, world_map, alphas, dt, sigma_r, sigma_phi, MCL_interval):
    """
    Parameters
    ----------
    X_prev : np.ndarray, shape (M, 3)
        Previous particle set X_{t-1}, one pose [x, y, theta] per row.
    control_v : float
        Control input v (m/s).
    control_w : float
        Control input w (rad/s).
    z_t : list of array-like
        Landmark measurements from get_landmark_measurements(),
        each [r, phi, j]: range (m), bearing (rad), landmark index.
    world_map : dict
        Known map m, same as passed to landmark_model_known_correspondance.
    alphas : array-like, shape (6,)
        Motion noise parameters [a1, a2, a3, a4, a5, a6].
    dt : float
        Time since the last update (s).
    sigma_r : float
        Range measurement noise std used by the filter (m).
    sigma_phi : float
        Bearing measurement noise std used by the filter (rad).

    Returns
    -------
    X : np.ndarray, shape (M, 3)
        New particle set X_t after resampling.
    """
    M = X_prev.shape[0]
    u_t = np.array([control_v, control_w])
    robot_params = (dt, *alphas)

    X_bar = np.zeros((M, 3))
    w = np.ones(M)

    for m in range(M):
        x_m = L1.sample_motion_model_velocity(X_prev[m], u_t, robot_params, sample_interval=MCL_interval)
        x_m[2] = np.arctan2(np.sin(x_m[2]), np.cos(x_m[2]))
        X_bar[m] = x_m

        for f in z_t:
            w[m] *= L2.landmark_model_known_correspondance(f, x_m, world_map, sigma_r, sigma_phi)

    if len(z_t) == 0:
        return X_bar

    total = w.sum()
    if total == 0 or not np.isfinite(total):
        print("MCL: all particle weights are zero, skipping resampling")
        return X_bar

    w = w / total

    X = np.zeros((M, 3))
    r = rng.uniform(0, 1 / M)
    c = w[0]
    i = 0
    for m in range(M):
        U = r + m / M
        while U > c:
            i += 1
            c += w[i]
        X[m] = X_bar[i]

    return X

def augmented_MCL(X_prev, control_v, control_w, z_t, world_map, alphas, dt, sigma_r, sigma_phi,
                  w_slow, w_fast, alpha_slow, alpha_fast, map_limits, MCL_interval):
    """
    Parameters
    ----------
    X_prev : np.ndarray, shape (M, 3)
        Previous particle set X_{t-1}, one pose [x, y, theta] per row.
    control_v : float
        Control input v (m/s).
    control_w : float
        Control input w (rad/s).
    z_t : list of array-like
        Landmark measurements from get_landmark_measurements(),
        each [r, phi, j]: range (m), bearing (rad), landmark index.
    world_map : dict
        Known map m, same as passed to landmark_model_known_correspondance.
    alphas : array-like, shape (6,)
        Motion noise parameters [a1, a2, a3, a4, a5, a6].
    dt : float
        Time since the last update (s).
    sigma_r : float
        Range measurement noise std used by the filter (m).
    sigma_phi : float
        Bearing measurement noise std used by the filter (rad).
    w_slow : float
        Long-term average of the measurement likelihood (kept between calls, start at 0).
    w_fast : float
        Short-term average of the measurement likelihood (kept between calls, start at 0).
    alpha_slow : float
        Decay rate of w_slow, 0 <= alpha_slow << alpha_fast.
    alpha_fast : float
        Decay rate of w_fast.
    map_limits : array-like, shape (4,)
        [x_min, x_max, y_min, y_max] (m), area where random particles are added.

    Returns
    -------
    X : np.ndarray, shape (M, 3)
        New particle set X_t.
    w_slow : float
        Updated w_slow, pass in on the next call.
    w_fast : float
        Updated w_fast, pass in on the next call.
    """
    M = X_prev.shape[0]
    u_t = np.array([control_v, control_w])
    robot_params = (dt, *alphas)

    X_bar = np.zeros((M, 3))
    w = np.ones(M)

    for m in range(M):
        x_m = L1.sample_motion_model_velocity(X_prev[m], u_t, robot_params, sample_interval=MCL_interval)
        x_m[2] = np.arctan2(np.sin(x_m[2]), np.cos(x_m[2]))
        X_bar[m] = x_m

        for f in z_t:
            w[m] *= L2.landmark_model_known_correspondance(f, x_m, world_map, sigma_r, sigma_phi)

    if len(z_t) == 0:
        return X_bar, w_slow, w_fast

    w_avg = (w ** (1 / len(z_t))).mean()

    w_slow = w_slow + alpha_slow * (w_avg - w_slow)
    w_fast = w_fast + alpha_fast * (w_avg - w_fast)

    if w_slow > 0:
        p_random = max(0.0, 1.0 - w_fast / w_slow)
    else:
        p_random = 0.0

    total = w.sum()
    if total == 0 or not np.isfinite(total):
        p_random = 1.0
        w = np.ones(M) / M
    else:
        w = w / total

    X = np.zeros((M, 3))
    x_min, x_max, y_min, y_max = map_limits
    for m in range(M):
        if rng.uniform() < p_random:
            X[m] = [rng.uniform(x_min, x_max), rng.uniform(y_min, y_max), rng.uniform(-np.pi, np.pi)]
        else:
            i = rng.choice(M, p=w)
            X[m] = X_bar[i]

    return X, w_slow, w_fast

def KLD_sampling_MCL(X_prev, w_prev, control_v, control_w, z_t, world_map, alphas, time_step, interval,
                     sigma_r, sigma_phi, epsilon, delta, bin_size, M_min, M_max):
    """
    Parameters
    ----------
    X_prev : np.ndarray, shape (M_prev, 3)
        Previous particle set X_{t-1}, one pose [x, y, theta] per row.
    w_prev : np.ndarray, shape (M_prev,)
        Weights of the previous particles.
    control_v : float
        Control input v (m/s).
    control_w : float
        Control input w (rad/s).
    z_t : list of array-like
        Landmark measurements from get_landmark_measurements(),
        each [r, phi, j]: range (m), bearing (rad), landmark index.
    world_map : dict
        Known map m, same as passed to landmark_model_known_correspondance.
    alphas : array-like, shape (6,)
        Motion noise parameters [a1, a2, a3, a4, a5, a6].
    time_step : float
        Simulator time step (s).
    interval : int
        Number of simulator steps since the last update.
    sigma_r : float
        Range measurement noise std used by the filter (m).
    sigma_phi : float
        Bearing measurement noise std used by the filter (rad).
    epsilon : float
        Maximum allowed KL-divergence between the particle set and the true posterior.
    delta : float
        Probability that the error exceeds epsilon.
    bin_size : array-like, shape (3,)
        Histogram bin size [x (m), y (m), theta (rad)].
    M_min : int
        Minimum number of particles, M_chi_min.
    M_max : int
        Maximum number of particles.

    Returns
    -------
    X : np.ndarray, shape (M, 3)
        New particle set X_t.
    w : np.ndarray, shape (M,)
        Weights of the new particles.
    """
    u_t = np.array([control_v, control_w])
    robot_params = (time_step, *alphas)
    z_quantile = NormalDist().inv_cdf(1 - delta)

    total = w_prev.sum()
    if total == 0 or not np.isfinite(total):
        cdf = np.arange(1, len(w_prev) + 1) / len(w_prev)
    else:
        cdf = np.cumsum(w_prev / total)
    cdf[-1] = 1.0

    X = []
    w = []
    bins = set()
    M = 0
    M_chi = 0
    k = 0

    while (M < M_chi or M < M_min) and M < M_max:
        i = np.searchsorted(cdf, rng.uniform())
        x_m = L1.sample_motion_model_velocity(X_prev[i], u_t, robot_params, interval)
        x_m[2] = np.arctan2(np.sin(x_m[2]), np.cos(x_m[2]))

        w_m = 1.0
        for f in z_t:
            w_m *= L2.landmark_model_known_correspondance(f, x_m, world_map, sigma_r, sigma_phi)

        X.append(x_m)
        w.append(w_m)

        b = (int(np.floor(x_m[0] / bin_size[0])), int(np.floor(x_m[1] / bin_size[1])), int(np.floor(x_m[2] / bin_size[2])))
        if b not in bins:
            k += 1
            bins.add(b)
            if k > 1:
                M_chi = (k - 1) / (2 * epsilon) * (1 - 2 / (9 * (k - 1)) + np.sqrt(2 / (9 * (k - 1))) * z_quantile) ** 3

        M += 1

    return np.array(X), np.array(w)

def test_range_measurement(z_k, z_star, z_max, intrinsic_params, chi):
    """
    Parameters
    ----------
    z_k : float
        Measured range of beam k (m).
    z_star : np.ndarray, shape (M,)
        Expected range of beam k from ray casting at each particle pose x_t^[m] (m).
    z_max : float
        Maximum sensor range (m).
    intrinsic_params : array-like, shape (6,)
        Beam model parameters [z_hit, z_short, z_max, z_rand, sigma_hit, lambda_short].
    chi : float
        Threshold on the probability that the measurement is short, 0 < chi < 1.

    Returns
    -------
    accept : bool
        True if the measurement is accepted, False if it is rejected.
    """
    w_hit, w_short, w_max, w_rand, sigma_hit, lambda_short = intrinsic_params

    p = 0.0
    q = 0.0
    for z_s in z_star:
        if 0 <= z_k <= z_max:
            eta_hit = 1 / (0.5 * (erf((z_max - z_s) / (sigma_hit * sqrt(2))) - erf(-z_s / (sigma_hit * sqrt(2)))))
            p_hit = eta_hit * np.exp(-0.5 * (z_k - z_s) ** 2 / sigma_hit ** 2) / np.sqrt(2 * np.pi * sigma_hit ** 2)
        else:
            p_hit = 0.0

        if 0 <= z_k <= z_s and z_s > 0:
            p_short = lambda_short * np.exp(-lambda_short * z_k) / (1 - np.exp(-lambda_short * z_s))
        else:
            p_short = 0.0

        p_max = 1.0 if z_k >= z_max else 0.0
        p_rand = 1 / z_max if 0 <= z_k < z_max else 0.0

        p += w_short * p_short
        q += w_hit * p_hit + w_short * p_short + w_max * p_max + w_rand * p_rand

    if q == 0 or p / q <= chi:
        return True
    return False
