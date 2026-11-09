import numpy as np

def EKF_SLAM_known_correspondences(mu_prev, Sigma_prev, u_t, z_t, seen, alphas, dt, sigma_r, sigma_phi, sigma_s):
    """
    Parameters
    ----------
    mu_prev : np.ndarray, shape (3 + 3N,)
        Previous state mu_{t-1} = [x, y, theta, m_1x, m_1y, m_1s, ..., m_Nx, m_Ny, m_Ns].
    Sigma_prev : np.ndarray, shape (3 + 3N, 3 + 3N)
        Covariance of the previous state, Sigma_{t-1}.
    u_t : array-like, shape (2,)
        Control input [v, w] (m/s, rad/s).
    z_t : list of array-like, each shape (3,)
        Landmark observations [r, phi, j]: range (m), bearing (rad), landmark index.
        j is both the correspondence c_t^i and the signature s_t^i.
    seen : np.ndarray, shape (N,), dtype bool
        True for landmarks that have been observed before.
    alphas : array-like, shape (4,)
        Motion noise parameters [a1, a2, a3, a4].
    dt : float
        Time step (s).
    sigma_r : float
        Range measurement noise std (m).
    sigma_phi : float
        Bearing measurement noise std (rad).
    sigma_s : float
        Signature measurement noise std.

    Returns
    -------
    mu : np.ndarray, shape (3 + 3N,)
        Updated state mu_t.
    Sigma : np.ndarray, shape (3 + 3N, 3 + 3N)
        Updated covariance Sigma_t.
    seen : np.ndarray, shape (N,), dtype bool
        Updated list of observed landmarks.
    """
    v, w = u_t
    a1, a2, a3, a4 = alphas
    n = mu_prev.size
    theta = mu_prev[2]

    F_x = np.zeros((3, n))
    F_x[:3, :3] = np.eye(3)

    if abs(w) > 1e-6:
        motion = np.array([
            -v / w * np.sin(theta) + v / w * np.sin(theta + w * dt),
            v / w * np.cos(theta) - v / w * np.cos(theta + w * dt),
            w * dt])
        g = np.array([
            [0, 0, -v / w * np.cos(theta) + v / w * np.cos(theta + w * dt)],
            [0, 0, -v / w * np.sin(theta) + v / w * np.sin(theta + w * dt)],
            [0, 0, 0]])
        V = np.array([
            [(-np.sin(theta) + np.sin(theta + w * dt)) / w,
             v * (np.sin(theta) - np.sin(theta + w * dt)) / w**2 + v * np.cos(theta + w * dt) * dt / w],
            [(np.cos(theta) - np.cos(theta + w * dt)) / w,
             -v * (np.cos(theta) - np.cos(theta + w * dt)) / w**2 + v * np.sin(theta + w * dt) * dt / w],
            [0, dt]])
    else:
        motion = np.array([v * dt * np.cos(theta), v * dt * np.sin(theta), 0.0])
        g = np.array([
            [0, 0, -v * dt * np.sin(theta)],
            [0, 0, v * dt * np.cos(theta)],
            [0, 0, 0]])
        V = np.array([
            [dt * np.cos(theta), -0.5 * v * dt**2 * np.sin(theta)],
            [dt * np.sin(theta), 0.5 * v * dt**2 * np.cos(theta)],
            [0, dt]])

    M = np.diag([a1 * v**2 + a2 * w**2, a3 * v**2 + a4 * w**2])
    R = V @ M @ V.T

    mu_bar = mu_prev + F_x.T @ motion
    mu_bar[2] = np.arctan2(np.sin(mu_bar[2]), np.cos(mu_bar[2]))
    G = np.eye(n) + F_x.T @ g @ F_x
    Sigma_bar = G @ Sigma_prev @ G.T + F_x.T @ R @ F_x

    Q = np.diag([sigma_r**2, sigma_phi**2, sigma_s**2])

    seen = seen.copy()
    for z in z_t:
        r, phi, s = z
        j = int(s)
        idx = 3 + 3 * j

        if not seen[j]:
            mu_bar[idx] = mu_bar[0] + r * np.cos(phi + mu_bar[2])
            mu_bar[idx + 1] = mu_bar[1] + r * np.sin(phi + mu_bar[2])
            mu_bar[idx + 2] = s
            seen[j] = True

        delta = np.array([mu_bar[idx] - mu_bar[0], mu_bar[idx + 1] - mu_bar[1]])
        q = delta.T @ delta
        sq = np.sqrt(q)
        z_hat = np.array([sq, np.arctan2(delta[1], delta[0]) - mu_bar[2], mu_bar[idx + 2]])

        F_xj = np.zeros((6, n))
        F_xj[:3, :3] = np.eye(3)
        F_xj[3:, idx:idx + 3] = np.eye(3)

        H = 1 / q * np.array([
            [-sq * delta[0], -sq * delta[1], 0, sq * delta[0], sq * delta[1], 0],
            [delta[1], -delta[0], -q, -delta[1], delta[0], 0],
            [0, 0, 0, 0, 0, q]]) @ F_xj

        K = Sigma_bar @ H.T @ np.linalg.inv(H @ Sigma_bar @ H.T + Q)

        innovation = np.array([r, phi, s]) - z_hat
        innovation[1] = np.arctan2(np.sin(innovation[1]), np.cos(innovation[1]))

        mu_bar = mu_bar + K @ innovation
        mu_bar[2] = np.arctan2(np.sin(mu_bar[2]), np.cos(mu_bar[2]))
        Sigma_bar = (np.eye(n) - K @ H) @ Sigma_bar

    return mu_bar, Sigma_bar, seen

def EKF_SLAM(mu_prev, Sigma_prev, u_t, z_t, alphas, dt, sigma_r, sigma_phi, sigma_s, alpha_new, Sigma_new=1e6):
    """
    Parameters
    ----------
    mu_prev : np.ndarray, shape (3 + 3N,)
        Previous state mu_{t-1} = [x, y, theta, m_1x, m_1y, m_1s, ..., m_Nx, m_Ny, m_Ns].
        N is the number of landmarks found so far, N_{t-1}.
    Sigma_prev : np.ndarray, shape (3 + 3N, 3 + 3N)
        Covariance of the previous state, Sigma_{t-1}.
    u_t : array-like, shape (2,)
        Control input [v, w] (m/s, rad/s).
    z_t : list of array-like, each shape (3,)
        Landmark observations [r, phi, s]: range (m), bearing (rad), signature.
    alphas : array-like, shape (4,)
        Motion noise parameters [a1, a2, a3, a4].
    dt : float
        Time step (s).
    sigma_r : float
        Range measurement noise std (m).
    sigma_phi : float
        Bearing measurement noise std (rad).
    sigma_s : float
        Signature measurement noise std.
    alpha_new : float
        Mahalanobis distance threshold for creating a new landmark, alpha.
    Sigma_new : float
        Initial variance of a new landmark, stands in for infinity.

    Returns
    -------
    mu : np.ndarray, shape (3 + 3N_t,)
        Updated state mu_t.
    Sigma : np.ndarray, shape (3 + 3N_t, 3 + 3N_t)
        Updated covariance Sigma_t.
    """
    v, w = u_t
    a1, a2, a3, a4 = alphas
    n = mu_prev.size
    theta = mu_prev[2]

    F_x = np.zeros((3, n))
    F_x[:3, :3] = np.eye(3)

    if abs(w) > 1e-6:
        motion = np.array([
            -v / w * np.sin(theta) + v / w * np.sin(theta + w * dt),
            v / w * np.cos(theta) - v / w * np.cos(theta + w * dt),
            w * dt])
        g = np.array([
            [0, 0, -v / w * np.cos(theta) + v / w * np.cos(theta + w * dt)],
            [0, 0, -v / w * np.sin(theta) + v / w * np.sin(theta + w * dt)],
            [0, 0, 0]])
        V = np.array([
            [(-np.sin(theta) + np.sin(theta + w * dt)) / w,
             v * (np.sin(theta) - np.sin(theta + w * dt)) / w**2 + v * np.cos(theta + w * dt) * dt / w],
            [(np.cos(theta) - np.cos(theta + w * dt)) / w,
             -v * (np.cos(theta) - np.cos(theta + w * dt)) / w**2 + v * np.sin(theta + w * dt) * dt / w],
            [0, dt]])
    else:
        motion = np.array([v * dt * np.cos(theta), v * dt * np.sin(theta), 0.0])
        g = np.array([
            [0, 0, -v * dt * np.sin(theta)],
            [0, 0, v * dt * np.cos(theta)],
            [0, 0, 0]])
        V = np.array([
            [dt * np.cos(theta), -0.5 * v * dt**2 * np.sin(theta)],
            [dt * np.sin(theta), 0.5 * v * dt**2 * np.cos(theta)],
            [0, dt]])

    M = np.diag([a1 * v**2 + a2 * w**2, a3 * v**2 + a4 * w**2])
    R = V @ M @ V.T

    mu_bar = mu_prev + F_x.T @ motion
    mu_bar[2] = np.arctan2(np.sin(mu_bar[2]), np.cos(mu_bar[2]))
    G = np.eye(n) + F_x.T @ g @ F_x
    Sigma_bar = G @ Sigma_prev @ G.T + F_x.T @ R @ F_x

    Q = np.diag([sigma_r**2, sigma_phi**2, sigma_s**2])

    for z in z_t:
        z = np.asarray(z, dtype=float)
        r, phi, s = z
        N = (mu_bar.size - 3) // 3

        mu_ext = np.concatenate([mu_bar, [
            mu_bar[0] + r * np.cos(phi + mu_bar[2]),
            mu_bar[1] + r * np.sin(phi + mu_bar[2]),
            s]])
        Sigma_ext = np.zeros((mu_ext.size, mu_ext.size))
        Sigma_ext[:-3, :-3] = Sigma_bar
        Sigma_ext[-3:, -3:] = np.eye(3) * Sigma_new
        n_ext = mu_ext.size

        H_k = []
        Psi_k = []
        z_hat_k = []
        pi_k = []
        for k in range(N + 1):
            idx = 3 + 3 * k
            delta = np.array([mu_ext[idx] - mu_ext[0], mu_ext[idx + 1] - mu_ext[1]])
            q = delta.T @ delta
            sq = np.sqrt(q)
            z_hat = np.array([sq, np.arctan2(delta[1], delta[0]) - mu_ext[2], mu_ext[idx + 2]])

            F_xk = np.zeros((6, n_ext))
            F_xk[:3, :3] = np.eye(3)
            F_xk[3:, idx:idx + 3] = np.eye(3)

            H = 1 / q * np.array([
                [-sq * delta[0], -sq * delta[1], 0, sq * delta[0], sq * delta[1], 0],
                [delta[1], -delta[0], -q, -delta[1], delta[0], 0],
                [0, 0, 0, 0, 0, q]]) @ F_xk

            Psi = H @ Sigma_ext @ H.T + Q

            innovation = z - z_hat
            innovation[1] = np.arctan2(np.sin(innovation[1]), np.cos(innovation[1]))

            H_k.append(H)
            Psi_k.append(Psi)
            z_hat_k.append(z_hat)
            pi_k.append(innovation.T @ np.linalg.inv(Psi) @ innovation)

        pi_k[N] = alpha_new
        j = int(np.argmin(pi_k))

        if j == N:
            mu_bar = mu_ext
            Sigma_bar = Sigma_ext
            H = H_k[j]
        else:
            H = H_k[j][:, :-3]

        K = Sigma_bar @ H.T @ np.linalg.inv(Psi_k[j])

        innovation = z - z_hat_k[j]
        innovation[1] = np.arctan2(np.sin(innovation[1]), np.cos(innovation[1]))

        mu_bar = mu_bar + K @ innovation
        mu_bar[2] = np.arctan2(np.sin(mu_bar[2]), np.cos(mu_bar[2]))
        Sigma_bar = (np.eye(mu_bar.size) - K @ H) @ Sigma_bar

    return mu_bar, Sigma_bar