import numpy as np

rng = np.random.default_rng()

# Helper functions
def _g(mu_prev, u_t, dt):
    v = u_t[0]
    omega = u_t[1]

    if abs(omega) < 1e-6: 
        omega = 1e-6 

    theta = mu_prev[2]

    mu_bar = np.zeros(3)
    mu_bar[0] = (-v * np.sin(theta) + v * np.sin(theta + omega * dt))/omega
    mu_bar[1] = (v * np.cos(theta) - v * np.cos(theta + omega * dt))/omega
    mu_bar[2] = omega * dt

    mu_bar = mu_prev + mu_bar
    mu_bar[2] = np.arctan2(np.sin(mu_bar[2]), np.cos(mu_bar[2]))

    return mu_bar


def EKF_localization_known_correspondence(
        mu_prev,
        Sigma_prev, 
        u_t, 
        z_t, 
        world_map, 
        alphas, 
        dt, 
        sigma_r, 
        sigma_phi):
    """
    Parameters
    ----------
    mu_prev : np.ndarray, shape (3,)
        Previous state estimate mu_{t-1} = [x, y, theta], in metres and radians.

    Sigma_prev : np.ndarray, shape (3, 3)
        Covariance of the previous state estimate, Sigma_{t-1}.

    u_t : array-like, shape (2,)
        Control input [v, w]: translational velocity (m/s) and rotational
        velocity (rad/s) applied during the last time step.

    z_t : list of array-like, each shape (3,)
        Landmark observations at time t, each given as [r, phi, j]:

    world_map : dict
        Known map m.

    alphas : array-like, shape (4,)
        Motion noise parameters [a1, a2, a3, a4] 

    dt : float
        Time step Delta t (s).

    sigma_r : float
        Standard deviation of the range measurement noise (m).

    sigma_phi : float
        Standard deviation of the bearing measurement noise (rad).

    Returns
    -------
    mu : np.ndarray, shape (3,)
        Updated state estimate mu_t, with theta wrapped to [-pi, pi].
    Sigma : np.ndarray, shape (3, 3)
        Updated covariance Sigma_t.
    p_z : float
        Likelihood of the measurements, i.e. the product over all observed
        landmarks of N(z_i; z_hat_i, S_i). Equals 1.0 if z_t is empty.
    """

    theta = mu_prev[2]

    v = u_t[0]
    omega = u_t[1]

    if abs(omega) < 1e-6: 
        omega = 1e-6 

    # Jacobian of the motion model
    G = np.eye(3)
    G[0, 2] = -v/omega * np.cos(theta) + v/omega * np.cos(theta + omega * dt)
    G[1, 2] = -v/omega * np.sin(theta) + v/omega * np.sin(theta + omega * dt)

    # Jacobian of the motion model with respect to the control
    V = np.zeros((3, 2))

    V[0, 0] = (-np.sin(theta) + np.sin(theta + omega * dt))/omega
    V[1, 0] = (np.cos(theta) - np.cos(theta + omega * dt))/omega
    
    V[0, 1] = (v * (np.sin(theta) - np.sin(theta + omega * dt))/(omega**2)
               + (v * np.cos(theta + omega * dt) * dt)/omega)

    V[1, 1] = (-v * (np.cos(theta) - np.cos(theta + omega * dt))/(omega**2)
               + (v * np.sin(theta + omega * dt) * dt)/omega)

    V[2, 1] = dt

    # The control noise covariance
    M = np.zeros((2, 2))
    M[0, 0] = alphas[0] * v**2 + alphas[1] * omega**2
    M[1, 1] = alphas[2] * v**2 + alphas[3] * omega**2

    mu_bar = np.zeros(3)
    mu_bar[0] = (-v * np.sin(theta) + v * np.sin(theta + omega * dt))/omega
    mu_bar[1] = (v * np.cos(theta) - v * np.cos(theta + omega * dt))/omega
    mu_bar[2] = omega * dt

    mu_bar = mu_prev + mu_bar
    mu_bar[2] = np.arctan2(np.sin(mu_bar[2]), np.cos(mu_bar[2]))

    Sigma_bar = G@Sigma_prev@G.T + V@M@V.T

    # The measurement noise covariance
    Q = np.zeros((2, 2))
    Q[0, 0] = sigma_r**2
    Q[1, 1] = sigma_phi**2

    p_z = 1.0

    for z_measure in z_t:
        # r_measure = z_measure[0]
        # phi_measure = z_measure[1]
        s_measure = z_measure[2]

        m_x, m_y = world_map["circles"][s_measure]["center"]

        dx = m_x - mu_bar[0]
        dy = m_y - mu_bar[1]
        q = dx**2 + dy**2

        z_hat = np.zeros(2)
        z_hat[0] = np.sqrt(q)
        z_hat[1] = np.arctan2(dy, dx) - mu_bar[2]
        z_hat[1] = np.arctan2(np.sin(z_hat[1]), np.cos(z_hat[1]))

        H = np.zeros((2, 3))
        H[0, 0] = -dx/np.sqrt(q)
        H[0, 1] = -dy/np.sqrt(q)

        H[1, 0] = dy/q
        H[1, 1] = -dx/q
        H[1, 2] = -1

        S = H@Sigma_bar@H.T + Q
        K = Sigma_bar@H.T@np.linalg.inv(S)

        mu_bar = mu_bar + K@(z_measure[:2] - z_hat)
        Sigma_bar = (np.identity(3) - K@H)@Sigma_bar

        p_z *= (np.linalg.det(2 * np.pi * S)**(-0.5)
                        * np.exp(-0.5 * (z_measure[:2] - z_hat) @ np.linalg.solve(S, (z_measure[:2] - z_hat))))

    return mu_bar, Sigma_bar, p_z

def EKF_localization(        
        mu_prev,
        Sigma_prev, 
        u_t, 
        z_t, 
        world_map, 
        alphas, 
        dt, 
        sigma_r, 
        sigma_phi,
        sigma_s):
    """
    Parameters
    ----------
    mu_prev : np.ndarray, shape (3,)
        Previous state estimate mu_{t-1} = [x, y, theta], in metres and radians.

    Sigma_prev : np.ndarray, shape (3, 3)
        Covariance of the previous state estimate, Sigma_{t-1}.

    u_t : array-like, shape (2,)
        Control input [v, w]: translational velocity (m/s) and rotational
        velocity (rad/s) applied during the last time step.

    z_t : list of array-like, each shape (3,)
        Landmark observations at time t, each given as [r, phi, j]:

    world_map : dict
        Known map m.

    alphas : array-like, shape (4,)
        Motion noise parameters [a1, a2, a3, a4] 

    dt : float
        Time step Delta t (s).

    sigma_r : float
        Standard deviation of the range measurement noise (m).

    sigma_phi : float
        Standard deviation of the bearing measurement noise (rad).

    sigma_s : float
        Standard deviation of the landmark measurement noise.

    Returns
    -------
    mu : np.ndarray, shape (3,)
        Updated state estimate mu_t, with theta wrapped to [-pi, pi].
    Sigma : np.ndarray, shape (3, 3)
        Updated covariance Sigma_t.
    """

    theta = mu_prev[2]
    
    v = u_t[0]
    omega = u_t[1]

    if abs(omega) < 1e-6: 
        omega = 1e-6 

    # Jacobian of the motion model
    G = np.eye(3)
    G[0, 2] = -v/omega * np.cos(theta) + v/omega * np.cos(theta + omega * dt)
    G[1, 2] = -v/omega * np.sin(theta) + v/omega * np.sin(theta + omega * dt)

    # Jacobian of the motion model with respect to the control
    V = np.zeros((3, 2))

    V[0, 0] = (-np.sin(theta) + np.sin(theta + omega * dt))/omega
    V[1, 0] = (np.cos(theta) - np.cos(theta + omega * dt))/omega
    
    V[0, 1] = (v * (np.sin(theta) - np.sin(theta + omega * dt))/(omega**2)
                + (v * np.cos(theta + omega * dt) * dt)/omega)

    V[1, 1] = (-v * (np.cos(theta) - np.cos(theta + omega * dt))/(omega**2)
                + (v * np.sin(theta + omega * dt) * dt)/omega)

    V[2, 1] = dt

    # The control noise covariance
    M = np.zeros((2, 2))
    M[0, 0] = alphas[0] * v**2 + alphas[1] * omega**2
    M[1, 1] = alphas[2] * v**2 + alphas[3] * omega**2

    mu_bar = np.zeros(3)
    mu_bar[0] = (-v * np.sin(theta) + v * np.sin(theta + omega * dt))/omega
    mu_bar[1] = (v * np.cos(theta) - v * np.cos(theta + omega * dt))/omega
    mu_bar[2] = omega * dt

    mu_bar = mu_prev + mu_bar
    mu_bar[2] = np.arctan2(np.sin(mu_bar[2]), np.cos(mu_bar[2]))

    Sigma_bar = G@Sigma_prev@G.T + V@M@V.T

    # The measurement noise covariance
    Q = np.zeros((3, 3))
    Q[0, 0] = sigma_r**2
    Q[1, 1] = sigma_phi**2
    Q[2, 2] = sigma_s**2

    for z_measure in z_t:
        z_khat =[]
        H_k =[]
        S_k =[]
        z_innovation = []
        for landmark_idx, landmark in enumerate(world_map["circles"]):
            m_x, m_y = landmark["center"]

            dx = m_x - mu_bar[0]
            dy = m_y - mu_bar[1]
            q = dx**2 + dy**2

            z_hat = np.zeros(3)
            z_hat[0] = np.sqrt(q)
            z_hat[1] = np.arctan2(dy, dx) - mu_bar[2]
            z_hat[1] = np.arctan2(np.sin(z_hat[1]), np.cos(z_hat[1]))
            z_hat[2] = landmark_idx

            H = np.zeros((3, 3))
            H[0, 0] = -dx/np.sqrt(q)
            H[0, 1] = -dy/np.sqrt(q)

            H[1, 0] = dy/q
            H[1, 1] = -dx/q
            H[1, 2] = -1

            z_khat.append(z_hat)
            H_k.append(H)
            S_k.append(H@Sigma_bar@H.T + Q)

            innovation = z_measure - z_hat
            innovation[1] = np.arctan2(np.sin(innovation[1]), np.cos(innovation[1]))
            z_innovation.append(innovation)

        
        z_khat = np.array(z_khat)
        H_k = np.array(H_k)
        S_k = np.array(S_k)
        z_innovation = np.array(z_innovation)

        S_inv = np.linalg.inv(S_k)

        j = np.argmax([np.linalg.det(2*np.pi*S_k[k])**(-0.5) * np.exp(-0.5 * z_innovation[k].T @ S_inv[k] @ z_innovation[k]) for k in range(len(S_k))])
        K = Sigma_bar@H_k[j].T@S_inv[j]

        mu_bar = mu_bar + K@(z_innovation[j])
        mu_bar[2] = np.arctan2(np.sin(mu_bar[2]), np.cos(mu_bar[2]))

        Sigma_bar = (np.identity(3) - K@H_k[j])@Sigma_bar
    return mu_bar, Sigma_bar

def UKF_localization(        
        mu_prev,
        Sigma_prev, 
        u_t, 
        z_t, 
        world_map, 
        alphas, 
        dt, 
        sigma_r, 
        sigma_phi,
        gamma_args):
    """
    Parameters
    ----------
    mu_prev : np.ndarray, shape (3,)
        Previous state estimate mu_{t-1} = [x, y, theta], in metres and radians.

    Sigma_prev : np.ndarray, shape (3, 3)
        Covariance of the previous state estimate, Sigma_{t-1}.

    u_t : array-like, shape (2,)
        Control input [v, w]: translational velocity (m/s) and rotational
        velocity (rad/s) applied during the last time step.

    z_t : list of array-like, each shape (3,)
        Landmark observations at time t, each given as [r, phi, j]:

    world_map : dict
        Known map m.

    alphas : array-like, shape (4,)
        Motion noise parameters [a1, a2, a3, a4] 

    dt : float
        Time step Delta t (s).

    sigma_r : float
        Standard deviation of the range measurement noise (m).

    sigma_phi : float
        Standard deviation of the bearing measurement noise (rad).

    gamma_args : list, shape (3)
        alpha_gamma, kappa_gamma, beta_gamma 

    Returns
    -------
    mu : np.ndarray, shape (3,)
        Updated state estimate mu_t, with theta wrapped to [-pi, pi].
    Sigma : np.ndarray, shape (3, 3)
        Updated covariance Sigma_t.
    p_z : float
        Likelihood of the measurements, i.e. the product over all observed
        landmarks of N(z_i; z_hat_i, S_i). Equals 1.0 if z_t is empty.
    """

    theta = mu_prev[2]
    
    v = u_t[0]
    omega = u_t[1]

    alpha_gamma = gamma_args[0] 
    kappa_gamma = gamma_args[1] 
    beta_gamma = gamma_args[2] 
    L_gamma = 7 

    gamma = np.sqrt(alpha_gamma**2*(L_gamma+kappa_gamma))


    if abs(omega) < 1e-6: 
        omega = 1e-6 

    # The control noise covariance
    M = np.zeros((2, 2))
    M[0, 0] = alphas[0] * v**2 + alphas[1] * omega**2
    M[1, 1] = alphas[2] * v**2 + alphas[3] * omega**2
    M = M + 1e-9*np.eye(2)

    # The measurement noise covariance
    Q = np.zeros((2, 2))
    Q[0, 0] = sigma_r**2
    Q[1, 1] = sigma_phi**2

    # Construct mu_a
    mu_a_prev = np.hstack((mu_prev, np.zeros((4))))
    mu_a_prev = mu_a_prev[:, None]

    # Construct Sigma_a
    S1 = np.hstack((Sigma_prev, np.zeros((3, 4))))
    S2 = np.hstack((np.zeros((2, 3)), M, np.zeros((2, 2))))
    S3 = np.hstack((np.zeros((2, 5)), Q))
    Sigma_a = np.vstack((S1, S2, S3))

    # Construct chi_a
    Sigma_a = (Sigma_a + Sigma_a.T)/2 + 1e-9*np.eye(L_gamma)
    Sigma_a_root = np.linalg.cholesky(Sigma_a)
    chi_a = np.hstack((mu_a_prev, mu_a_prev + gamma*Sigma_a_root, mu_a_prev - gamma*Sigma_a_root))

    # Construct chi_bar
    u_noise = np.zeros(2)
    chi_bar = []
    for idx in range(chi_a.shape[1]):
        g_prev = chi_a[0:3, idx]
        u_noise[0] = u_t[0] + chi_a[3, idx]
        u_noise[1] = u_t[1] + chi_a[4, idx]
        chi_bar.append(_g(g_prev, u_noise, dt))

    chi_bar = np.array(chi_bar).T

    # mu bar
    lambda_weight = alpha_gamma**2*(L_gamma + kappa_gamma) - L_gamma
    mu_bar = lambda_weight/(L_gamma+lambda_weight)*chi_bar[:, 0]
    for idx in range(1, chi_bar.shape[1]):
        mu_bar = mu_bar + (1/(2*(L_gamma+lambda_weight))*chi_bar[:, idx])


    # Sigma bar
    w_c0 = lambda_weight/(L_gamma+lambda_weight) + (1 - alpha_gamma**2 + beta_gamma)
    w_ci = 1/(2*(L_gamma+lambda_weight))

    d = chi_bar[:, 0] - mu_bar
    Sigma_bar = w_c0*np.outer(d, d)
    for idx in range(1, chi_bar.shape[1]):
        d = chi_bar[:, idx] - mu_bar
        d[2] = np.arctan2(np.sin(d[2]), np.cos(d[2]))
        Sigma_bar = Sigma_bar + w_ci*np.outer(d, d)

    n_sigma = 2*L_gamma + 1

    w_m = np.full(n_sigma, 1/(2*(L_gamma+lambda_weight)))
    w_c = np.full(n_sigma, 1/(2*(L_gamma+lambda_weight)))
    w_m[0] = lambda_weight/(L_gamma+lambda_weight)
    w_c[0] = w_m[0] + (1 - alpha_gamma**2 + beta_gamma)

    if len(z_t) == 0:
        return mu_bar, Sigma_bar, 1.0

    k = len(z_t)

    Z_bar = np.zeros((2*k, n_sigma))
    z_meas = np.zeros(2*k)
    for obs_idx, z_measure in enumerate(z_t):
        j = int(z_measure[2])
        m_x, m_y = world_map["circles"][j]["center"]
        z_meas[2*obs_idx:2*obs_idx+2] = z_measure[:2]

        for i in range(n_sigma):
            dx = m_x - chi_bar[0, i]
            dy = m_y - chi_bar[1, i]
            phi = np.arctan2(dy, dx) - chi_bar[2, i]
            Z_bar[2*obs_idx, i] = np.sqrt(dx**2 + dy**2)
            Z_bar[2*obs_idx+1, i] = np.arctan2(np.sin(phi), np.cos(phi))

    z_hat = Z_bar @ w_m
    z_hat[1::2] = np.arctan2(np.sin(Z_bar[1::2]) @ w_m, np.cos(Z_bar[1::2]) @ w_m)

    S = np.kron(np.eye(k), Q)
    Sigma_xz = np.zeros((3, 2*k))
    for i in range(n_sigma):
        dz = Z_bar[:, i] - z_hat
        dz[1::2] = np.arctan2(np.sin(dz[1::2]), np.cos(dz[1::2]))
        dz = dz.reshape(-1, 1)

        dx = chi_bar[:, i] - mu_bar
        dx[2] = np.arctan2(np.sin(dx[2]), np.cos(dx[2]))
        dx = dx.reshape(-1, 1)

        S = S + w_c[i] * (dz @ dz.T)
        Sigma_xz = Sigma_xz + w_c[i] * (dx @ dz.T)

    K = Sigma_xz @ np.linalg.inv(S)

    innovation = z_meas - z_hat
    innovation[1::2] = np.arctan2(np.sin(innovation[1::2]), np.cos(innovation[1::2]))
    mu = mu_bar + K @ innovation
    mu[2] = np.arctan2(np.sin(mu[2]), np.cos(mu[2]))

    Sigma = Sigma_bar - K @ S @ K.T
    Sigma = (Sigma + Sigma.T) / 2

    eigvals, eigvecs = np.linalg.eigh(Sigma)
    eigvals = np.maximum(eigvals, 1e-9)
    Sigma = eigvecs @ np.diag(eigvals) @ eigvecs.T

    p_z = (np.linalg.det(2*np.pi*S)**(-0.5)
           * np.exp(-0.5 * innovation @ np.linalg.solve(S, innovation)))

    return mu, Sigma, p_z




    