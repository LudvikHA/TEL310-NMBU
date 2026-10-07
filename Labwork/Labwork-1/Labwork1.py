import numpy as np
import pandas as pd

rng = np.random.default_rng()


def prob_normal_distro(a: float, b: float) -> float:
    """
    Calculate probability based on a gaussian distribution
    """
    return 1/(np.sqrt(2*np.pi*b**2))*np.exp(-((a**2)/(2*b**2)))

def prob_tri_distro(a: float, b: float) -> float:
    """
    Calculate probability based on a triangle distribution
    """
    if abs(a) > np.sqrt(6)*b:
        return 0.0
    else: 
        return 1/(np.sqrt(6)*b)-abs(a)/(6*b**2)
    
def sample_normal_distro(b: float, sample_size: int=12) -> float:
    """
    Draw random samples between -b and b based on a gaussian distribution
    """
    return 0.5*np.sum(rng.uniform(low=-b, high=b, size=sample_size))

def sample_tri_distro(b: float) -> float:
    """
    Draw two random samples between -b and b based on a triangle distribution
    """
    return (np.sqrt(6)/2)*(rng.uniform(low=-b, high=b)+rng.uniform(low=-b, high=b))

def motion_model_velocity(x_prev: np.ndarray, x_t: np.ndarray, u_t: np.ndarray, robot_params: tuple, triangle: bool=False) -> float:
    """
    x_prev = [x y theta]

    x_t = [x' y' theta']

    u_t = [v omega]

    robot_params = (delta_time, noise_alpha_1, noise_alpha_2, ... noise_alpha_6)
    """

    # Coefficient
    # x_diff = x_prev-x_t

    x = x_prev[0]
    y = x_prev[1]
    theta = x_prev[2]

    x_prime = x_t[0]
    y_prime = x_t[1]
    theta_prime = x_t[2]

    delta_t = robot_params[0]
    alpha1 = robot_params[1]
    alpha2 = robot_params[2]
    alpha3 = robot_params[3]
    alpha4 = robot_params[4]
    alpha5 = robot_params[5]
    alpha6 = robot_params[6]

    numerator = (x-x_prime)*np.cos(theta)+(y-y_prime)*np.sin(theta)
    denominator = (y-y_prime)*np.cos(theta)-(x-x_prime)*np.sin(theta)

    if abs(denominator) <= 1e-6*abs(numerator) + 1e-12:
        omega_hat = 0.0
        v_hat = ((x_prime-x)*np.cos(theta)+(y_prime-y)*np.sin(theta))/delta_t
    else:
        mu = 0.5*(numerator/denominator)

        # Center of rotation
        x_star = (x+x_prime)/2+mu*(y-y_prime)
        y_star = (y+y_prime)/2+mu*(x_prime-x)

        # Radius
        r_star = np.sqrt((x-x_star)**2+(y-y_star)**2)
        delta_theta = np.arctan2(y_prime-y_star, x_prime-x_star)-np.arctan2(y-y_star, x-x_star)

        # Velocities obtained from the hypotheses
        omega_hat = delta_theta/delta_t
        v_hat = omega_hat*r_star

    gamma_hat = (theta_prime-theta)/delta_t-omega_hat


    if triangle == True:
        p1 = prob_tri_distro(u_t[0]-v_hat, alpha1*u_t[0]**2+alpha2*u_t[1]**2)
        p2 = prob_tri_distro(u_t[1]-omega_hat, alpha3*u_t[0]**2+alpha4*u_t[1]**2)
        p3 = prob_tri_distro(gamma_hat, alpha5*u_t[0]**2+alpha6*u_t[1]**2)

    else:
        p1 = prob_normal_distro(u_t[0]-v_hat, alpha1*u_t[0]**2+alpha2*u_t[1]**2)
        p2 = prob_normal_distro(u_t[1]-omega_hat, alpha3*u_t[0]**2+alpha4*u_t[1]**2)
        p3 = prob_normal_distro(gamma_hat, alpha5*u_t[0]**2+alpha6*u_t[1]**2)

    return p1*p2*p3

def sample_motion_model_velocity(
        x_prev: np.ndarray,
        u_t: np.ndarray, 
        robot_params: tuple, 
        sample_interval: int,
        triangle: bool=False):
    """
    x_prev = [x y theta]

    u_t = [v omega]

    robot_params = (delta_time, noise_alpha_1, noise_alpha_2, ... noise_alpha_6)
    """
    v = u_t[0] 
    omega = u_t[1] 
    x = x_prev[0]
    y = x_prev[1]
    theta = x_prev[2]

    delta_t = robot_params[0]*sample_interval
    alpha1 = robot_params[1]
    alpha2 = robot_params[2]
    alpha3 = robot_params[3]
    alpha4 = robot_params[4]
    alpha5 = robot_params[5]
    alpha6 = robot_params[6]

    if triangle == False:
        v_hat = v+sample_normal_distro(alpha1*v**2+alpha2*omega**2)
        omega_hat = omega+sample_normal_distro(alpha3*v**2+alpha4*omega**2)
        gamma_hat = sample_normal_distro(alpha5*v**2+alpha6*omega**2)

    else:
        v_hat = v+sample_tri_distro(alpha1*v**2+alpha2*omega**2)
        omega_hat = omega+sample_tri_distro(alpha3*v**2+alpha4*omega**2)
        gamma_hat = sample_tri_distro(alpha5*v**2+alpha6*omega**2)


    # Radius
    theta_hat = theta+omega_hat*delta_t

    # Sample coordinates
    if abs(omega_hat) > 1e-6:
        r_hat = v_hat/omega_hat
        x_prime = x - r_hat*(np.sin(theta)-np.sin(theta_hat))
        y_prime = y + r_hat*(np.cos(theta)-np.cos(theta_hat))
    else:
        x_prime = x + v_hat*delta_t*np.cos(theta)
        y_prime = y + v_hat*delta_t*np.sin(theta)

    theta_prime = theta_hat+gamma_hat*delta_t

    return np.array((x_prime, y_prime, theta_prime))

def sample_motion_model_odometry(
        x_prev: np.ndarray,
        u_t: np.ndarray,
        robot_params: tuple,
        triangle: bool=False):
    """
    x_prev = [x y theta]

    u_t = [x_bar_prev, x_bar_t]

    x_bar_prev = [x_bar, y_bar, theta_bar]
    x_bar_t = [x_bar_prime, y_bar_prime, theta_bar_prime]

    robot_params = (delta_time, noise_alpha_1, noise_alpha_2, ... noise_alpha_6)
    """

    # Variables from u
    x_bar_prev = u_t[0] 
    x_bar_t = u_t[1] 

    x_bar = x_bar_prev[0]
    y_bar = x_bar_prev[1]
    theta_bar = x_bar_prev[2]

    x_bar_prime = x_bar_t[0]
    y_bar_prime = x_bar_t[1]
    theta_bar_prime = x_bar_t[2]

    # Variables from x_prev
    x = x_prev[0]
    y = x_prev[1]
    theta = x_prev[2]

    delta_t = robot_params[0]
    alpha1 = robot_params[1]
    alpha2 = robot_params[2]
    alpha3 = robot_params[3]
    alpha4 = robot_params[4]
    alpha5 = robot_params[5]
    alpha6 = robot_params[6]

    rot1 = np.arctan2(y_bar_prime-y_bar, x_bar_prime-x_bar)-theta_bar
    trans = np.sqrt((x_bar_prime-x_bar)**2+(y_bar_prime-y_bar)**2)
    rot2 = theta_bar_prime - theta_bar - rot1

    if triangle == False:
        rot1_hat = rot1 - sample_normal_distro(alpha1*rot1**2+alpha2*trans**2)
        trans_hat = trans - sample_normal_distro(alpha3*trans**2+alpha4*(rot1**2+rot2**2))
        rot2_hat = rot2 - sample_normal_distro(alpha5*rot2**2+alpha6*trans**2)

    else:
        rot1_hat = rot1 - sample_tri_distro(alpha1*rot1**2+alpha2*trans**2)
        trans_hat = trans - sample_tri_distro(alpha3*trans**2+alpha4*(rot1**2+rot2**2))
        rot2_hat = rot2 - sample_tri_distro(alpha5*rot2**2+alpha6*trans**2)

    x_prime = x + trans_hat*np.cos(theta+rot1_hat)
    y_prime = y + trans_hat*np.sin(theta+rot1_hat)
    theta_prime = theta + rot1_hat + rot2_hat

    return np.array([x_prime, y_prime, theta_prime])

def motion_model_odometry(
    x_prev: np.ndarray,
    x_t: np.ndarray,    
    u_t: np.ndarray,
    robot_params: tuple,
    triangle: bool=False):
    """
    x_prev = [x y theta]

    x_t = [x' y' theta']

    u_t = [x_bar_prev, x_bar_t]

    x_bar_prev = [x_bar, y_bar, theta_bar]
    x_bar_t = [x_bar_prime, y_bar_prime, theta_bar_prime]

    robot_params = (delta_time, noise_alpha_1, noise_alpha_2, ... noise_alpha_6)
    """

    # Variables from u
    x_bar_prev = u_t[0] 
    x_bar_t = u_t[1] 

    x_bar = x_bar_prev[0]
    y_bar = x_bar_prev[1]
    theta_bar = x_bar_prev[2]

    x_bar_prime = x_bar_t[0]
    y_bar_prime = x_bar_t[1]
    theta_bar_prime = x_bar_t[2]

    # Variables from x_prev
    x = x_prev[0]
    y = x_prev[1]
    theta = x_prev[2]

    # Variables from x_t
    x_prime = x_t[0]
    y_prime = x_t[1]
    theta_prime = x_t[2]

    delta_t = robot_params[0]
    alpha1 = robot_params[1]
    alpha2 = robot_params[2]
    alpha3 = robot_params[3]
    alpha4 = robot_params[4]
    alpha5 = robot_params[5]
    alpha6 = robot_params[6]

    # Odometry components
    rot1 = np.arctan2(y_bar_prime-y_bar, x_bar_prime-x_bar)-theta_bar
    trans = np.sqrt((x_bar_prime-x_bar)**2+(y_bar_prime-y_bar)**2)
    rot2 = theta_bar_prime - theta_bar - rot1

    rot1_hat = np.arctan2(y_prime - y, x_prime - x) - theta
    trans_hat = np.sqrt((x_prime - x)**2+(y_prime - y)**2)
    rot2_hat = theta_prime - theta - rot1_hat

    if triangle == False:
        p1 = prob_normal_distro(rot1 - rot1_hat, alpha1*abs(rot1) + alpha2*trans)
        p2 = prob_normal_distro(trans - trans_hat, alpha3*abs(trans) + alpha4*(abs(rot1) + abs(rot2)))
        p3 = prob_normal_distro(rot2 - rot2_hat, alpha5*abs(rot2) + alpha6*trans)
    else:
        p1 = prob_tri_distro(rot1 - rot1_hat, alpha1*abs(rot1) + alpha2*trans)
        p2 = prob_tri_distro(trans - trans_hat, alpha3*abs(trans) + alpha4*(abs(rot1) + abs(rot2)))
        p3 = prob_tri_distro(rot2 - rot2_hat, alpha5*abs(rot2) + alpha6*trans)  

    return p1*p2*p3


