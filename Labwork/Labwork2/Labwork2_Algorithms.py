import numpy as np
import scipy.stats as ss
import Labwork1.Labwork1 as L1

rng = np.random.default_rng()

# Change map variable name to something else

# Helper functions
def normalizer(z_true, lambda_short):
    return 1/(1-(np.e**(-lambda_short*z_true)))

def p_hit(z_measurement, z_true, sigma_hit, max_range):
    if max_range-z_measurement < 0.0001:
        return 0

    a = -z_true/sigma_hit
    b = (max_range-z_true)/sigma_hit
    return ss.truncnorm.pdf(z_measurement, a=a, b=b, loc=z_true, scale=sigma_hit)

def p_short(z_measurement, z_true, lambda_short):
    if 0 <= z_measurement <= z_true:
        return normalizer(z_true, lambda_short)*lambda_short*(np.e**(-lambda_short*z_measurement))
    else:
        return 0

def p_max(z_measurement, max_range):
    if z_measurement == max_range:
        return 1
    else:
        return 0

def p_rand(z_measurement, max_range):
    if 0 <= z_measurement < max_range:
        return 1/max_range
    else:
        return 0

# Algorithms
def beam_range_finder_model(
    z_true: np.ndarray, 
    z_noise: np.ndarray,
    max_range: float,
    params: list,
          ):
    """
    z_true: All beams from a single measurement without unexpected obstacles. Essentially the expected measurements

    z_noise: All beams from a single measurement with unexpected obstacles. Essentially what is actually measured

    max_range: The maximum range of ray casting beams

    All z params should sum to 1
    params:
        z_hit

        z_short

        z_max

        z_rand

        sigma_hit 

        lambda_short

        hit: Correct range with noise. Modelled as gaussian with standardeviation of sigma_hit
        short: Unexpected objects. The rate is described by lambda_short
        max: failure, no return. The beam didn't get a return, glass, black surfaces, long distances 
        rand: random measurements. Unexplainable readings such as sonar crosstalk or phantom echoes
    """

    q = 1

    z_hit = params[0]
    z_short = params[1]
    z_max = params[2]
    z_rand = params[3]
    sigma_hit = params[4]
    lambda_short = params[5]

    for ray_true, ray_noise in zip(z_true, z_noise):
        p = (z_hit*p_hit(ray_noise, ray_true, sigma_hit, max_range) 
        + z_short*p_short(ray_noise, ray_true, lambda_short)
        + z_max*p_max(ray_noise, max_range)
        + z_rand*p_rand(ray_noise, max_range))
        q = q * p
    return q

def learn_intrinsic_parameters(beams_true, beams_noise, max_range, params):
    """
    beams_true: All beams from a single measurement without unexpected obstacles. Essentially the expected measurements

    beams_noise: All beams from a single measurement with unexpected obstacles. Essentially what is actually measured

    max_range: Maximum ray cast range

    All z params should sum to 1
    params:
        z_hit

        z_short

        z_max

        z_rand

        sigma_hit 

        lambda_short

        hit: Correct range with noise. Modelled as gaussian with standardeviation of sigma_hit
        short: Unexpected objects. The rate is described by lambda_short
        max: failure, no return. The beam didn't get a return, glass, black surfaces, long distances 
        rand: random measurements. Unexplainable readings such as sonar crosstalk or phantom echoes
    """

    z_hit = params[0]
    z_short = params[1]
    z_max = params[2]
    z_rand = params[3]
    sigma_hit = params[4]
    lambda_short = params[5]

    learned_params = np.zeros(6)

    beams_true = beams_true.flatten()
    beams_noise = beams_noise.flatten()

    e_hit = np.zeros(len(beams_true))
    e_short = np.zeros(len(beams_true))
    e_max = np.zeros(len(beams_true))
    e_rand = np.zeros(len(beams_true))

    for idx, beam in enumerate(zip(beams_true, beams_noise)):
        z_true, z_noise = beam
        eta = ((p_hit(z_noise, z_true, sigma_hit, max_range)
                + p_short(z_noise, z_true, lambda_short)
                + p_max(z_noise, max_range)
                + p_rand(z_noise, max_range))**-1)

        e_hit[idx] = eta*p_hit(z_noise, z_true, sigma_hit, max_range)
        e_short[idx] = eta*p_short(z_noise, z_true, lambda_short)
        e_max[idx] = eta*p_max(z_noise, max_range)
        e_rand[idx] = eta*p_rand(z_noise, max_range)

    learned_params[0] = ((beams_noise.size)**-1)*np.sum(e_hit)
    learned_params[1] = ((beams_noise.size)**-1)*np.sum(e_short)
    learned_params[2] = ((beams_noise.size)**-1)*np.sum(e_max)
    learned_params[3] = ((beams_noise.size)**-1)*np.sum(e_rand)

    learned_params[4] = np.sqrt((1/np.sum(e_hit))*np.sum(e_hit*(beams_noise.flatten()-beams_true.flatten())**2))
    learned_params[5] = np.sum(e_short)/np.sum(e_short*beams_noise.flatten())

    if np.all(np.abs(params-learned_params) < 0.001):
        learned = True
    else:
        learned = False

    return learned_params, learned

def likelihood_field_range_finder_model(beams_noise, pose, world_map, sensor_mounting, max_range, params, use_triangle = False):
    """
    beams_noise: All beams from a single measurement with unexpected obstacles. Essentially what is actually measured

    pose: Robot pose

    world_map: All static objects in the world map

    sensor_mounting: offset of each ray

    max_range: Maximum ray cast range

    All z params should sum to 1
    params:
        z_hit

        z_short

        z_max

        z_rand

        sigma_hit 

        lambda_short

        hit: Correct range with noise. Modelled as gaussian with standardeviation of sigma_hit
        short: Unexpected objects. The rate is described by lambda_short
        max: failure, no return. The beam didn't get a return, glass, black surfaces, long distances 
        rand: random measurements. Unexplainable readings such as sonar crosstalk or phantom echoes
    """

    q = 1

    z_hit = params[0]
    z_short = params[1]
    z_max = params[2]
    z_rand = params[3]
    sigma_hit = params[4]
    lambda_short = params[5]

    x = pose[0]
    y = pose[1]
    theta = pose[2]

    # x_sens = y_sens = 0 because alle beams are placed int the robots center
    for beam, theta_sens in zip(beams_noise, sensor_mounting):
        if beam != max_range:
            x_beam = x + beam*np.cos(theta+theta_sens)
            y_beam = y + beam*np.sin(theta+theta_sens)

            endpoint = np.array([x_beam, y_beam])
            min_distance = np.inf

            for circle in world_map['circles']:
                distance = np.abs(np.linalg.norm(endpoint - circle["center"]) - circle["radius"])
                if  min_distance > distance:
                    min_distance = distance
            
            for line in world_map['lines']:
                p1 = line["p1"]
                p2 = line["p2"]

                line_vec = p2 - p1
                end_to_p1_vec = endpoint - p1
                
                line_length_sq = np.dot(line_vec, line_vec)
                if line_length_sq == 0: 
                    distance = np.linalg.norm(end_to_p1_vec)

                else:
                    t = np.dot(end_to_p1_vec, line_vec) / line_length_sq
                    t = np.clip(t, 0.0, 1.0)
                    
                    closest_point = p1 + t * line_vec
                    
                    distance = np.linalg.norm(closest_point - endpoint)

                if  min_distance > distance:
                    min_distance = distance

            if use_triangle == True:
                q = q * (z_hit * L1.prob_tri_distro(min_distance, sigma_hit) + z_rand/max_range)
            else: 
                q = q * (z_hit * L1.prob_normal_distro(min_distance, sigma_hit) + z_rand/max_range)
    return q

def landmark_model_known_correspondance(measurements, pose, world_map, dist_noise, bearing_noise):
    """
    Measurements: Landmark measurements [r, phi, s], r is distance, phi bearing, and s the landmark id
    
    pose: pose being tested

    world_map: a map that contains landmarks

    range_noise: noise of distance

    bearing_noise: noise of bearing
    """

    x = pose[0]
    y = pose[1]
    theta = pose[2]

    r_measured = measurements[0]
    bearing_measured = measurements[1]
    landmark_idx = measurements[2]

    landmark = world_map["circles"][landmark_idx]
    x_landmark, y_landmark = landmark["center"]

    vec_robot_landmark = pose[:2]-landmark["center"]
    dist_robot_landmark = np.linalg.norm(vec_robot_landmark) 

    bearing = np.arctan2(y_landmark - y, x_landmark - x) - theta
    bearing = np.arctan2(np.sin(bearing), np.cos(bearing))

    wrapped_bearing = np.arctan2(np.sin(bearing - bearing_measured), np.cos(bearing - bearing_measured))

    q = (L1.prob_normal_distro(dist_robot_landmark - r_measured, dist_noise)
         * L1.prob_normal_distro(wrapped_bearing, bearing_noise))
    
    return q
    
def sample_landmark_model_known_correspondance(measurement, world_map, dist_noise, bearing_noise):
    """
    measurement: Landmark measurement [r, phi, s], r is distance, phi bearing, and s the landmark id
    
    pose: pose being tested

    world_map: a map that contains landmarks

    range_noise: noise of distance

    bearing_noise: noise of bearing
    """
    r_measured = measurement[0]
    bearing_measured = measurement[1]
    landmark_idx = measurement[2]

    landmark = world_map["circles"][landmark_idx]
    x_landmark, y_landmark = landmark["center"]

    gamma_hat = rng.uniform(0, 2*np.pi)

    distance_hat = r_measured + L1.sample_normal_distro(dist_noise)
    bearing_hat = bearing_measured + L1.sample_normal_distro(bearing_noise)

    x = x_landmark + distance_hat*np.cos(gamma_hat)
    y = y_landmark + distance_hat*np.sin(gamma_hat)
    theta = gamma_hat - np.pi - bearing_hat
    theta = np.arctan2(np.sin(theta), np.cos(theta))
    
    return np.array([x, y, theta])