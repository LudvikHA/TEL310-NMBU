#=============================================================================================
# SET-Prob: A Modular Simulation Environment for Testing Algorithms in Probabilistic Robotics
# Author: Igor Ferreira da Costa
# Course name: TEL-310
# Course responsible: Antonio Candea Leite / Shailendra Singh
#
# Description: 
# A simple, modular simulation environment featuring:
# A unicycle robot (x–y pose, differential drive kinematics).
# A 2D LiDAR with ray-casting for range sensing.
# Obstacle design and detection (segments, circles, polygons).
#
# Designed for labs on probabilistic motion models, perception models, localization, and SLAM.
#
# Dependency: numpy, cv2
#
#=============================================================================================

import numpy as np
import cv2
import Labwork1 as L1
import matplotlib.pyplot as plt
import time

class Robot:
    def __init__(self, initial_pose=np.zeros(3), a_max=0.5, alpha_max=0.5, dt=0.1, radius=0.1):
        self.pose = initial_pose
        self.v = 0.0
        self.w = 0.0
        self.des_v = 0.0
        self.des_w = 0.0
        self.a_max = a_max
        self.alpha_max = alpha_max
        self.dt = dt
        self.radius = radius

    def get_pose(self):
        return self.pose
    
    def get_vel(self):
        return [float(self.v), float(self.w)]

    def update_speed(self):
        dv = np.clip(self.des_v - self.v, -self.a_max * self.dt, self.a_max * self.dt)
        dw = np.clip(self.des_w - self.w, -self.alpha_max * self.dt, self.alpha_max * self.dt)
        
        self.v += dv
        self.w += dw
        
    def update_pose(self):
        x, y, theta = self.pose
        
        x_new = x + self.v * np.cos(theta) * self.dt
        y_new = y + self.v * np.sin(theta) * self.dt
        theta_new = theta + self.w * self.dt
        
        return np.array([x_new, y_new, theta_new])

class Camera:
    def __init__(self, width=640, height=480, meters_per_pixel=0.01):
        self.width = width
        self.height = height
        self.meters_per_pixel = meters_per_pixel
        
        self.image_center_x = self.width / 2
        self.image_center_y = self.height / 2

    def project_point(self, point_world):
        x_world, y_world = point_world[0], point_world[1]
        
        u = self.image_center_x + x_world / self.meters_per_pixel
        v = self.image_center_y - y_world / self.meters_per_pixel

        return (int(u), int(v))

class Renderer:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.lines = []
        self.points = []

    def add_line(self, prev_pos, true_pos, true_vel, robot_params, odo_or_vel, odo_vel=None):
        x, y, _ = np.int16(np.round(prev_pos*100, 0))
        x_true, y_true, _ = np.int16(np.round(true_pos*100, 0))
        self.lines.append(((x+400, -y+400), (x_true+400, -y_true+400)))

        sample_poses = []
        if odo_or_vel == "vel":
            for i in range(50):
                x_prime, y_prime, _ = L1.sample_motion_model_velocity(prev_pos, true_vel, robot_params, sample_interval, use_triangle)

                x_pred = np.int16(np.round(x_prime*100, 0))
                y_pred = np.int16(np.round(y_prime*100, 0))
                sample_poses.append((x_pred+400, -(y_pred)+400))

        elif odo_or_vel == "odo":
            for i in range(50):
                x_prime, y_prime, _ = L1.sample_motion_model_odometry(prev_pos, odo_vel, robot_params, use_triangle)

                x_pred = np.int16(np.round(x_prime*100, 0))
                y_pred = np.int16(np.round(y_prime*100, 0))
                sample_poses.append((x_pred+400, -(y_pred)+400))
        self.points.append(sample_poses)

    def render_sample_model(self, img):
        for line in self.lines:
            p1, p2 = line
            cv2.line(img, p1, p2, (200, 0, 20), 3)

        for cloud in self.points:
            for point in cloud:
                cv2.circle(img, center=point, radius=5, color=(0, 0, 255), thickness=-1)
        return img

    def render_image(self, objects):
        image = np.full((self.height, self.width, 3), (255, 255, 255), dtype=np.uint8)
        
        for obj_type, data in objects:
            if obj_type == 'line':
                p1, p2, color, thickness = data
                cv2.line(image, p1, p2, tuple(reversed(color)), thickness)
            elif obj_type == 'circle':
                center, radius, color, thickness = data
                cv2.circle(image, center, radius, tuple(reversed(color)), thickness)
            elif obj_type == 'robot':
                center, radius, color, theta = data
                cv2.circle(image, center, radius, tuple(reversed(color)), -1)
                
                end_point_x = center[0] + int(radius * np.cos(theta))
                end_point_y = center[1] - int(radius * np.sin(theta))
                cv2.line(image, center, (end_point_x, end_point_y), (0, 0, 0), 2)
            
        return image

class Simulator:
    def __init__(self, robot_init_pose, cam_params, time_step, world_map):
        self.camera = Camera(**cam_params)
        self.robot = Robot(**robot_init_pose)
        self.renderer = Renderer(self.camera.width, self.camera.height)
        self.dt = time_step

        self.lidar_max_range = 2.0
        self.lidar_angle_step = np.radians(1.0)
        self.lidar_angles = np.arange(-np.pi/2, np.pi/2, self.lidar_angle_step)
        
        self.world_map = world_map

    def _create_world_map(self):
        world_map = {'lines': [], 'circles': []}
        
        walls = [
            (( -4.0, -4.0), ( 4.0, -4.0), (0, 0, 0)),
            ((  4.0, -4.0), ( 4.0,  4.0), (0, 0, 0)),
            ((  4.0,  4.0), (-4.0,  4.0), (0, 0, 0)),
            (( -4.0,  4.0), (-4.0, -4.0), (0, 0, 0))
        ]
        
        circles = [
            (( 1.5,  1.5,  0.1), (100, 100, 200)),
            ((-1.5,  1.5,  0.1), (100, 200, 200)),
            ((-1.5, -1.5,  0.1), (200, 100, 200)),
            (( 1.5, -1.5,  0.1), (100, 100, 100))
        ]
        
        for p1_world, p2_world, color in walls:
            world_map['lines'].append({'p1': np.array(p1_world), 'p2': np.array(p2_world), 'color': np.array(color)})
            
        for circle, color in circles:
            x, y, r = circle
            world_map['circles'].append({'center': np.array([x, y]), 'radius': r, 'color': np.array(color)})

        return world_map

    def _get_render_data(self):
        render_data = []

        for wall in self.world_map['lines']:
            color = wall['color'].tolist()
            p1_proj = self.camera.project_point(wall['p1'])
            p2_proj = self.camera.project_point(wall['p2'])
            render_data.append(('line', (p1_proj, p2_proj, color, 2)))
        
        for obstacle in self.world_map['circles']:
            color = obstacle['color'].tolist()
            center_proj = self.camera.project_point(obstacle['center'])
            radius_proj = int(obstacle['radius'] / self.camera.meters_per_pixel)
            render_data.append(('circle', (center_proj, radius_proj, color, -1)))

        pos_x, pos_y, theta = self.robot.pose
        robot_center_proj = self.camera.project_point((pos_x, pos_y))
        robot_radius_proj = int(self.robot.radius / self.camera.meters_per_pixel)
        render_data.append(('robot', (robot_center_proj, robot_radius_proj, (0, 255, 0), theta)))

        return render_data
    
    def _is_colliding_with_circle(self, robot_pos, robot_radius, circle_pos, circle_radius):
        distance = np.linalg.norm(robot_pos - circle_pos)
        return distance < (robot_radius + circle_radius)

    def _is_colliding_with_line(self, robot_pos, robot_radius, p1, p2):
        line_vec = p2 - p1
        p_to_p1_vec = robot_pos - p1
        
        line_length_sq = np.dot(line_vec, line_vec)
        if line_length_sq == 0: 
            return np.linalg.norm(p_to_p1_vec) < robot_radius
        
        t = np.dot(p_to_p1_vec, line_vec) / line_length_sq
        t = np.clip(t, 0.0, 1.0)
        
        closest_point = p1 + t * line_vec
        
        distance = np.linalg.norm(closest_point - robot_pos)
        
        return distance < robot_radius

    def _check_collision(self, next_robot_pose):
        next_pos = next_robot_pose[:2]
        
        for circle in self.world_map['circles']:
            if self._is_colliding_with_circle(next_pos, self.robot.radius, circle['center'], circle['radius']):
                return True
        
        for line in self.world_map['lines']:
            if self._is_colliding_with_line(next_pos, self.robot.radius, line['p1'], line['p2']):
                return True
                
        return False
    
    def _get_line_circle_intersection(self, ray_start, ray_direction, circle_center, circle_radius, max_dist):
        m = ray_direction
        c = ray_start - circle_center
        
        a = np.dot(m, m)
        b = 2 * np.dot(m, c)
        C = np.dot(c, c) - circle_radius**2
        
        delta = b**2 - 4*a*C
        
        if delta < 0:
            return None
            
        t1 = (-b + np.sqrt(delta)) / (2*a)
        t2 = (-b - np.sqrt(delta)) / (2*a)
        
        t_min = min(t1, t2)
        t_max = max(t1, t2)

        if t_min > 0 and t_min <= max_dist:
            return ray_start + t_min * ray_direction
        elif t_max > 0 and t_max <= max_dist:
            return ray_start + t_max * ray_direction
        
        return None    
    
    def _get_line_line_intersection(self, p1, p2, p3, p4):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4

        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if den == 0:
            return None
            
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den

        if 0 < t < 1 and 0 < u:
            intersection_point = np.array([x1 + t * (x2 - x1), y1 + t * (y2 - y1)])
            return intersection_point
        
        return None
    
    def get_lidar(self):
        robot_pos = self.robot.pose[:2]
        robot_theta = self.robot.pose[2]
        
        lidar_readings = []
        
        for angle in self.lidar_angles:
            ray_angle = robot_theta + angle
            ray_direction = np.array([np.cos(ray_angle), np.sin(ray_angle)])
            
            closest_point = None
            min_dist = float('inf')
            hit_color = None

            for wall in self.world_map['lines']:
                intersection = self._get_line_line_intersection(robot_pos, robot_pos + ray_direction * self.lidar_max_range, wall['p1'], wall['p2'])
                if intersection is not None:
                    dist = np.linalg.norm(intersection - robot_pos)
                    if dist < min_dist:
                        min_dist = dist
                        closest_point = intersection
                        hit_color = wall['color']
            
            for circle in self.world_map['circles']:
                intersection = self._get_line_circle_intersection(robot_pos, ray_direction, circle['center'], circle['radius'], self.lidar_max_range)
                if intersection is not None:
                    dist = np.linalg.norm(intersection - robot_pos)
                    if dist < min_dist:
                        min_dist = dist
                        closest_point = intersection
                        hit_color = circle['color']

            if closest_point is not None:
                lidar_readings.append({
                    'angle': float(angle),
                    'distance': float(min_dist),
                    #'point_world': closest_point,
                    'color': hit_color.tolist()
                })
        
        return lidar_readings
    
    def get_range(self):
        robot_pos = self.robot.pose[:2]
        robot_theta = self.robot.pose[2]
        
        ray_angle = robot_theta
        ray_direction = np.array([np.cos(ray_angle), np.sin(ray_angle)])
        
        closest_point = None
        min_dist = float('inf')
        
        for wall in self.world_map['lines']:
            intersection = self._get_line_line_intersection(robot_pos, robot_pos + ray_direction * self.lidar_max_range, wall['p1'], wall['p2'])
            if intersection is not None:
                dist = np.linalg.norm(intersection - robot_pos)
                if dist < min_dist:
                    min_dist = dist
                    closest_point = intersection

        for circle in self.world_map['circles']:
            intersection = self._get_line_circle_intersection(robot_pos, ray_direction, circle['center'], circle['radius'], self.lidar_max_range)
            if intersection is not None:
                dist = np.linalg.norm(intersection - robot_pos)
                if dist < min_dist:
                    min_dist = dist
                    closest_point = intersection
        
        if closest_point is not None:
            return min_dist
        else:
            return self.lidar_max_range

    def step(self, des_v, des_w):
        self.robot.des_v = des_v
        self.robot.des_w = des_w
        
        self.robot.update_speed()
        
        next_robot_pose = self.robot.update_pose()
        
        if not self._check_collision(next_robot_pose):
            self.robot.pose = next_robot_pose
        else:
            self.robot.v = 0
            self.robot.w = 0

        render_data = self._get_render_data()

        rendered_img = self.renderer.render_image(render_data)

        return rendered_img
    
def render_lidar_data(lidar_data, max_range, resolution):
    img_width, img_height = int(2.1*max_range/resolution), int(2.1*max_range/resolution)
    image = np.full((img_height, img_width, 3), (255, 255, 255), dtype=np.uint8)
    center = (img_width // 2, img_height // 2)

    for reading in lidar_data:
        dist = reading['distance']
        angle = reading['angle']
        color = reading['color']
        
        point_x = int(center[0] + dist / resolution * np.cos(angle - np.pi/2))
        point_y = int(center[1] + dist / resolution * np.sin(angle - np.pi/2))
        
        cv2.circle(image, (point_x, point_y), 2, tuple(reversed(color)), -1)

    cv2.circle(image, center, 10, (0, 255, 0), -1)

    line_length = 10
    end_x = int(center[0] + line_length * np.cos(0 - np.pi/2))
    end_y = int(center[1] + line_length * np.sin(0 - np.pi/2))
    cv2.line(image, center, (end_x, end_y), (0, 0, 0), 2)

    #max_range_pixels = int(max_range / resolution)
    #cv2.circle(image, center, max_range_pixels, (100, 100, 100), 1)

    return image

if __name__ == "__main__":

    world_map = {'lines': [], 'circles': []}
    
    walls = [
        (( -4.0, -4.0), ( 4.0, -4.0), (0, 0, 0)),
        ((  4.0, -4.0), ( 4.0,  4.0), (0, 0, 0)),
        ((  4.0,  4.0), (-4.0,  4.0), (0, 0, 0)),
        (( -4.0,  4.0), (-4.0, -4.0), (0, 0, 0)),
        ((  1.0,  1.0), ( 1.0,  2.0), (0, 0, 0))
    ]
    
    circles = [
        (( 1.5,  1.5,  0.1), (  0,   0, 200)),
        ((-1.5,  1.5,  0.1), (  0, 200, 200)),
        ((-1.5, -1.5,  0.1), (200,   0, 200)),
        (( 1.5, -1.5,  0.1), (200,   0,   0))
    ]
    
    for p1_world, p2_world, color in walls:
        world_map['lines'].append({'p1': np.array(p1_world), 'p2': np.array(p2_world), 'color': np.array(color)})
        
    for circle, color in circles:
        x, y, r = circle
        world_map['circles'].append({'center': np.array([x, y]), 'radius': r, 'color': np.array(color)})
    
    camera_params = {'width' : 800,
                     'height': 800,
           'meters_per_pixel': 0.01}
    
    time_step = 0.1

    robot_params = {
        'initial_pose': np.array([0.0, 0.0, 0.0]),
        'a_max': 0.5,
        'alpha_max': 0.5,
        'dt': time_step,
        'radius': 0.15
    }

    sim = Simulator(robot_params, camera_params, time_step, world_map)

    des_v, des_w = 0.0, 0.0

    # -------------------------- Settings --------------------------
    # Warning running odometry and velocity motion model together is not supported. 
    # Same with odometry and velocity sample algorithms.
    noise = 0.1 # Set alle noise parameters equal, only works when random is false
    random_noise = True # default is True
    noise_max = 0.1
    noise_min = 0.01

    # Motion model settings
    plot_offset = 10 # How many time time steps to offset plotting. Low numbers cause low FPS. Default 10
    use_triangle = False

    # Sample model settings
    sample_interval = 60 # How many time steps to wait before plotting the sample points
    sample_plot_amount = 10

    # Odometry probabilty density specific settings
    plot_p_odo_density = False
    if plot_p_odo_density:
        sweep_upper = 0.01 # default is 0.0004
        sweep_lower = 0.0 # default is 0.0
        sweep_step = 0.0001 # default is 0.000002

    # Odometry sample specific settings
    plot_sample_odo = False

    # Velocity sample specific settings
    plot_sample = False
    
    # Velocity probability density specific settings
    plot_p_vel_density = False
    if plot_p_vel_density:
        sweep_upper = 0.0004 # default is 0.0004
        sweep_lower = 0.0 # default is 0.0
        sweep_step = 0.000002 # default is 0.000002
    # -------------------------- Settings --------------------------

    first_run = True
    if random_noise == True:
        rng = np.random.default_rng()
        robot_params_model = [robot_params['dt'],
                            rng.uniform(noise_min, noise_max),
                            rng.uniform(noise_min, noise_max),
                            rng.uniform(noise_min, noise_max),
                            rng.uniform(noise_min, noise_max),
                            rng.uniform(noise_min, noise_max),
                            rng.uniform(noise_min, noise_max),
                               ]
        for idx, noise_param in enumerate(robot_params_model[1:]):
            print(f"Alpha {idx+1} is {noise_param:.2f}")
    else:
        robot_params_model = [robot_params['dt'], noise, noise, noise, noise, noise, noise]

    prev_pos = sim.robot.get_pose()
    prev_sample_pos = sim.robot.get_pose()
    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([], [])
    ax.set_ylim(0, 100)
    ax.set_xlabel('x offset [m]')
    ax.set_ylabel('Probability [%]')

    if plot_p_odo_density:
        ax.set_title('Odometry model predictions')

    if plot_p_vel_density:
        ax.set_title('Velocity model predictions')

    if plot_p_odo_density and plot_p_vel_density:
        raise Exception("Incompatible algorithms chosen")

    if plot_sample and plot_sample_odo:
        raise Exception("Incompatible algorithms chosen")


    current_iteration = 0

    while True:
        current_iteration += 1
        rendered_img = sim.step(des_v, des_w)

        lidar_data = sim.get_lidar()
        range_finder = sim.get_range()
        true_pos = sim.robot.get_pose()
        true_vel = sim.robot.get_vel()

        rendered_lidar = render_lidar_data(lidar_data, sim.lidar_max_range, sim.camera.meters_per_pixel)
        cv2.imshow('Lidar', cv2.cvtColor(rendered_lidar, cv2.COLOR_RGB2BGR))

        # Sample algorithm
        if (plot_sample_odo or plot_sample) and current_iteration % sample_interval == 0 and current_iteration <= sample_interval*(sample_plot_amount+1):
            if first_run == True:
                first_run = False
            else:
                if plot_sample == True:
                    sim.renderer.add_line(prev_sample_pos, true_pos, true_vel, robot_params_model, "vel")

                elif plot_sample_odo == True:
                    odometry_u_t = (prev_sample_pos, true_pos)
                    sim.renderer.add_line(prev_sample_pos, true_pos, true_vel, robot_params_model, "odo", odometry_u_t)

            prev_time_sample = current_iteration
            prev_sample_pos = true_pos

        rendered_img = sim.renderer.render_sample_model(rendered_img)
        cv2.imshow('TEL310: 2D Simulation Tool', cv2.cvtColor(rendered_img, cv2.COLOR_RGB2BGR))



        key = cv2.waitKey(int(1/time_step)*2)

        if key == ord('w'): des_v += 0.1
        elif key == ord('s'): des_v -= 0.1
        elif key == ord('a'): des_w += 0.1
        elif key == ord('d'): des_w -= 0.1
        elif key == ord('h'): des_w = des_v = 0
        elif key == ord('r'): des_v = des_w = 0.0; sim.robot.pose = np.array([0.0, 0.0, 0.0])
        
        if key == ord('q'): break

        # Odometry motion algorithm
        if plot_p_odo_density and not np.all(prev_pos == true_pos) and current_iteration % plot_offset == 0:
            prev_time_plot = current_iteration
            neg_distro = np.zeros(int(round(sweep_upper/(sweep_step), 0))) 
            pos_distro = np.zeros(int(round(sweep_upper/(sweep_step), 0))) 

            model_offsets = np.arange(sweep_lower, sweep_upper, sweep_step)
            odometry_u_t = (prev_pos, true_pos)
            for idx, increment in enumerate(model_offsets):
                neg_distro[idx] = L1.motion_model_odometry(prev_pos, [true_pos[0]-increment, true_pos[1], true_pos[2]], odometry_u_t, robot_params_model, use_triangle)
                pos_distro[idx] = L1.motion_model_odometry(prev_pos, [true_pos[0]+increment, true_pos[1], true_pos[2]], odometry_u_t, robot_params_model, use_triangle)

            result = np.hstack((np.flip(neg_distro), pos_distro))

            # Skip plotting when NaN is found inside the result array
            if not np.isnan(result).any():    
                norm_result = result / result.sum() * 100
                plot_offsets = np.hstack((np.flip(model_offsets*-1), model_offsets))
                line.set_data(plot_offsets, norm_result)
                ax.relim()
                ax.autoscale_view()
                fig.canvas.draw_idle()
                fig.canvas.flush_events()


        # Velocity motion algorithm
        if plot_p_vel_density and not np.all(prev_pos == true_pos) and current_iteration % plot_offset == 0:
            prev_time_plot = current_iteration
            neg_distro = []
            pos_distro = []

            model_offsets = np.arange(sweep_lower, sweep_upper, sweep_step)
            for idx, increment in enumerate(model_offsets):
                neg_distro.append(L1.motion_model_velocity(prev_pos, [true_pos[0]-increment, true_pos[1], true_pos[2]], true_vel, robot_params_model, use_triangle))
                pos_distro.append(L1.motion_model_velocity(prev_pos, [true_pos[0]+increment, true_pos[1], true_pos[2]], true_vel, robot_params_model, use_triangle))

            neg_distro = np.array(neg_distro)
            pos_distro = np.array(pos_distro)
            result = np.hstack((np.flip(neg_distro), pos_distro))
            norm_result = result / result.sum() * 100
            plot_offsets = np.hstack((np.flip(model_offsets*-1), model_offsets))
            line.set_data(plot_offsets, norm_result)
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw_idle()
            fig.canvas.flush_events()



        prev_pos = true_pos


    cv2.destroyAllWindows()