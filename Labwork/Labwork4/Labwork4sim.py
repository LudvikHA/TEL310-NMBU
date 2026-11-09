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
import Labwork1.Labwork1 as L1
import Labwork2.Labwork2_Algorithms as L2
import Labwork3.Labwork3_Algorithms as L3
import Labwork4.Labwork4_Algorithms as L4
import matplotlib.pyplot as plt
import time

rng = np.random.default_rng()

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

        self.prev_cat_pose = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.pose_to_replace = 0

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

                # Draw each ray casting line
                for offset in sim.ray_angles:
                    end_point_x = center[0] + int(sim.lidar_max_range*100 * np.cos(theta+offset))
                    end_point_y = center[1] - int(sim.lidar_max_range*100 * np.sin(theta+offset))
                    cv2.line(image, center, (end_point_x, end_point_y), (0, 0, 255), 2)

                cv2.circle(image, center, radius, tuple(reversed(color)), -1)
                
                end_point_x = center[0] + int(radius * np.cos(theta))
                end_point_y = center[1] - int(radius * np.sin(theta))
                cv2.line(image, center, (end_point_x, end_point_y), (0, 0, 0), 2)



            elif obj_type == 'cat':
                center, radius, color, theta = data
                center = np.array(center)
                
                end_point_x = center[0] + int(radius * np.cos(theta))
                end_point_y = center[1] - int((radius+20) * np.sin(theta))
                cv2.circle(image, center, radius, color, thickness=-1, lineType=cv2.LINE_8)
                cv2.line(image, center, (end_point_x, end_point_y), (0, 0, 0), 2)

            
        return image

class Simulator:
    def __init__(
        self, 
        robot_init_pose, 
        cam_params, 
        time_step, 
        world_map, 
        ray_cast_params: list, 
        ray_cast_noise, 
        cat_amount: int = 3, 
        lidar_max_range: float = 2.0,
        use_ekf: bool = False,
        ):

        self.camera = Camera(**cam_params)
        self.robot = Robot(**robot_init_pose)
        self.renderer = Renderer(self.camera.width, self.camera.height)
        self.dt = time_step

        self.ray_cast = ray_cast
        self.cats = [Robot(initial_pose=np.array([rng.uniform(-4, 4), rng.uniform(-4, 4), rng.uniform(0, np.pi*2)])) for x in range(cat_amount)]
        if use_ekf == False:
            self.ray_angles = np.array([(np.pi/4)*x for x in range(8)])
        else:
            self.ray_angles = [-np.pi/6, np.pi/6]

        self.use_ekf = use_ekf

        self.true_ray_cast = []
        self.noisy_ray_cast = []
        self.intrinsic_params = ray_cast_params # Initial params
        self.intrinsic_params_learned = False
        self.ray_cast_noise = ray_cast_noise

        self.lidar_max_range = lidar_max_range
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

        for cat in self.cats:
            pos_x, pos_y, theta = cat.pose
            cat_center_proj = self.camera.project_point((pos_x, pos_y))
            cat_radius_proj = int(cat.radius / self.camera.meters_per_pixel)
            render_data.append(('cat', (cat_center_proj, cat_radius_proj, (255, 0, 0), theta)))


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

    def _check_collision_robot(self, next_robot_pose):
        next_pos = next_robot_pose[:2]
        
        for circle in self.world_map['circles']:
            if self._is_colliding_with_circle(next_pos, self.robot.radius, circle['center'], circle['radius']):
                return True
        
        for line in self.world_map['lines']:
            if self._is_colliding_with_line(next_pos, self.robot.radius, line['p1'], line['p2']):
                return True

        for cat in self.cats:
            if self._is_colliding_with_circle(next_pos, self.robot.radius, cat.get_pose()[:2], cat.radius):
                return True
                
        return False

    def _check_collision_cat(self, next_robot_pose):
        next_pos = next_robot_pose[:2]
        
        for circle in self.world_map['circles']:
            if self._is_colliding_with_circle(next_pos, self.robot.radius, circle['center'], circle['radius']):
                return True
        
        for line in self.world_map['lines']:
            if self._is_colliding_with_line(next_pos, self.robot.radius, line['p1'], line['p2']):
                return True    

        for cat in self.cats:
            if self._is_colliding_with_circle(next_pos, cat.radius, self.robot.get_pose()[:2], self.robot.radius):
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

        if 0 < t < 1 and 0 <= u <= 1:
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

            for cat in self.cats:
                intersection = self._get_line_circle_intersection(robot_pos, ray_direction, cat.get_pose()[:2], cat.radius, self.lidar_max_range)
                if intersection is not None:
                    dist = np.linalg.norm(intersection - robot_pos)
                    if dist < min_dist:
                        min_dist = dist
                        closest_point = intersection
                        hit_color = np.array((0, 0, 255))

            if closest_point is not None:
                lidar_readings.append({
                    'angle': float(angle),
                    'distance': float(min_dist),
                    #'point_world': closest_point,
                    'color': hit_color.tolist()
                })
        
        return lidar_readings

    def _get_true_ray_cast(self, pose = None):
        if pose is None:
            pose = self.robot.get_pose()

        robot_pos = pose[:2]
        robot_theta = pose[2]
        
        ray_cast_readings = []

        for angle in self.ray_angles:
            ray_direction = np.array([np.cos(robot_theta-angle), np.sin(robot_theta-angle)])

            
            closest_point = None
            min_dist = float('inf')

            for wall in self.world_map['lines']:
                intersection = self._get_line_line_intersection(robot_pos, robot_pos + ray_direction * self.lidar_max_range, wall['p1'], wall['p2'])
                if intersection is not None:
                    dist = np.linalg.norm(intersection - robot_pos)
                    if dist < min_dist:
                        min_dist = dist
                        closest_point = min_dist
                        # closest_point = np.abs(intersection-robot_pos)[1]
            
            for circle in self.world_map['circles']:
                intersection = self._get_line_circle_intersection(robot_pos, ray_direction, circle['center'], circle['radius'], self.lidar_max_range)
                if intersection is not None:
                    dist = np.linalg.norm(intersection - robot_pos)
                    if dist < min_dist:
                        min_dist = dist
                        closest_point = min_dist
                        # closest_point = np.abs(intersection-robot_pos)[1]

            if closest_point is not None:
                ray_cast_readings.append(closest_point)
            else:
                ray_cast_readings.append(self.lidar_max_range)
                
        
        return np.array(ray_cast_readings)
    
    def _get_noisy_ray_cast(self, pose = None):
        if pose is None:
            pose = self.robot.get_pose()

        robot_pos = pose[:2]
        robot_theta = pose[2]
        
        ray_cast_readings = []

        for angle in self.ray_angles:
            ray_direction = np.array([np.cos(robot_theta-angle), np.sin(robot_theta-angle)])

            
            closest_point = None
            min_dist = float('inf')

            for wall in self.world_map['lines']:
                intersection = self._get_line_line_intersection(robot_pos, robot_pos + ray_direction * self.lidar_max_range, wall['p1'], wall['p2'])
                if intersection is not None:
                    dist = np.linalg.norm(intersection - robot_pos)
                    if dist < min_dist:
                        min_dist = dist
                        closest_point = np.clip(min_dist+rng.normal(0, self.ray_cast_noise), 0, self.lidar_max_range)
                        # closest_point = np.abs(intersection-robot_pos)[1]
            
            for circle in self.world_map['circles']:
                intersection = self._get_line_circle_intersection(robot_pos, ray_direction, circle['center'], circle['radius'], self.lidar_max_range)
                if intersection is not None:
                    dist = np.linalg.norm(intersection - robot_pos)
                    if dist < min_dist:
                        min_dist = dist
                        closest_point = np.clip(min_dist+rng.normal(0, self.ray_cast_noise), 0, self.lidar_max_range)
                        # closest_point = np.abs(intersection-robot_pos)[1]

            for cat in self.cats:
                intersection = self._get_line_circle_intersection(robot_pos, ray_direction, cat.get_pose()[:2], cat.radius, self.lidar_max_range)
                if intersection is not None:
                    dist = np.linalg.norm(intersection - robot_pos)
                    if dist < min_dist:
                        min_dist = dist
                        closest_point = np.clip(min_dist+rng.normal(0, self.ray_cast_noise), 0, self.lidar_max_range)
            if closest_point is not None:
                ray_cast_readings.append(closest_point)
            else:
                ray_cast_readings.append(self.lidar_max_range)
                
        
        return np.array(ray_cast_readings)     
    
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

    def learn_params(self, readings_true, readings_noise, readings_min):
        if len(readings_true) >= readings_min:
            readings_true = np.array(readings_true) 
            readings_noise = np.array(readings_noise) 
            self.intrinsic_params, self.intrinsic_params_learned = L2.learn_intrinsic_parameters(
                readings_true,
                readings_noise,
                self.lidar_max_range,
                self.intrinsic_params)

        else: 
            print("Gathering readings for intrinsic parameters...")

    def get_ray_cast(self):
        img_width, img_height = int(4.2*self.lidar_max_range/sim.camera.meters_per_pixel), int(2.1*self.lidar_max_range/sim.camera.meters_per_pixel)
        image = np.full((img_height, img_width, 3), (255, 255, 255), dtype=np.uint8)

        self.true_ray_cast = self._get_true_ray_cast()
        self.noisy_ray_cast = self._get_noisy_ray_cast()

        cv2.putText(image, f"True ray cast:", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(image, f"Noisy ray cast:", (400, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(image, f"Intrinsic parameters learned: {self.intrinsic_params_learned}", (30, 40+40*(len(self.true_ray_cast)+1)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)
        for idx, ray in enumerate(zip(self.true_ray_cast, self.noisy_ray_cast)):
            true_cast, noisy_cast = ray
            cv2.putText(image, f"Ray {idx+1}: {true_cast:.2f}", (30, 40+40*(idx+1)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f"Ray {idx+1}: {noisy_cast:.2f}", (400, 40+40*(idx+1)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)
        return image

    def plot_beam_range_finder_model(self):
        model_offsets = np.arange(sweep_lower, sweep_upper, sweep_step)

        neg_distro = np.zeros(model_offsets.size) 
        pos_distro = np.zeros(model_offsets.size) 

        for idx, offset in enumerate(model_offsets):
            # Offset pose by sweep step
            offset_pose_pos = self.robot.pose.copy()
            offset_pose_pos[0] += offset

            offset_pose_neg = self.robot.pose.copy()
            offset_pose_neg[0] -= offset

            # Calculate probability for positive and negative side
            offset_true_ray_cast = self._get_true_ray_cast(pose=offset_pose_pos)
            pos_distro[idx] = L2.beam_range_finder_model(offset_true_ray_cast, self.noisy_ray_cast, self.lidar_max_range, self.intrinsic_params)

            offset_true_ray_cast = self._get_true_ray_cast(pose=offset_pose_neg)
            neg_distro[idx] = L2.beam_range_finder_model(offset_true_ray_cast, self.noisy_ray_cast, self.lidar_max_range, self.intrinsic_params)

        result = np.hstack((np.flip(neg_distro), pos_distro))

        norm_result = result / result.sum() * 100
        plot_offsets = np.hstack((np.flip(model_offsets*-1), model_offsets))
        line.set_data(plot_offsets, norm_result)
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

    def plot_likelihood_field_range_finder_model(self):
        model_offsets = np.arange(sweep_lower, sweep_upper, sweep_step)

        neg_distro = np.zeros(model_offsets.size) 
        pos_distro = np.zeros(model_offsets.size) 

        for idx, offset in enumerate(model_offsets):
            # Offset pose by sweep step
            offset_pose_pos = self.robot.pose.copy()
            offset_pose_pos[0] += offset

            offset_pose_neg = self.robot.pose.copy()
            offset_pose_neg[0] -= offset

            # Calculate probability for positive and negative side
            pos_distro[idx] = L2.likelihood_field_range_finder_model(self.noisy_ray_cast, offset_pose_pos, self.world_map, -self.ray_angles, self.lidar_max_range, self.intrinsic_params)
            neg_distro[idx] = L2.likelihood_field_range_finder_model(self.noisy_ray_cast, offset_pose_neg, self.world_map, -self.ray_angles, self.lidar_max_range, self.intrinsic_params)

        result = np.hstack((np.flip(neg_distro), pos_distro))

        norm_result = result / result.sum() * 100
        plot_offsets = np.hstack((np.flip(model_offsets*-1), model_offsets))
        line.set_data(plot_offsets, norm_result)
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

    def _get_landmark_ray(self, bearing, pose = None, ):
        if pose is None:
            pose = self.robot.get_pose()

        robot_pos = pose[:2]
        robot_theta = pose[2]
        
        ray_direction = np.array([np.cos(robot_theta+bearing), np.sin(robot_theta+bearing)])

        closest_point = None
        min_dist = float('inf')

        for wall in self.world_map['lines']:
            intersection = self._get_line_line_intersection(robot_pos, robot_pos + ray_direction * self.lidar_max_range, wall['p1'], wall['p2'])
            if intersection is not None:
                dist = np.linalg.norm(intersection - robot_pos)
                if dist < min_dist:
                    min_dist = dist
                    closest_point = min_dist
        
        for circle in self.world_map['circles']:
            intersection = self._get_line_circle_intersection(robot_pos, ray_direction, circle['center'], circle['radius'], self.lidar_max_range)
            if intersection is not None:
                dist = np.linalg.norm(intersection - robot_pos)
                if dist < min_dist:
                    min_dist = dist
                    closest_point = min_dist

        for cat in self.cats:
            intersection = self._get_line_circle_intersection(robot_pos, ray_direction, cat.get_pose()[:2], cat.radius, self.lidar_max_range)
            if intersection is not None:
                dist = np.linalg.norm(intersection - robot_pos)
                if dist < min_dist:
                    min_dist = dist
                    closest_point = min_dist

        if closest_point is not None:
            return closest_point
        else:
            return self.lidar_max_range

    def get_landmark_measurements(self):
        x = self.robot.pose[0]
        y = self.robot.pose[1]
        theta = self.robot.pose[2]
        measurements = []
        for landmark_idx, landmark in enumerate(self.world_map["circles"]):
            x_landmark, y_landmark = landmark["center"]

            vec_robot_landmark = self.robot.pose[:2]-landmark["center"]
            dist_robot_landmark = np.linalg.norm(vec_robot_landmark) 

            bearing = np.arctan2(y_landmark - y, x_landmark - x) - theta
            bearing = np.arctan2(np.sin(bearing), np.cos(bearing))

            # Skip landmarks that are out of range
            ray_to_landmark = self._get_landmark_ray(bearing, self.robot.pose)
            if dist_robot_landmark < ray_detection_range:
                # Skip landmarks if they are not within cone of ekf
                if self.use_ekf == False or self.ray_angles[0]+theta < theta + bearing < self.ray_angles[1]+theta:
                    # Skip landmarks if they are blocked
                    if dist_robot_landmark-ray_to_landmark < landmark["radius"] + 0.01: 
                        dist_robot_landmark_noise = np.linalg.norm(vec_robot_landmark) + rng.normal(0, landmark_dist_noise)
                        bearing_noise = np.arctan2(y_landmark - y, x_landmark - x) - theta + rng.normal(0, landmark_bearing_noise)
                        bearing_noise = np.arctan2(np.sin(bearing_noise), np.cos(bearing_noise))

                        # As this is known correspondance s can be disregarded.

                        measurements.append([dist_robot_landmark_noise, bearing_noise, landmark_idx])
        return measurements

    def render_ekf_known(self, img, mu_prev, Sigma_prev, alphas, sigma_r, sigma_phi):
        """
        Render the uncertainty of EKF_localization_known_correspondence as ellipses beneath the robot
        """
        f = self.get_landmark_measurements()
        mu, Sigma, p_z = L3.EKF_localization_known_correspondence(
            mu_prev,
            Sigma_prev,
            (self.robot.des_v, self.robot.des_w), 
            f,
            self.world_map,
            alphas,
            self.dt,
            sigma_r,
            sigma_phi)

        eigvals, eigvec = np.linalg.eigh(Sigma[:2, :2])
        angle = -np.degrees(np.arctan2(eigvec[1, -1], eigvec[0, -1]))
        center = self.camera.project_point(mu[:2])

        # print(f"Eigvals: {eigvals}")
        # print(f"Eigvec: {eigvec}")

        colors = (( 60,  60,  60),   # dark grey
          (120, 120, 120),   # medium grey
          (180, 180, 180))

        for std, color in enumerate(colors):
            axes = (int(std * np.sqrt(eigvals[-1]) / self.camera.meters_per_pixel),
            int(std * np.sqrt(eigvals[0]) / self.camera.meters_per_pixel))
            cv2.ellipse(img, center, axes, angle, 0, 360, color=color, thickness=4)

        return img, mu, Sigma, p_z

    def render_ekf_local(self, img, mu_prev, Sigma_prev, alphas, sigma_r, sigma_phi, sigma_s):
        """
        Render the uncertainty of EKF_localization as ellipses beneath the robot
        """
        f = self.get_landmark_measurements()
        mu, Sigma = L3.EKF_localization(
            mu_prev,
            Sigma_prev,
            (self.robot.des_v, self.robot.des_w), 
            f,
            self.world_map,
            alphas,
            self.dt,
            sigma_r,
            sigma_phi,
            sigma_s)

        eigvals, eigvec = np.linalg.eigh(Sigma[:2, :2])
        angle = -np.degrees(np.arctan2(eigvec[1, -1], eigvec[0, -1]))
        center = self.camera.project_point(mu[:2])

        colors = (( 60,  60,  60),   
          (120, 120, 120),   
          (180, 180, 180))

        for std, color in enumerate(colors):
            axes = (int(std * np.sqrt(eigvals[-1]) / self.camera.meters_per_pixel),
            int(std * np.sqrt(eigvals[0]) / self.camera.meters_per_pixel))
            cv2.ellipse(img, center, axes, angle, 0, 360, color=color, thickness=4)

        return img, mu, Sigma

    def render_ukf_local(self, img, mu_prev, Sigma_prev, alphas, sigma_r, sigma_phi, gamma):
            """
            Render the uncertainty of EKF_localization as ellipses beneath the robot
            """
            f = self.get_landmark_measurements()
            mu, Sigma, p_z = L3.UKF_localization(
                mu_prev,
                Sigma_prev,
                (self.robot.des_v, self.robot.des_w), 
                f,
                self.world_map,
                alphas,
                self.dt,
                sigma_r,
                sigma_phi,
                gamma)

            eigvals, eigvec = np.linalg.eigh(Sigma[:2, :2])
            angle = -np.degrees(np.arctan2(eigvec[1, -1], eigvec[0, -1]))
            center = self.camera.project_point(mu[:2])

            colors = (( 60,  60,  60),   
            (120, 120, 120),   
            (180, 180, 180))

            for std, color in enumerate(colors):
                axes = (int(std * np.sqrt(eigvals[-1]) / self.camera.meters_per_pixel),
                int(std * np.sqrt(eigvals[0]) / self.camera.meters_per_pixel))
                cv2.ellipse(img, center, axes, angle, 0, 360, color=color, thickness=4)

            return img, mu, Sigma, p_z

    def render_landmark_likelihood_map(self, base_img):
        base_img = base_img.copy()

        measurements = self.get_landmark_measurements()

        map_coordinates_x = np.arange(-4, 4, 0.05)
        map_coordinates_y = np.arange(-4, 4, 0.05)
        grid = np.zeros((map_coordinates_x.size, map_coordinates_y.size))

        for x_idx, x_map in enumerate(map_coordinates_x):
            for y_idx, y_map in enumerate(map_coordinates_y):
                score = 1
                for f in measurements:
                    score = score * L2.landmark_model_known_correspondance(f, np.array([x_map, y_map, self.robot.pose[2]]), self.world_map, landmark_dist_noise, landmark_bearing_noise)
                
                grid[x_idx, y_idx] = score

        if grid.max() == 0:
            return None

        grid = np.flipud(grid.T)
        log_q = np.log(grid + 1e-300)
        clipped_q = np.clip(log_q - log_q.max(), -20, 0)

        grid = cv2.applyColorMap((((clipped_q + 20) / 20)*255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        grid = cv2.cvtColor(grid, cv2.COLOR_BGR2RGB)
        grid = cv2.resize(grid, (self.renderer.width, self.renderer.height), interpolation=cv2.INTER_LINEAR)
        grid = cv2.addWeighted(base_img, 0.5, grid, 0.5, 0)

        return grid

    def render_grid_localization_map(self, base_img, alphas, render_grid_local_interval, p=None):
        base_img = base_img.copy()

        measurements = self.get_landmark_measurements()

        map_coordinates_x = np.arange(-4, 4, 0.5) + 0.25
        map_coordinates_y = np.arange(-4, 4, 0.5) + 0.25
        map_heading_theta = np.arange(-np.pi, np.pi, np.deg2rad(30))
        if p is None:
            p = np.ones((map_coordinates_x.size, map_coordinates_y.size, map_heading_theta.size)) / (map_coordinates_x.size * map_coordinates_y.size * map_heading_theta.size)
        

        v, w = sim.robot.get_vel()
        p = L4.grid_localization(
            p,
            v,
            w,
            measurements,
            self.world_map,
            map_coordinates_x,
            map_coordinates_y,
            map_heading_theta,
            alphas,
            self.dt * render_grid_local_interval,
            0.2,
            0.2)
            
        grid = p.sum(axis=2)

        grid = np.flipud(grid.T)
        log_q = np.log(grid + 1e-300)
        clipped_q = np.clip(log_q - log_q.max(), -5, 0)

        grid = cv2.applyColorMap((((clipped_q + 5) / 5)*255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        grid = cv2.cvtColor(grid, cv2.COLOR_BGR2RGB)
        grid = cv2.resize(grid, (self.renderer.width, self.renderer.height), interpolation=cv2.INTER_NEAREST)
        grid = cv2.addWeighted(base_img, 0.5, grid, 0.5, 0)

        return grid, p

    def render_mcl(self, base_img, X, alphas, dt, sigma_r, sigma_phi, MCL_interval):
        base_img = base_img.copy()

        measurements = self.get_landmark_measurements()
        v, w = self.robot.get_vel()

        X = L4.MCL(X, v, w, measurements, self.world_map, alphas, dt, sigma_r, sigma_phi, MCL_interval)

        for x, y, theta in X:
            center = self.camera.project_point((x, y))
            cv2.circle(base_img, center, radius=2, color=(255, 0, 0), thickness=-1)

        return base_img, X

    def render_augmented_mcl(self, base_img, X, w_slow, w_fast, alphas, dt, sigma_r, sigma_phi, alpha_slow, alpha_fast, MCL_interval):
        base_img = base_img.copy()

        measurements = self.get_landmark_measurements()
        v, w = self.robot.get_vel()

        X, w_slow, w_fast = L4.augmented_MCL(
            X, v, w, measurements, self.world_map, alphas, dt, sigma_r, sigma_phi,
            w_slow, w_fast, alpha_slow, alpha_fast, [-4, 4, -4, 4], MCL_interval)

        for x, y, theta in X:
            center = self.camera.project_point((x, y))
            cv2.circle(base_img, center, radius=2, color=(0, 150, 0), thickness=-1)

        return base_img, X, w_slow, w_fast

    def render_kld_mcl(self, base_img, X, w, alphas, interval, sigma_r, sigma_phi, epsilon, delta, bin_size, M_min, M_max):
        base_img = base_img.copy()

        measurements = self.get_landmark_measurements()
        v, w_vel = self.robot.get_vel()

        X, w = L4.KLD_sampling_MCL(
            X, w, v, w_vel, measurements, self.world_map, alphas, self.dt, interval,
            sigma_r, sigma_phi, epsilon, delta, bin_size, M_min, M_max)

        for x, y, theta in X:
            center = self.camera.project_point((x, y))
            cv2.circle(base_img, center, radius=2, color=(200, 0, 200), thickness=-1)

        cv2.putText(base_img, f"Particles: {len(X)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)

        return base_img, X, w

    def render_test_range_measurement(self, base_img, X, chi, n_particles=200):
        base_img = base_img.copy()

        if len(X) > n_particles:
            X = X[rng.choice(len(X), n_particles, replace=False)]

        z_noisy = self._get_noisy_ray_cast()
        z_star_all = np.array([self._get_true_ray_cast(pose=x) for x in X])

        x, y, theta = self.robot.get_pose()
        start = self.camera.project_point((x, y))

        for k, angle in enumerate(self.ray_angles):
            accept = L4.test_range_measurement(z_noisy[k], z_star_all[:, k], self.lidar_max_range, self.intrinsic_params, chi)
            end = self.camera.project_point((x + z_noisy[k] * np.cos(theta - angle), y + z_noisy[k] * np.sin(theta - angle)))
            color = (0, 200, 0) if accept else (255, 0, 0)
            cv2.line(base_img, start, end, color, 3)

        return base_img

    def render_landmark_samples(self, base_img, samples = 200):
        base_img = base_img.copy()

        measurements = self.get_landmark_measurements()
        circle_color = [(255, 165, 0), (255, 192, 203), (34, 139, 34), (0, 215, 255), (19, 69, 139), (235, 206, 135)]
        for f, color in zip(measurements, circle_color):
            for _ in range(samples):
                sample_pose = L2.sample_landmark_model_known_correspondance(f, self.world_map, landmark_dist_noise, landmark_bearing_noise)
                center = self.camera.project_point(sample_pose[:2])
                cv2.circle(base_img, center, radius=3, color=color, thickness=-1)

        return base_img

    def step(self, des_v, des_w):
        self.robot.des_v = des_v
        self.robot.des_w = des_w
        self.robot.update_speed()
        next_robot_pose = self.robot.update_pose()

        if not self._check_collision_robot(next_robot_pose):
            self.robot.pose = next_robot_pose
        else:
            self.robot.v = 0
            self.robot.w = 0


        for cat in self.cats:  
            cat.des_v = rng.choice((0, 0, 0, 0.2, 0.2, 0.2, 0.4, 0.4, 0.6))
            cat.des_w = rng.choice((0, 0, -0.2, -0.2, 0.2, 0.2, -0.4, 0.4, -0.6, 0.6))
            cat.update_speed()
            next_robot_pose = cat.update_pose()

            if not self._check_collision_cat(next_robot_pose):
                cat.pose = next_robot_pose
            else:
                cat.v = 0
                cat.w = 0
                cat_x, cat_y, cat_theta = cat.get_pose()
                cat.pose = np.array([cat_x, cat_y, cat_theta+np.pi])

                # Check if cats are stuck
                cat.prev_cat_pose[cat.pose_to_replace] = cat.pose[:1].copy()
                cat.pose_to_replace += 1
                if cat.pose_to_replace == len(cat.prev_cat_pose):
                    cat.pose_to_replace = 0
                    if np.all(cat.prev_cat_pose[1:] - cat.prev_cat_pose[0] < 0.001):
                        print("A cat is stuck and has been saved!")
                        cat.pose = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(0, np.pi*2)])
            
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
        # Outer boundry
        (( -4.0, -4.0), ( 4.0, -4.0), (0, 0, 0)),
        ((  4.0, -4.0), ( 4.0,  4.0), (0, 0, 0)),
        ((  4.0,  4.0), (-4.0,  4.0), (0, 0, 0)),
        (( -4.0,  4.0), (-4.0, -4.0), (0, 0, 0)),
]
    
    circles = [
        (( 0.000,  3.000, 0.2), (200,   0,   0)),  # 0: red
        (( 2.853,  0.927, 0.2), (  0,   0, 200)),  # 1: blue
        (( 1.763, -2.427, 0.2), (  0, 160,   0)),  # 2: green
        ((-1.763, -2.427, 0.2), (230, 150,   0)),  # 3: orange
        ((-2.853,  0.927, 0.2), (160,   0, 160)),  # 4: purple
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
        'initial_pose': np.array([0, 0, 0]),
        'a_max': 0.5,
        'alpha_max': 0.5,
        'dt': time_step,
        'radius': 0.15
    }



    # -------------------------- Settings --------------------------
    # Common settings
    ray_detection_range = 2.0 # In meters. Default when using ray cast = 2.0 m, default when using landmark = 4.0 m 
    landmark_dist_noise = 0.05 # Default 0.05, ekf also uses this
    landmark_bearing_noise = 0.05 # Default 0.05, ekf also uses this
    landmark_index_noise = 0.05 # Default 0.05, ekf also uses this

    # Labwork 2 settings 
    # General settings
    plot_y_limit = 50 # range 0-100
    cat_amount = 0 # Amount of cats to spawn
    
    # Ray cast settings
    readings_min = 200 # Amount readings to collect before running learn_params()
    intrinsic_params_iteration_cap = 200
    ray_cast_noise = 0.05
    plot_offset_ray_cast = 60

    ray_cast = False # If true use ray casting
    plot_range_finder = False
    plot_field_range = False

    if plot_range_finder or plot_field_range:
        sweep_upper = 0.5 # default is 0.05
        sweep_lower = 0.0 # default is 0.0
        sweep_step = 0.005 # default is 0.005

    ray_cast_params = [0.7, 0.1, 0.1, 0.1, 0.2, 1] # Initial params: z_hit, z_short, z_max, z_rand, sigma_hit, lambda_short

    # Landmark settings

    render_landmark_heatmap = False
    render_landmark_sample = False
    render_offset_heatmap = 60

    # Labwork 3 settings 
    # General settings
    use_ekf = False # Changes to from ray casting beams to cone in front of robot

    # EKF localization known correspondence settings
    alphas_ekf_known = [0.01, 0.001, 0.001, 0.01]
    render_ekf_known = False

    # EKF localization settings
    alphas_ekf_unknown = [0.01, 0.001, 0.001, 0.01]
    render_ekf_unknown = False

    # UKF localization settings
    alphas_ukf_unknown = [0.01, 0.001, 0.001, 0.01]
    render_ukf_unknown = False 
    ukf_gamma = [1, 0, 2] # alpha_gamma, kappa_gamma, beta_gamma 
    
    # Labwork 4 settings
    # Grid local settings
    render_grid_local = False
    render_grid_local_interval = 40
    grid_local_alphas = [0.2, 0.05, 0.05, 0.2, 0.05, 0.05]

    # MCL settings
    render_mcl = False
    mcl_interval = 20
    mcl_particles = 1000
    mcl_alphas = [0.2, 0.05, 0.05, 0.2, 0.05, 0.05]
    mcl_sigma_r = 0.2
    mcl_sigma_phi = 0.2

    # AMCL settings
    render_augmented_mcl = False
    amcl_interval = 20
    amcl_particles = 1000
    amcl_alphas = [0.2, 0.05, 0.05, 0.2, 0.05, 0.05]
    amcl_sigma_r = 0.2
    amcl_sigma_phi = 0.2
    amcl_alpha_slow = 0.05
    amcl_alpha_fast = 0.5

    # KLD sampling settings
    render_kld_mcl = False
    kld_interval = 5
    kld_alphas = [0.2, 0.05, 0.05, 0.2, 0.05, 0.05]
    kld_sigma_r = 0.2
    kld_sigma_phi = 0.2
    kld_epsilon = 0.05
    kld_delta = 0.01
    kld_bin_size = [0.25, 0.25, np.deg2rad(15)]
    kld_M_min = 100
    kld_M_max = 3000

    # Test range settings
    render_test_range = True
    trm_chi = 0.5

    # -------------------------- Settings --------------------------

    readings_true = []
    readings_noise = []

    # Initial ekf values, ukf also uses these values for simplicity
    ekf_mu = robot_params["initial_pose"]
    ekf_Sigma = np.diag([0.01, 0.01, 0.01])
    ekf_p_z = 1.0
    rendered_img = None

    sim = Simulator(
        robot_params, 
        camera_params, 
        time_step, 
        world_map, 
        ray_cast_params, 
        ray_cast_noise, 
        cat_amount=cat_amount, 
        lidar_max_range=ray_detection_range,
        use_ekf=use_ekf,
    )

    des_v, des_w = 0.0, 0.0
    first_run = True
    p = None

    prev_pos = sim.robot.get_pose()
    prev_sample_pos = sim.robot.get_pose()
    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([], [])
    ax.set_ylim(0, plot_y_limit)
    ax.set_xlabel('x offset [m]')
    ax.set_ylabel('Probability [%]')

    current_iteration = 0

    mcl_X = np.column_stack([
        rng.uniform(-4, 4, mcl_particles),
        rng.uniform(-4, 4, mcl_particles),
        rng.uniform(-np.pi, np.pi, mcl_particles)])

    amcl_X = np.column_stack([
        rng.uniform(-4, 4, amcl_particles),
        rng.uniform(-4, 4, amcl_particles),
        rng.uniform(-np.pi, np.pi, amcl_particles)])
    amcl_w_slow = 0.0
    amcl_w_fast = 0.0

    kld_X = np.column_stack([
        rng.uniform(-4, 4, kld_M_max),
        rng.uniform(-4, 4, kld_M_max),
        rng.uniform(-np.pi, np.pi, kld_M_max)])
    kld_w = np.ones(kld_M_max)

    while True:
        current_iteration += 1


        rendered_img = sim.step(des_v, des_w)

        lidar_data = sim.get_lidar()
        range_finder = sim.get_range()
        true_pos = sim.robot.get_pose()
        true_vel = sim.robot.get_vel()

        rendered_lidar = render_lidar_data(lidar_data, sim.lidar_max_range, sim.camera.meters_per_pixel)
        cv2.imshow('Lidar', cv2.cvtColor(rendered_lidar, cv2.COLOR_RGB2BGR))

        if render_ekf_known == True:
            rendered_img, ekf_mu, ekf_Sigma, p_z = sim.render_ekf_known(rendered_img, ekf_mu, ekf_Sigma, alphas_ekf_known, landmark_dist_noise, landmark_bearing_noise)
            cv2.imshow('TEL310: 2D Simulation Tool', cv2.cvtColor(rendered_img, cv2.COLOR_RGB2BGR))

        elif render_ekf_unknown == True:
            rendered_img, ekf_mu, ekf_Sigma = sim.render_ekf_local(rendered_img, ekf_mu, ekf_Sigma, alphas_ekf_unknown, landmark_dist_noise, landmark_bearing_noise, landmark_index_noise)
            cv2.imshow('TEL310: 2D Simulation Tool', cv2.cvtColor(rendered_img, cv2.COLOR_RGB2BGR))

        elif render_ukf_unknown == True:
            rendered_img, ekf_mu, ekf_Sigma, p_z = sim.render_ukf_local(rendered_img, ekf_mu, ekf_Sigma, alphas_ukf_unknown, landmark_dist_noise, landmark_bearing_noise, ukf_gamma)
            cv2.imshow('TEL310: 2D Simulation Tool', cv2.cvtColor(rendered_img, cv2.COLOR_RGB2BGR))

        else:
            cv2.imshow('TEL310: 2D Simulation Tool', cv2.cvtColor(rendered_img, cv2.COLOR_RGB2BGR))

        if render_test_range == True and current_iteration % mcl_interval == 0:
            trm_img = sim.render_test_range_measurement(rendered_img, mcl_X, trm_chi)
            cv2.imshow('Test range measurement', cv2.cvtColor(trm_img, cv2.COLOR_RGB2BGR))

        if render_kld_mcl == True and current_iteration % kld_interval == 0:
            kld_img, kld_X, kld_w = sim.render_kld_mcl(
                rendered_img, kld_X, kld_w, kld_alphas, kld_interval,
                kld_sigma_r, kld_sigma_phi, kld_epsilon, kld_delta,
                kld_bin_size, kld_M_min, kld_M_max)
            cv2.imshow('KLD-sampling MCL particles', cv2.cvtColor(kld_img, cv2.COLOR_RGB2BGR))

        if render_augmented_mcl == True and current_iteration % amcl_interval == 0:
            amcl_img, amcl_X, amcl_w_slow, amcl_w_fast = sim.render_augmented_mcl(
                rendered_img, amcl_X, amcl_w_slow, amcl_w_fast, amcl_alphas,
                time_step * amcl_interval, amcl_sigma_r, amcl_sigma_phi,
                amcl_alpha_slow, amcl_alpha_fast, amcl_interval)
            cv2.imshow('Augmented MCL particles', cv2.cvtColor(amcl_img, cv2.COLOR_RGB2BGR))

        if render_mcl == True and current_iteration % mcl_interval == 0:
            mcl_img, mcl_X = sim.render_mcl(rendered_img, mcl_X, mcl_alphas, time_step * mcl_interval, mcl_sigma_r, mcl_sigma_phi, mcl_interval)
            cv2.imshow('MCL particles', cv2.cvtColor(mcl_img, cv2.COLOR_RGB2BGR))

        if render_grid_local == True and current_iteration % render_grid_local_interval == 0:
            grid_heatmap, p = sim.render_grid_localization_map(rendered_img, grid_local_alphas, render_grid_local_interval, p=p)
            if grid_heatmap is not None:
                cv2.imshow('Grid localization heatmap', cv2.cvtColor(grid_heatmap, cv2.COLOR_RGB2BGR))
  
        if ray_cast == True:
            rendered_outputs = sim.get_ray_cast()
            cv2.imshow('Ray Casting', cv2.cvtColor(rendered_outputs, cv2.COLOR_RGB2BGR))
            readings_true.append(sim.true_ray_cast)
            readings_noise.append(sim.noisy_ray_cast)
        
            if not sim.intrinsic_params_learned and len(readings_true) >= readings_min:
                for iteration in range(intrinsic_params_iteration_cap):
                    if sim.intrinsic_params_learned == True:
                        print("Intrinsic params have converged!")
                        break

                    print(f"Converging intrinsic params! {iteration+1}/{intrinsic_params_iteration_cap}")
                    sim.learn_params(readings_true, readings_noise, readings_min = readings_min)

                assert sim.intrinsic_params_learned, "Failed to converge. Aborting..."

            if current_iteration % plot_offset_ray_cast == 0 and sim.intrinsic_params_learned and plot_range_finder:
                ax.set_title(f'How likely readings are at robot pose: x: {sim.robot.pose[0]:.2f}, y: {sim.robot.pose[1]:.2f}, theta: {sim.robot.pose[2]:.2f}')
                sim.plot_beam_range_finder_model()

            elif current_iteration % plot_offset_ray_cast == 0 and sim.intrinsic_params_learned and plot_field_range:
                ax.set_title(f'How likely readings are at robot pose: x: {sim.robot.pose[0]:.2f}, y: {sim.robot.pose[1]:.2f}, theta: {sim.robot.pose[2]:.2f}')
                sim.plot_likelihood_field_range_finder_model()

            prev_time_sample = current_iteration
            prev_sample_pos = true_pos

        if render_landmark_heatmap and current_iteration % render_offset_heatmap == 0:
            landmark_heatmap = sim.render_landmark_likelihood_map(rendered_img)
            if landmark_heatmap is not None:
                cv2.imshow('Landmark likelihood map', cv2.cvtColor(landmark_heatmap, cv2.COLOR_RGB2BGR))

        if render_landmark_sample and current_iteration % render_offset_heatmap == 0:
            landmark_sample = sim.render_landmark_samples(rendered_img)
            cv2.imshow('Landmark samples', cv2.cvtColor(landmark_sample, cv2.COLOR_RGB2BGR))



        key = cv2.waitKey(int(1/time_step)*2)

        if key == ord('w'): des_v += 0.1
        elif key == ord('s'): des_v -= 0.1
        elif key == ord('a'): des_w += 0.1
        elif key == ord('d'): des_w -= 0.1
        elif key == ord('h'): des_w = des_v = 0
        elif key == ord('r'): des_v = des_w = 0.0; sim.robot.pose = np.array([0.0, 0.0, 0.0])
        
        if key == ord('q'): break


    cv2.destroyAllWindows()