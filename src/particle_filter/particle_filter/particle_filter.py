#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from tf2_ros import TransformBroadcaster, Buffer, TransformListener
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseArray, Pose, PoseStamped, Point, Quaternion, PoseWithCovarianceStamped
from tf_transformations import euler_from_quaternion, quaternion_from_euler
import random
import math
from visualization_msgs.msg import Marker, MarkerArray
from message_filters import Subscriber, ApproximateTimeSynchronizer
from sensor_msgs.msg import Image, CameraInfo


class ParticleFilter(Node):
    def __init__(self):
        super().__init__('particle_filter_localization')
        
        # Parameters
        self.declare_parameter('initial_pose_x', 0.0)
        self.declare_parameter('initial_pose_y', 0.0)
        self.declare_parameter('initial_pose_a', 0.0)
        self.declare_parameter('initial_cov_x', 0.5)
        self.declare_parameter('initial_cov_y', 0.5)
        self.declare_parameter('initial_cov_a', 0.1)
        
        # Increased number of particles for better representation
        self.num_particles = 100
        self.initial_pose_x = self.get_parameter('initial_pose_x').value
        self.initial_pose_y = self.get_parameter('initial_pose_y').value
        self.initial_pose_a = self.get_parameter('initial_pose_a').value
        self.initial_cov_x = self.get_parameter('initial_cov_x').value
        self.initial_cov_y = self.get_parameter('initial_cov_y').value
        self.initial_cov_a = self.get_parameter('initial_cov_a').value
        
        # Motion model noise parameters - REDUCED VALUES
        self.alpha1 = 0.1  # Rotation noise component
        self.alpha2 = 0.1  # Translation noise component
        self.alpha3 = 0.1  # Additional translation noise
        self.alpha4 = 0.1   # Additional rotation noise
        
        # Improved sensor model parameters
        self.z_hit = 0.85   # Increased weight for hit model
        self.z_rand = 0.15   # Decreased weight for random model
        self.sigma_hit = 0.1  # Reduced sigma for sharper distribution
        self.laser_max_range = 3.5
        
        # KLD-sampling parameters
        self.kld_err = 0.01
        self.kld_z = 0.99
        
        # Resampling parameters
        # Reduced threshold for less frequent resampling
        self.resampling_threshold = 0.3
        
        # Adaptive resampling parameters
        self.min_effective_particles = self.num_particles / 2.0
        
        # State variables
        self.particles = np.zeros((self.num_particles, 3))  # x, y, theta
        self.weights = np.ones(self.num_particles) / self.num_particles
        self.map_data = None
        self.map_info = None
        self.last_odom = None
        self.current_odom = None
        self.map_resolution = 0.05
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Low variance sampler state
        self.use_low_variance_sampler = True
        
        # Previous scan for scan matching
        self.previous_scan = None
        
        # Publishers
        self.particle_pub = self.create_publisher(PoseArray, '~/particle_filter/particle_cloud', 10)
        self.pose_pub = self.create_publisher(Odometry, '/pf/odom', 10)
        self.amcl_pose_pub = self.create_publisher(PoseWithCovarianceStamped, 'amcl_pose', 10)
        self.best_particle_marker_pub = self.create_publisher(Marker, 'best_particle', 10)
        
        # Subscribers
        self.initial_pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            'initialpose',
            self.initial_pose_callback,
            10
        )
        
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            rclpy.qos.qos_profile_sensor_data
        )
        self.get_logger().info("Waiting for map...")
        
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )
        
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            rclpy.qos.qos_profile_sensor_data
        )
        
        # Wait for map to be available
        while rclpy.ok() and self.map_data is None:
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().info("Map received!")
        
        # Precompute distance transform for faster sensor model
        self.distance_transform = None
        if self.map_data is not None:
            self.precompute_distance_transform()
        
        self.initialize_particles()
        self.update_timer = self.create_timer(0.1, self.timer_callback)

    def precompute_distance_transform(self):
        """Precompute distance transform for faster sensor model"""
        from scipy.ndimage import distance_transform_edt
        
        # Create obstacle map (1 for obstacles, 0 for free space)
        obstacle_map = np.zeros_like(self.map_data, dtype=np.uint8)
        obstacle_map[self.map_data > 50] = 1
        
        # Compute distance transform
        self.distance_transform = distance_transform_edt(1 - obstacle_map) * self.map_resolution
        self.get_logger().info("Distance transform precomputed")

    def initial_pose_callback(self, msg):
        """Handle initial pose estimate from RViz"""
        pose = msg.pose.pose
        self.initial_pose_x = pose.position.x
        self.initial_pose_y = pose.position.y
        _, _, self.initial_pose_a = euler_from_quaternion([
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w
        ])
        cov = msg.pose.covariance
        self.initial_cov_x = cov[0]
        self.initial_cov_y = cov[7]
        self.initial_cov_a = cov[35]
        self.initialize_particles()
        self.get_logger().info(f"Particles reinitialized to: x={self.initial_pose_x}, y={self.initial_pose_y}, θ={self.initial_pose_a}")
        
    def initialize_particles(self):
        """Vectorized particle initialization"""
        self.particles[:, 0] = self.initial_pose_x + np.random.normal(0, self.initial_cov_x*0.5, self.num_particles)
        self.particles[:, 1] = self.initial_pose_y + np.random.normal(0, self.initial_cov_y*0.5, self.num_particles)
        self.particles[:, 2] = self.initial_pose_a + np.random.normal(0, self.initial_cov_a*0.5, self.num_particles)
        self.particles[:, 2] = self.normalize_angles(self.particles[:, 2])
        
        # Vectorized validity check
        valid = np.array([self.is_valid_position(x, y) for x, y in self.particles[:, :2]])
        invalid_idx = np.where(~valid)[0]
        if len(invalid_idx) > 0:
            self.particles[invalid_idx, 0] = self.initial_pose_x
            self.particles[invalid_idx, 1] = self.initial_pose_y
            
        self.publish_particles()
        
    def map_callback(self, msg):
        """Process incoming map message"""
        self.map_data = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.map_info = msg.info
        self.map_resolution = msg.info.resolution
        self.get_logger().info(f"Map received: {msg.info.width}x{msg.info.height}, resolution: {self.map_resolution}")
        
        # Precompute distance transform when map changes
        self.precompute_distance_transform()
        
    def odom_callback(self, msg):
        """Process odometry data for motion update"""
        if self.current_odom is None:
            self.current_odom = msg
            self.last_odom = msg
            return
            
        self.last_odom = self.current_odom
        self.current_odom = msg
        
    def timer_callback(self):
        """Regular updates for motion model and publishing"""
        if self.last_odom is not None and self.current_odom is not None:
            # Perform motion update
            self.motion_update()
            
            # Publish current state
            self.publish_particles()
            self.publish_estimated_pose()
        
    def scan_callback(self, msg):
        """Process laser scan data for measurement update"""
        if len(self.particles) == 0 or self.map_data is None:
            return
            
        self.laser_max_range = msg.range_max
        
        # Perform measurement update using scan data
        self.measurement_update(msg)
        
        # Calculate effective sample size for adaptive resampling
        n_eff = 1.0 / np.sum(np.square(self.weights))
        
        # Resample only when effective sample size is low
        # This helps prevent particle depletion
        if n_eff < self.min_effective_particles:
            self.get_logger().debug(f"Resampling triggered. Effective sample size: {n_eff}")
            self.resample()
            
        # Store scan for future reference
        self.previous_scan = msg
            
        # Publish results
        self.publish_particles()
        self.publish_estimated_pose()
    
    def normalize_angles(self, angles):
        """Vectorized angle normalization"""
        return np.mod(angles + np.pi, 2*np.pi) - np.pi
    
    def motion_update(self):
        """Update particle positions based on odometry"""
        if self.last_odom is None or self.current_odom is None:
            return
            
        # Extract pose from odometry
        x1, y1, theta1 = self.get_pose_from_odometry(self.last_odom)
        x2, y2, theta2 = self.get_pose_from_odometry(self.current_odom)
        
        # Calculate odometry increment
        dx = x2 - x1
        dy = y2 - y1
        dtheta = self.normalize_angle(theta2 - theta1)
        
        # Convert to robot-relative motion
        trans = np.sqrt(dx**2 + dy**2)
        rot1 = np.arctan2(dy, dx) - theta1 if trans > 0.001 else 0
        rot2 = dtheta - rot1
        
        # Skip if no significant motion
        if trans < 0.001 and abs(dtheta) < 0.001:
            return
        
        # Update each particle with noise
        self.particles = np.array(self.particles)
    
        # Vectorized motion update
        noise_scale = np.clip(trans * 5 + np.abs(dtheta) * 5, 0, 1)
        
        # Generate all random values at once
        noise = np.random.normal(0, 1, (self.num_particles, 3))
        noisy_rot1 = rot1 + (self.alpha1 * np.abs(rot1) + self.alpha2 * trans) * noise[:, 0] * noise_scale
        noisy_trans = trans + (self.alpha3 * trans + self.alpha4 * (np.abs(rot1) + np.abs(rot2))) * noise[:, 1] * noise_scale
        noisy_rot2 = rot2 + (self.alpha1 * np.abs(rot2) + self.alpha2 * trans) * noise[:, 2] * noise_scale
        
        # Apply motion vectorized
        theta_new = self.normalize_angles(self.particles[:, 2] + noisy_rot1)
        self.particles[:, 0] += noisy_trans * np.cos(theta_new)
        self.particles[:, 1] += noisy_trans * np.sin(theta_new)
        self.particles[:, 2] = self.normalize_angles(theta_new + noisy_rot2)
        
        # Validity check and recovery
        valid = np.array([self.is_valid_position(x, y) for x, y in self.particles[:, :2]])
        invalid_idx = np.where(~valid)[0]
        if len(invalid_idx) > 0:
            best_idx = np.argmax(self.weights)
            self.particles[invalid_idx] = self.particles[best_idx] + np.random.normal(0, [0.3, 0.3, 0.15], (len(invalid_idx), 3))
            self.particles[invalid_idx, 2] = self.normalize_angles(self.particles[invalid_idx, 2])
        
    def measurement_update(self, scan_msg):
        """Update particle weights based on laser scan measurements"""
        if len(self.particles) == 0 or self.map_data is None:
            return
            
        angles = np.arange(scan_msg.angle_min, scan_msg.angle_max + scan_msg.angle_increment, scan_msg.angle_increment)
        
        # Use more beams for better accuracy, but still subsample for efficiency
        step = 10  # Use every 10th beam instead of 20th
        ranges = np.array(scan_msg.ranges[::step])
        scan_angles = angles[::step]
        
        # Calculate new weights for each particle
        new_weights = []
        for i, (px, py, ptheta) in enumerate(self.particles):
            # Calculate probability of this scan given the particle position
            p = self.get_scan_probability(px, py, ptheta, ranges, scan_angles, scan_msg)
            new_weights.append(p)
        
        # Normalize weights
        new_weights = np.array(new_weights)
        if np.sum(new_weights) > 0:
            new_weights = new_weights / np.sum(new_weights)
        else:
            # If all weights are zero, reset to uniform weights
            self.get_logger().warn("All particle weights are zero! Resetting to uniform.")
            new_weights = np.ones(len(self.particles)) / len(self.particles)
            
        # Apply exponential weighting to emphasize better particles
        # This helps reduce variance by making the weight distribution more peaky
        alpha = 1.0  # Tunable parameter: higher values increase peakiness
        new_weights = np.power(new_weights, alpha)
        if np.sum(new_weights) > 0:
            new_weights = new_weights / np.sum(new_weights)
        
        # Low-pass filter for weights to reduce jitter
        if hasattr(self, 'weights') and len(self.weights) == len(new_weights):
            beta = 0.3  # Weight blending factor (0-1)
            self.weights = beta * new_weights + (1 - beta) * self.weights
            self.weights = self.weights / np.sum(self.weights)
        else:
            self.weights = new_weights
            
    def get_scan_probability(self, px, py, ptheta, ranges, scan_angles, scan_msg):
        """Improved beam-based sensor model with multiple error components"""
        prob = 1.0
        z_short = 0.1  # Weight for unexpected obstacles
        z_max = 0.05   # Weight for max range measurements
        lambda_short = 0.1  # Exponential decay parameter

        for i, beam_range in enumerate(ranges):
            if beam_range > scan_msg.range_max or beam_range < scan_msg.range_min:
                continue
                
            # Calculate expected range
            expected_range = self.improved_raycasting(px, py, 
                self.normalize_angle(ptheta + scan_angles[i]),
                scan_msg.range_max
            )

            # Calculate individual components
            p_hit = self.z_hit * np.exp(-0.5 * ((beam_range - expected_range)/self.sigma_hit)**2)
            p_short = z_short * (lambda_short * np.exp(-lambda_short * beam_range) 
                    if beam_range < expected_range else 0)
            p_rand = self.z_rand / scan_msg.range_max
            p_max = z_max if beam_range >= scan_msg.range_max else 0
            
            # Total probability
            p_total = p_hit + p_short + p_rand + p_max
            prob *= p_total

        return max(prob, 1e-10)
        
    def improved_raycasting(self, x, y, angle, max_range):
        """Improved raycasting with dynamic step size for accuracy and speed"""
        if self.map_data is None:
            return max_range
            
        # Convert from world coordinates to map coordinates
        mx, my = self.world_to_map(x, y)
        
        # Check if starting point is valid
        if not (0 <= mx < self.map_info.width and 0 <= my < self.map_info.height):
            return max_range
            
        # If starting in an obstacle, return 0
        if self.map_data[int(my), int(mx)] > 50:  # Assuming >50 is obstacle
            return 0.0
            
        # Use distance transform if available
        if self.distance_transform is not None:
            # Get the distance to the nearest obstacle
            dist = self.distance_transform[int(my), int(mx)]
            if dist < 0.05:  # Very close to obstacle
                return 0.0
            
        # Improved ray casting with adaptive step size
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        
        # Start with a small step size
        step_size = 0.01
        distance = 0.0
        
        while distance < max_range:
            # Calculate new point
            nx = x + distance * cos_angle
            ny = y + distance * sin_angle
            
            # Convert to map coordinates
            mx, my = self.world_to_map(nx, ny)
            
            # Check if point is valid
            if not (0 <= mx < self.map_info.width and 0 <= my < self.map_info.height):
                return max_range
                
            # Check if hit an obstacle
            if self.map_data[int(my), int(mx)] > 50:  # Assuming >50 is obstacle
                # Backtrack slightly to get more accurate distance
                for i in range(10):
                    back_dist = distance - (step_size * i / 10.0)
                    back_x = x + back_dist * cos_angle
                    back_y = y + back_dist * sin_angle
                    back_mx, back_my = self.world_to_map(back_x, back_y)
                    
                    if (0 <= back_mx < self.map_info.width and 0 <= back_my < self.map_info.height and 
                        self.map_data[int(back_my), int(back_mx)] <= 50):
                        return back_dist
                
                return distance
            
            # Use distance transform to determine next step size
            if self.distance_transform is not None:
                # Get the distance to the nearest obstacle
                dist = self.distance_transform[int(my), int(mx)]
                # Use smaller of current step size and distance to obstacle
                step_size = min(dist, 0.1)
            
            distance += step_size
                
        return max_range
        
    def world_to_map(self, wx, wy):
        """Convert world coordinates to map coordinates"""
        mx = (wx - self.map_info.origin.position.x) / self.map_resolution
        my = (wy - self.map_info.origin.position.y) / self.map_resolution
        return int(mx), int(my)
        
    def map_to_world(self, mx, my):
        """Convert map coordinates to world coordinates"""
        wx = mx * self.map_resolution + self.map_info.origin.position.x
        wy = my * self.map_resolution + self.map_info.origin.position.y
        return wx, wy
        
    def is_valid_position(self, x, y):
        """Check if a given world position is valid (not in obstacle)"""
        if self.map_data is None:
            return True
            
        mx, my = self.world_to_map(x, y)
        
        # Check if position is within map bounds
        if not (0 <= mx < self.map_info.width and 0 <= my < self.map_info.height):
            return False
            
        # Check if position is not an obstacle (assuming >50 is obstacle)
        # Adding a small buffer to stay away from obstacles
        if self.map_data[int(my), int(mx)] > 50:
            return False
            
        # Use distance transform to check if too close to obstacles
        if self.distance_transform is not None:
            if self.distance_transform[int(my), int(mx)] < 0.1:  # 10cm buffer
                return False
                
        return True
        
    def resample(self):
        """Resample particles based on their weights"""
        # Use low variance sampler for more stable resampling
        if self.use_low_variance_sampler:
            indices = self.low_variance_resampling()
        else:
            indices = self.systematic_resampling()
        
        # Create new particle set
        new_particles = []
        for i in indices:
            # Add small noise proportional to particle weight
            # Less noise for high-weight particles
            noise_scale = max(0.001, 0.01 * (1.0 - self.weights[i]))
            
            x, y, theta = self.particles[i]
            x += np.random.normal(0, noise_scale)
            y += np.random.normal(0, noise_scale)
            theta = self.normalize_angle(theta + np.random.normal(0, noise_scale * 0.5))
            
            # Check if valid position (can help prevent accumulation in walls)
            if self.is_valid_position(x, y):
                new_particles.append((x, y, theta))
            else:
                # Keep original position if new one is invalid
                new_particles.append(self.particles[i])
            
        self.particles = new_particles
        
        # Reset weights
        self.weights = np.ones(self.num_particles) / self.num_particles
        
    def low_variance_resampling(self):
        """Low variance resampling algorithm - better than systematic for PF"""
        indices = np.zeros(self.num_particles, dtype=np.int32)
        r = np.random.uniform(0, 1.0/self.num_particles)
        c = self.weights[0]
        i = 0
        
        for m in range(self.num_particles):
            u = r + m / self.num_particles
            while u > c and i < self.num_particles - 1:
                i += 1
                c += self.weights[i]
            indices[m] = i
            
        return indices
        
    def systematic_resampling(self):
        """Systematic resampling algorithm"""
        positions = (np.arange(self.num_particles) + np.random.uniform()) / self.num_particles
        indices = np.zeros(self.num_particles, dtype=np.int32)
        cumulative_sum = np.cumsum(self.weights)
        i, j = 0, 0
        
        while i < self.num_particles:
            if j >= len(cumulative_sum):  # If we reached the end of weights array
                indices[i:] = len(cumulative_sum) - 1  # Assign remaining indices to the last particle
                break
            if positions[i] < cumulative_sum[j]:
                indices[i] = j
                i += 1
            else:
                j += 1
                
        return indices
        
    def get_estimated_pose(self):
        """Get estimated pose from particles using weighted average"""
        if len(self.particles) == 0:
            return None, None, None
            
        # Find particle with highest weight
        best_particle_idx = np.argmax(self.weights)
        best_x, best_y, best_theta = self.particles[best_particle_idx]
        
        # Calculate weighted average for position
        # Use more sophisticated method for circular coordinates
        sum_x = 0
        sum_y = 0
        sum_cos = 0
        sum_sin = 0
            
        for i, (px, py, ptheta) in enumerate(self.particles):
            w = self.weights[i]
            sum_x += px * w
            sum_y += py * w
            sum_cos += np.cos(ptheta) * w
            sum_sin += np.sin(ptheta) * w
                
        mean_theta = np.arctan2(sum_sin, sum_cos)
        
        # Calculate variance in x and y
        var_x = 0
        var_y = 0
        for i, (px, py, _) in enumerate(self.particles):
            w = self.weights[i]
            var_x += w * (px - sum_x)**2
            var_y += w * (py - sum_y)**2
            
        # If variance is very low, use the best particle
        # This helps prevent jitter in stable scenarios
        if var_x < 0.01 and var_y < 0.01:
            return best_x, best_y, best_theta
        
        # Use weighted average
        return sum_x, sum_y, mean_theta
        
    def publish_particles(self):
        """Publish particle cloud as PoseArray"""
        if len(self.particles) == 0:
            return
            
        msg = PoseArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        
        for x, y, theta in self.particles:
            pose = Pose()
            pose.position.x = x
            pose.position.y = y
            pose.position.z = 0.0
            
            q = quaternion_from_euler(0, 0, theta)
            pose.orientation.x = q[0]
            pose.orientation.y = q[1]
            pose.orientation.z = q[2]
            pose.orientation.w = q[3]
            
            msg.poses.append(pose)
            
        self.particle_pub.publish(msg)
        
    def publish_estimated_pose(self):
        """Publish estimated pose with ROS2 API changes"""
        x, y, theta = self.get_estimated_pose()
        if x is None:
            return
            
        # Calculate covariance based on particle distribution
        cov_xx = 0.0
        cov_yy = 0.0
        cov_tt = 0.0
        
        for i, (px, py, ptheta) in enumerate(self.particles):
            w = self.weights[i]
            cov_xx += w * (px - x)**2
            cov_yy += w * (py - y)**2
            
            # Angle difference needs to be normalized
            angle_diff = self.normalize_angle(ptheta - theta)
            cov_tt += w * angle_diff**2
            
        # Set minimum covariance to avoid numerical issues
        cov_xx = max(0.01, cov_xx)
        cov_yy = max(0.01, cov_yy)
        cov_tt = max(0.01, cov_tt)
        
        cov = np.zeros((36,))
        cov[0] = cov_xx
        cov[7] = cov_yy
        cov[35] = cov_tt
        
        q = quaternion_from_euler(0, 0, theta)
        
        # Publish AMCL pose
        amcl_pose = PoseWithCovarianceStamped()
        amcl_pose.header.stamp = self.get_clock().now().to_msg()
        amcl_pose.header.frame_id = 'map'
        amcl_pose.pose.pose.position.x = x
        amcl_pose.pose.pose.position.y = y
        amcl_pose.pose.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        amcl_pose.pose.covariance = cov.tolist()
        
        self.amcl_pose_pub.publish(amcl_pose)
        
        # Publish odometry message
        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = "map"
        odom_msg.child_frame_id = "base_link"  # Child frame should be robot's base frame
        
        # Populate pose with covariance
        odom_msg.pose.pose.position.x = x
        odom_msg.pose.pose.position.y = y
        odom_msg.pose.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        odom_msg.pose.covariance = cov.tolist()
        
        # Initialize twist (assuming zero velocity if not available)
        odom_msg.twist.twist.linear.x = 0.0
        odom_msg.twist.twist.linear.y = 0.0
        odom_msg.twist.twist.angular.z = 0.0
        # Set twist covariance if available, otherwise use default
        odom_msg.twist.covariance = [0.0] * 36  # Default to zero covariance
        
        self.pose_pub.publish(odom_msg)
        
        # Publish best particle marker
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "particle_filter"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        marker.scale.x = 0.3  # Arrow length
        marker.scale.y = 0.05  # Arrow width
        marker.scale.z = 0.05  # Arrow height
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        
        self.best_particle_marker_pub.publish(marker)
        
        # Broadcast transform
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = 'map'
        transform.child_frame_id = 'particle_filter_pose'
        transform.transform.translation.x = x
        transform.transform.translation.y = y
        transform.transform.translation.z = 0.0
        transform.transform.rotation.x = q[0]
        transform.transform.rotation.y = q[1]
        transform.transform.rotation.z = q[2]
        transform.transform.rotation.w = q[3]
        
        # Broadcast transform
        self.tf_broadcaster.sendTransform(transform)
        
    def get_pose_from_odometry(self, odom_msg):
        """Extract pose from odometry message"""
        x = odom_msg.pose.pose.position.x
        y = odom_msg.pose.pose.position.y
        orientation = odom_msg.pose.pose.orientation
        _, _, theta = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
        return x, y, theta
        
    def gaussian_probability(self, x, mu, sigma):
        """Calculate Gaussian probability density"""
        return np.exp(-0.5 * ((x - mu) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))
        
    def normalize_angle(self, angle):
        """Normalize angle to be between -pi and pi"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

def main(args=None):
    rclpy.init(args=args)
    pf = ParticleFilter()
    rclpy.spin(pf)
    pf.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
