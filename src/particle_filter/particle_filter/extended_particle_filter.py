#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped
from builtin_interfaces.msg import Time
import tf2_ros
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
from message_filters import Subscriber, ApproximateTimeSynchronizer
from sensor_msgs.msg import Image, CameraInfo
# from builtin_interfaces.msg import Time, Duration
from rclpy.duration import Duration

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
        self.num_particles = 20
        self.initial_pose_x = self.get_parameter('initial_pose_x').value
        self.initial_pose_y = self.get_parameter('initial_pose_y').value
        self.initial_pose_a = self.get_parameter('initial_pose_a').value
        self.initial_cov_x = self.get_parameter('initial_cov_x').value
        self.initial_cov_y = self.get_parameter('initial_cov_y').value
        self.initial_cov_a = self.get_parameter('initial_cov_a').value
        

        # for converting ROS depth→numpy
        from cv_bridge import CvBridge
        self.bridge = CvBridge()

        # depth‐sensor model parameters
        self.depth_z_hit   = 0.75
        self.depth_z_rand  = 0.25
        self.depth_sigma   = 0.15    # meters
        self.num_depth_samples = 10  # how many pixels per update
        self.floor_z = 0.0       # Expected floor height (meters)
        self.floor_tolerance = 0.1  # Allow 10cm variation
        # Motion model noise parameters - REDUCED VALUES
        self.alpha1 = 0.1  # Rotation noise component
        self.alpha2 = 0.05  # Translation noise component
        self.alpha3 = 0.05  # Additional translation noise
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
        
        # Camera Variables
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        # State variables
        self.particles = []
        self.weights = []
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
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/intel_realsense_r200_depth/depth/camera_info',
            self._camera_info_callback,
            10
        )
        self.get_logger().info("Waiting for Camera...")
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # names of your camera & base frames (match your URDF)
        self.camera_frame = 'camera_depth_frame'
        self.base_frame   = 'base_link'
        
        # Wait for map to be available
        while rclpy.ok() and self.map_data is None:
            rclpy.spin_once(self, timeout_sec=None)
        self.get_logger().info("Map received!")
        while rclpy.ok() and self.fx is None:
            rclpy.spin_once(self, timeout_sec=None)
        self.get_logger().info(f"Camera Info Recieved {self.fx}, {self.fy}, {self.cx}, {self.cy}")
        # lidar subscriber
        self.scan_sub = Subscriber(self, LaserScan, '/scan')
        # depth camera + intrinsics
        self.depth_sub = Subscriber(self, Image, '/intel_realsense_r200_depth/depth/image_raw')

        # synchronize scan + depth + camera_info
        self.sync = ApproximateTimeSynchronizer(
            [self.scan_sub, self.depth_sub],
            queue_size=10,
            slop=0.1
        )
        self.sync.registerCallback(self.sensor_callback)
        
        # Precompute distance transform for faster sensor model
        self.distance_transform = None
        if self.map_data is not None:
            self.precompute_distance_transform()
        
        self.initialize_particles()
        self.update_timer = self.create_timer(0.1, self.timer_callback)
        

    def _camera_info_callback(self, info_msg):
        K = info_msg.k
        if self.fx is None:
            self.fx, self.fy, self.cx, self.cy = K[0], K[4], K[2], K[5]
        else:
            # unsubscribe after grabbing intrinsics
            self.destroy_subscription(self.camera_info_sub)
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
        """Initialize particles uniformly around the initial pose estimate"""
        self.particles = []
        self.weights = np.ones(self.num_particles) / self.num_particles
        
        # Use tighter distribution around initial pose
        for i in range(self.num_particles):
            x = self.initial_pose_x + np.random.normal(0, self.initial_cov_x * 0.5)
            y = self.initial_pose_y + np.random.normal(0, self.initial_cov_y * 0.5)
            theta = self.initial_pose_a + np.random.normal(0, self.initial_cov_a * 0.5)
            theta = self.normalize_angle(theta)
            
            # Ensure the particle is valid (not in obstacle)
            if self.is_valid_position(x, y):
                self.particles.append((x, y, theta))
            else:
                # Try again with the initial position if invalid
                self.particles.append((self.initial_pose_x, self.initial_pose_y, theta))
            
        self.publish_particles()
        
    def map_callback(self, msg):
        """Process incoming map message"""
        self.map_data = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.map_info = msg.info
        self.map_resolution = msg.info.resolution
        self.get_logger().info(f"Map received: {msg.info.width}x{msg.info.height}, resolution: {self.map_resolution}")
        
        # Precompute distance transform when map changes
        self.precompute_distance_transform()

    def transform_camera_to_base(self, x_c: float, y_c: float, z_c: float):
        """
        Transform a 3D point from the depth‐camera frame into the base_link frame.
        Returns (x_b, y_b, z_b).
        """
        # build a stamped point in the camera frame
        pt_cam = PointStamped()
        pt_cam.header.stamp = self.get_clock().now().to_msg()    # use current time
        pt_cam.header.frame_id = self.camera_frame
        pt_cam.point.x = x_c
        pt_cam.point.y = y_c
        pt_cam.point.z = float(z_c)

        try:
            # lookup & apply transform → get a new PointStamped in base_frame
            pt_base: PointStamped = self.tf_buffer.transform(
                pt_cam,
                self.base_frame,
                timeout=Duration(seconds=1)
            )
            return (pt_base.point.x,
                    pt_base.point.y,
                    pt_base.point.z)

        except (tf2_ros.LookupException,
                tf2_ros.ExtrapolationException,
                tf2_ros.TransformException) as e:
            self.get_logger().warn(f"TF transform failed: {e}")
            # fallback: return camera coords unchanged
            return (x_c, y_c, z_c)
                
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
        
    def sensor_callback(self, scan_msg, depth_msg):
        self.get_logger().info("Sensor called")
        # extract depth image as float32 numpy (meters)
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')

        self.measurement_update(scan_msg, depth_image, self.fx, self.fy, self.cx, self.cy)
        
        # Calculate effective sample size for adaptive resampling
        n_eff = 1.0 / np.sum(np.square(self.weights))
        
        # Resample only when effective sample size is low
        # This helps prevent particle depletion
        if n_eff < self.min_effective_particles:
            self.get_logger().debug(f"Resampling triggered. Effective sample size: {n_eff}")
            self.resample()
            
        # Store scan for future reference
        self.previous_scan = scan_msg
            
        # Publish results
        self.publish_particles()
        self.publish_estimated_pose()
        
    def motion_update(self):
        """Update particle positions based on odometry"""
        if self.last_odom is None or self.current_odom is None:
            self.get_logger().info(f'Odom Null')

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
        new_particles = []
        for i, (px, py, ptheta) in enumerate(self.particles):
            # Add noise proportional to motion magnitude
            # Use smaller noise for small movements
            noise_scale = min(1.0, trans * 5.0 + abs(dtheta) * 5.0)
            
            # Add noise to motion estimates (probabilistic motion model)
            noisy_rot1 = rot1 + np.random.normal(0, self.alpha1 * abs(rot1) + self.alpha2 * trans) * noise_scale
            noisy_trans = trans + np.random.normal(0, self.alpha3 * trans + self.alpha4 * (abs(rot1) + abs(rot2))) * noise_scale
            noisy_rot2 = rot2 + np.random.normal(0, self.alpha1 * abs(rot2) + self.alpha2 * trans) * noise_scale
            
            # Apply motion to particle
            ptheta_new = self.normalize_angle(ptheta + noisy_rot1)
            px_new = px + noisy_trans * np.cos(ptheta_new)
            py_new = py + noisy_trans * np.sin(ptheta_new)
            ptheta_new = self.normalize_angle(ptheta_new + noisy_rot2)
            
            # Check if the new position is valid (not in an obstacle)
            if self.is_valid_position(px_new, py_new):
                new_particles.append((px_new, py_new, ptheta_new))
            else:
                # Keep the old particle with slightly modified orientation
                # This helps particles "escape" from invalid positions
                # new_particles.append((px, py, ptheta + np.random.normal(0, 0.05)))
                best_idx = np.argmax(self.weights)
                best_x, best_y, best_theta = self.particles[best_idx]
                px_new = best_x + np.random.normal(0, 0.1)
                py_new = best_y + np.random.normal(0, 0.1)
                ptheta_new = best_theta + np.random.normal(0, 0.05)
                new_particles.append((px_new, py_new, ptheta_new))
                
        self.particles = new_particles
        
    def measurement_update(self, scan_msg, depth_image, fx, fy, cx, cy):
        """Update particle weights based on laser scan measurements"""
        if len(self.particles) == 0 or self.map_data is None:
            self.get_logger().info(f'Measurement Null')

            return
            
        angles = np.arange(scan_msg.angle_min, scan_msg.angle_max + scan_msg.angle_increment, scan_msg.angle_increment)
        
        # Use more beams for better accuracy, but still subsample for efficiency
        step = 10  # Use every 10th beam instead of 20th
        ranges = np.array(scan_msg.ranges[::step])
        scan_angles = angles[::step]
        # floor_z = -0.1  # Expected floor height in base frame
        # floor_threshold = 0.05
        
        # # Modify depth processing to include floor likelihood
        # for u,v in zip(...):
        #     z = depth_image[v,u]
        #     if z > 4.0:  # Max range returns
        #         # Penalize particles not aligned with expected floor geometry
        #         x_b, y_b, z_b = self.transform_camera_to_base(x_c, y_c, z)
        #         floor_prob = np.exp(-abs(z_b - floor_z)/floor_threshold)
        #         p += self.depth_z_floor * floor_prob
        lidar_logs = []
        for px,py,pt in self.particles:
            log_w = 0.0
            for r,ang in zip(ranges, scan_angles):
                # … same per‐beam likelihood‐field …
                p = self.get_scan_probability(px, py, pt, ranges, scan_angles, scan_msg)
                log_w += math.log(max(p, 1e-10))
        lidar_logs.append(log_w) 
        #–– now add depth log‐likelihood per particle ––
        depth_logs = []
        H, W = depth_image.shape
        for px,py,pt in self.particles:
            log_w = 0.0
            # randomly sample some pixels
            for u,v in zip(
                np.random.randint(0,W,self.num_depth_samples),
                np.random.randint(0,H,self.num_depth_samples)
            ):
                z = depth_image[v,u]
                if not np.isfinite(z) or z<=0.1 or z>5.0:
                    continue
                # back‐project into camera frame
                x_c = (u - cx)*z/fx
                y_c = (v - cy)*z/fy
                # transform to robot base, then world
                # assume depth camera frame → base_link known as T_cb
                x_b, y_b, z_b = self.transform_camera_to_base(x_c, y_c, z)

                # Ground plane filtering
                if abs(z_b - self.floor_z) > self.floor_tolerance:
                    continue

                # Particle-relative transformation
                wx = px + x_b * math.cos(pt) - y_b * math.sin(pt)
                wy = py + x_b * math.sin(pt) + y_b * math.cos(pt)

                # Map check
                mx, my = self.world_to_map(wx, wy)
                # mx,my = self.world_to_map(bx,by)
                d = (0 <= mx < self.map_info.width and
                     0 <= my < self.map_info.height) \
                    and self.distance_transform[my,mx] \
                    or scan_msg.range_max
                # depth‐likelihood‐field
                p_hit  = (1.0/(self.depth_sigma*math.sqrt(2*math.pi))) \
                         * math.exp(-0.5*(d/self.depth_sigma)**2)
                p_rand = 1.0/scan_msg.range_max
                p = self.depth_z_hit*p_hit + self.depth_z_rand*p_rand
                log_w += math.log(max(p,1e-10))
            depth_logs.append(log_w)

        #–– combine logs and normalize ––
        combined = np.array(lidar_logs) + np.array(depth_logs)
        combined -= combined.max()
        w = np.exp(combined)
        self.weights = w / w.sum()
        self.get_logger().info(f'Measurement Called')

            
    def get_scan_probability(self, px, py, ptheta, ranges, scan_angles, scan_msg):
        """Calculate probability of a scan given particle position using improved model"""
        # Start with a very small probability to avoid zero
        probability = 0.0
        
        # Count how many beams we actually use
        valid_beams = 0
        total_error = 0.0
        
        # Use a more robust algorithm for handling beam errors
        errors = []
        
        # Check a subset of the beams for efficiency
        for i, beam_range in enumerate(ranges):
            # Skip invalid measurements
            if np.isnan(beam_range) or beam_range > scan_msg.range_max or beam_range < scan_msg.range_min:
                continue
                
            # Calculate global angle of this beam
            beam_angle = self.normalize_angle(ptheta + scan_angles[i])
            valid_beams += 1
            
            # Calculate expected range by ray casting in the map
            # expected_range = self.improved_raycasting(px, py, beam_angle, scan_msg.range_max)
            expected_range = self.improved_raycasting(px, py, beam_angle, scan_msg.range_max)
            # mx, my = self.world_to_map(px, py)
            # if (0 <= mx < self.map_info.width and 0 <= my < self.map_info.height):
            #     expected_range = self.distance_transform[my, mx]
            # else:
            #     expected_range = scan_msg.range_max
            
            # Calculate error between expected and actual range
            error = abs(beam_range - expected_range)
            errors.append(error)
            total_error += error
            
        # If we didn't use any valid beams, return a small probability
        if valid_beams == 0:
            return 1e-10
            
        # Calculate mean error
        mean_error = total_error / valid_beams
        
        # Calculate probability using an exponential model
        # This penalizes large errors more severely
        probability = np.exp(-mean_error / self.sigma_hit)
        
        # Clip to prevent extremely small probabilities
        probability = max(probability, 1e-10)
        
        return probability
        
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