#!/usr/bin/env python3
from tf2_ros import Buffer, TransformListener # for locolization
import rclpy
import traceback
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_cdt, distance_transform_edt, binary_dilation
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy
from geometry_msgs.msg import TransformStamped  # Use TransformStamped instead of Rigids
from collections import deque  # For implementing a circular buffer
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker
from nav_msgs.msg import OccupancyGrid # for applying local costmap
import std_msgs
import time
import math
import copy

DEFAULT_OBS_COST = 1e4
ACTION_WEIGHT = 10.0

class Config:
  """ Configurations that are typically fixed throughout execution. """
  def __init__(self, 
               T=2, # Horizon (s)
               dt=0.1, # Length of each step (s)
               num_control_rollouts=16384, # Number of control sequences
               num_vis_state_rollouts=16384, # Number of visualization rollouts
               seed=1,
               mppi_type=0): # Normal dist / 1: NLN):
    self.seed = seed
    self.T = T
    self.dt = dt
    self.num_steps = int(T/dt)
    self.mppi_type = mppi_type # Normal dist / 1: NLN
    assert T > 0
    assert dt > 0
    assert T > dt
    assert self.num_steps > 0

    # Number of control rollouts are currently limited by the number of blocks
    self.num_control_rollouts = num_control_rollouts

    
    # For visualizing state rollouts
    self.num_vis_state_rollouts = num_vis_state_rollouts
    self.num_vis_state_rollouts = min([self.num_vis_state_rollouts, self.num_control_rollouts])
    self.num_vis_state_rollouts = max([1, self.num_vis_state_rollouts])

''' @brief: Returning the mean and standard deviation of the lognormal distribution, 
           given mean and variance of Normal distribution'''
def Normal2LogN(m, v):
    """ m: mean, v: variance
    Return: mu: mean, sigma: standard deviation of LN dist"""
    mu = np.exp(m + 0.5 * v)
    var = np.exp(2 * m + v) * (np.exp(v) - 1)
    sigma = np.sqrt(var)
    return mu, sigma

class MPPI_CPU:
    def __init__(self, cfg):
        self.cfg = cfg
        self.T = cfg.T
        self.dt = cfg.dt
        self.num_steps = cfg.num_steps
        self.num_control_rollouts = cfg.num_control_rollouts
        self.num_vis_state_rollouts = cfg.num_vis_state_rollouts
        self.vehicle_length = 0.138
        self.vehicle_width = 0.138
        self.u_seq0 = np.zeros((self.num_steps, 2), dtype=np.float32)
        self.u_seq0[:, 0] = 0.1
        self.params = None
        self.costmap_loaded = False
        self.rng = np.random.default_rng(cfg.seed)
        self.mppi_type = cfg.mppi_type
        self.local_costmap_size = 120
        self.mu_LogN, self.std_LogN = Normal2LogN(0, np.mean([0.05, 0.05]))
        self.reset()
        # self.u_seq0[:, 0] = 0.1  # Linear velocity
        self.u_seq0[:, 1] = 0.0  # Angular velocity (no initial turn)

    def reset(self):
        self.u_cur = self.u_seq0.copy()
        self.params = None
        self.costmap_loaded = False

    def setup(self, params):
        self.params = copy.deepcopy(params)
        if self.params['costmap'] is not None:
            self.costmap_loaded = True

    def check_solve_conditions(self):
        return self.params is not None and self.costmap_loaded

    def solve(self):
        if not self.check_solve_conditions():
            return None
        return self.solve_with_nominal_dynamics()

    def random_noise_sample(self):
        if self.mppi_type == 0:
            return self.rng.normal(0, 1, (self.num_control_rollouts, self.num_steps, 2))
        else:
            log_noise = self.rng.lognormal(self.mu_LogN, self.std_LogN, 
                                        (self.num_control_rollouts, self.num_steps, 2))
            return log_noise * self.rng.normal(0, 1, (self.num_control_rollouts, self.num_steps, 2))

    def solve_with_nominal_dynamics(self):
        params = self.params
        u_std = params['u_std']
        noise_samples = self.random_noise_sample()
        noise_samples *= u_std.reshape(1, 1, 2)
        
        costs = np.zeros(self.num_control_rollouts)
        for k in range(params['num_opt']):
            costs = self.rollout(params, noise_samples)
            self.update_control_sequence(costs, noise_samples, params)
        return self.u_cur

    def rollout(self, params, noise_samples):
        # Pre-allocate arrays for state trajectories
        x0 = params['x0']
        x_all = np.tile(x0, (self.num_control_rollouts, 1))
        costs = np.zeros(self.num_control_rollouts)
        
        # Get parameters once
        dt = params['dt']
        goal = params['xgoal']
        goal_tol = params['goal_tolerance']
        costmap = params['costmap']
        resolution = params['costmap_resolution']
        origin = params['costmap_origin']
        
        # Process each time step
        for t in range(self.num_steps):
            # Apply noise to controls and clip
            v_nom = self.u_cur[t, 0] + noise_samples[:, t, 0]
            v = np.clip(v_nom, params['vrange'][0], params['vrange'][1])
            w_nom = self.u_cur[t, 1] + noise_samples[:, t, 1]
            w = np.clip(w_nom, params['wrange'][0], params['wrange'][1])
            
            # Update all states in parallel
            cos_theta = np.cos(x_all[:, 2])
            sin_theta = np.sin(x_all[:, 2])
            
            x_all[:, 0] += dt * v * cos_theta
            x_all[:, 1] += dt * v * sin_theta
            x_all[:, 2] += dt * w
            x_all[:, 2] = x_all[:, 2] % (2 * np.pi)
            
            # Calculate indices for costmap lookup
            ix = np.clip(((x_all[:, 0] - origin[0]) / resolution).astype(int), 
                        0, costmap.shape[1]-1)
            iy = np.clip(((x_all[:, 1] - origin[1]) / resolution).astype(int),
                        0, costmap.shape[0]-1)
            
            # Apply obstacle costs
            valid_indices = (0 <= ix) & (ix < costmap.shape[1]) & (0 <= iy) & (iy < costmap.shape[0])
            costs[valid_indices] += costmap[iy[valid_indices], ix[valid_indices]] * params['obs_penalty']
            
            # Apply distance costs
            dist_to_goal = np.hypot(x_all[:, 0]-goal[0], x_all[:, 1]-goal[1])
            costs += dist_to_goal * 1e5
            # if t > 0:
            #     dw = np.abs(w - noise_samples[:, t-1, 1])  # Penalize angular acceleration
            #     costs += 1.0 * dw  # Adjust weight as needed
            # costs += 10.0 * np.abs(w)
            dx = goal[0] - x_all[:, 0]
            dy = goal[1] - x_all[:, 1]
            desired_theta = np.arctan2(dy, dx)
            theta_error = np.abs(x_all[:, 2] - desired_theta) % (2 * np.pi)
            theta_error = np.minimum(theta_error, 2 * np.pi - theta_error)
            costs += 5.0 * theta_error 

            
        # Terminal costs
        goal_reached = dist_to_goal < goal_tol
        costs[~goal_reached] += dist_to_goal[~goal_reached] * 50.0
        
        return costs
    def update_control_sequence(self, costs, noise_samples, params):
        smoothing_factor = 1
        lambda_weight = params['lambda_weight']
        beta = np.min(costs)
        weights = np.exp(-(costs - beta) / lambda_weight)
        weights /= np.sum(weights)
        
        # Update control sequence
        u_new = np.sum(weights[:, None, None] * noise_samples, axis=0)
        self.u_cur = (1 - smoothing_factor) * self.u_cur + smoothing_factor * u_new
        self.u_cur[:, 0] = np.clip(self.u_cur[:, 0],
                          params['vrange'][0],
                          params['vrange'][1])
        self.u_cur[:, 1] = np.clip(self.u_cur[:, 1], 
                                  params['wrange'][0], 
                                  params['wrange'][1])

    def shift_and_update(self, new_x0, u_cur, num_shifts=1):
        self.u_cur[:-num_shifts] = u_cur[num_shifts:]
        self.u_cur[-num_shifts:] = u_cur[-1]

class MPPIPlannerNode(Node):
    def __init__(self):
        super().__init__('mppi_planner_node')

        # Initialize configuration for MPPI
        self.cfg = Config(
            T = 15,
            dt = 0.5,
            num_control_rollouts = 512,  # Reduce from 1000 to 256
            num_vis_state_rollouts = 128,  # Reduce from 500 to 128
            seed = 1,
            mppi_type = 1
        )
        self.mppi = MPPI_CPU(self.cfg)

        # MPPI initial parameters
        self.mppi_params = dict(
          # Task specification
          dt = self.cfg.dt, 
          x0 = np.zeros(3), # Start state
          xgoal = np.array([-2.2, 9.5]), # Goal position
          # vehicle length(lf and lr wrt the cog) and width
          vehicle_length = 0.2,
          vehicle_width = 0.2,
          vehicle_wheelbase= 0.32,
          # For risk-aware min time planning
          goal_tolerance = 0.10,
          dist_weight = 10, #  Weight for dist-to-goal cost.

          lambda_weight = 1.0, # Temperature param in MPPI
          num_opt = 2, # Number of steps in each solve() function call.

          # Control and sample specification
        #   variance = 0.1
          u_std = np.array([0.05, 0.1]),
          vrange = np.array([0.0, 0.22]), # Linear velocity range. Constant Linear Velocity
          wrange = np.array([-2.84, 2.84]), # Angular velocity range.
        #   wrange = np.array([-1.5, 1.5]),
          costmap = None, # intiallly nothing
          obs_penalty = 1e4
        )
        
        self.rear_x = self.mppi_params['x0'][0] - ((self.mppi_params['vehicle_wheelbase'] / 2) * math.cos(self.mppi_params['x0'][2]))
        self.rear_y = self.mppi_params['x0'][1] - ((self.mppi_params['vehicle_wheelbase'] / 2) * math.sin(self.mppi_params['x0'][2]))

        self.min_lookahead = 1.0
        self.max_lookahead = 3.0
        self.lookahead_ratio = 1.5
        self.current_index = None
        self.target_index = 0

        self.mppi_path_pub = self.create_publisher(Path, "/mppi_path", 10)

        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        '''############### for costmap ##############'''
        self.local_costmap = None  # store the latest costmap
        # self.costmap_sub = self.create_subscription(
        #     OccupancyGrid, # Type: nav_msgs/msg/OccupancyGrid
        #     '/local_costmap/costmap',
        #     self.costmap_callback,
        #     1 # only the most recent message is kept in the queue
        # )
        self.costmap_sub = self.create_subscription(
            OccupancyGrid, # Type: nav_msgs/msg/OccupancyGrid
            '/costmap/costmap',
            self.costmap_callback,
            1 # only the most recent message is kept in the queue
        )
        # self.debug_local_costmap_pub = self.create_publisher(OccupancyGrid, '/debug_local_costmap', 1)
        cmd_vel_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,  
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.cmd_vel_pub = self.create_publisher(
            msg_type=Twist,
            topic="/cmd_vel",
            qos_profile=cmd_vel_qos,
        )
        # Create a timer to call the MPPI solver every 100ms (0.1s)
        self.timer = self.create_timer(0.1, self.solve_mppi)
        self.i = 0
        self.isGoalReached = False
        self.mppi.setup(self.mppi_params)
        self.get_logger().info('MPPI Planner Node started')


    def costmap_callback(self, msg: OccupancyGrid):
        # Convert msg.data to a 2D list or np.array
        width = msg.info.width
        height = msg.info.height
        # Convert msg data to float costmap and store resolution/origin
        costmap_int8 = np.array(msg.data, dtype=np.int8).reshape(height, width)
        costmap_int8[costmap_int8 == -1] = 0 # make unknown area as obstacles 
        self.local_costmap = costmap_int8.astype(np.float32)
        self.mppi_params['costmap_resolution'] = msg.info.resolution
        self.mppi_params['costmap_origin'] = [msg.info.origin.position.x, msg.info.origin.position.y]


    def calc_distance(self, point_x, point_y):
        dx = self.rear_x - point_x
        dy = self.rear_y - point_y
        return math.hypot(dx, dy)

    def _unicycle_dynamics(self, state, action, dt):
        x, y, theta = state
        v, w = action  # Now using linear/angular velocity directly
        
        x_new = x + v * np.cos(theta) * dt
        y_new = y + v * np.sin(theta) * dt
        theta_new = theta + w * dt
        
        return (x_new, y_new, theta_new)

    def _state_to_pose(self, state):
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(state[0])
        pose.pose.position.y = float(state[1])
        pose.pose.position.z = 0.0
        q = R.from_euler('z', float(state[2])).as_quat()
        pose.pose.orientation.x = q[0]
        pose.pose.orientation.y = q[1]
        pose.pose.orientation.z = q[2]
        pose.pose.orientation.w = q[3]
        return pose

    def solve_mppi(self):
        try:
            # 1. Look up transform from map -> base_link

            transform = self.tf_buffer.lookup_transform(
                'map',           # source frame (or "map")
                'base_link',     # target frame (your robot)
                rclpy.time.Time()
            )
            # 2. Extract x, y
            x_robot = transform.transform.translation.x
            y_robot = transform.transform.translation.y

            # 3. Convert quaternion to yaw
            quat = transform.transform.rotation
            r = R.from_quat([quat.x, quat.y, quat.z, quat.w])
            yaw_robot = r.as_euler('xyz', degrees=False)[2]            

            # 4. Update the MPPI initial state
            self.mppi_params['x0'] = np.array([x_robot, y_robot, yaw_robot])
            self.rear_x = self.mppi_params['x0'][0] - ((self.mppi_params['vehicle_wheelbase'] / 2) * math.cos(self.mppi_params['x0'][2]))
            self.rear_y = self.mppi_params['x0'][1] - ((self.mppi_params['vehicle_wheelbase'] / 2) * math.sin(self.mppi_params['x0'][2]))

            # If we have a valid costmap, pass it to MPPI
            if self.local_costmap is not None:
                self.mppi_params['costmap'] = self.local_costmap

            self.mppi.setup(self.mppi_params)
            self.mppi.local_costmap_origin = self.mppi_params['costmap_origin']
            # start_time = time.time()
            # Solve MPPI
            result = self.mppi.solve()
            mppi_path_msg = Path()
            mppi_path_msg.header.frame_id = "map"
            mppi_path_msg.header.stamp = self.get_clock().now().to_msg()
            propagated_state = self.mppi_params['x0'].copy()
            mppi_path_msg.poses.append(self._state_to_pose(propagated_state))
            for action in result:
                clipped_action = [
                np.clip(action[0], self.mppi_params['vrange'][0], self.mppi_params['vrange'][1]),
                np.clip(action[1], self.mppi_params['wrange'][0], self.mppi_params['wrange'][1])
                ]
                propagated_state = self._unicycle_dynamics(propagated_state, clipped_action, self.cfg.dt)
                mppi_path_msg.poses.append(self._state_to_pose(propagated_state))
            self.mppi_path_pub.publish(mppi_path_msg)

            # self.get_logger().info(f"Elapsed time for solving mppi: {time.time() - start_time}")
            # self.publish_local_costmap_debug()

            #get the first action 
            u_execute = result[0]
            if self.i > 3:
              self.get_logger().info(
                  f"[DEBUG] u_execute → linear: {u_execute[0]:.3f}, angular: {u_execute[1]:.3f}"
              )
              h = std_msgs.msg.Header()
              h.stamp = self.get_clock().now().to_msg()
              cmd = Twist()
              if ((self.i % 10) == 0): 
                self.get_logger().info(f"Controls - Linear: {u_execute[0]:.2f} m/s, Angular: {u_execute[1]:.2f} rad/s")
              if self.isGoalReached:
                u_execute = [0.0, 0.0]
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
                self.get_logger().info(f"Goal Reached!!!!")
              else: 
                # drive = AckermannDrive(steering_angle=1.0*np.arctan2((self.mppi_params['vehicle_wheelbase'])*u_execute[1], u_execute[0]), speed=1.0)
                # drive = AckermannDrive(steering_angle=-0.8*(np.tan(u_execute[1]*1.0)*(self.mppi_params['vehicle_wheelbase'])), speed=1.0)
                cmd.linear.x = float(np.clip(u_execute[0], 
                                             self.mppi_params['vrange'][0], 
                                             self.mppi_params['vrange'][1]))
                cmd.angular.z = float(np.clip(u_execute[1],
                                              self.mppi_params['wrange'][0],
                                              self.mppi_params['wrange'][1]))
              self.cmd_vel_pub.publish(cmd)
              self.mppi.shift_and_update(self.mppi_params['x0'], result, 1)

              dist2goal2 = (self.mppi_params['xgoal'][0] - x_robot)**2 \
                         + (self.mppi_params['xgoal'][1] - y_robot)**2
              
              goaltol2 = self.mppi_params['goal_tolerance'] * self.mppi_params['goal_tolerance']
              if ((self.i % 10) == 0): 
                self.get_logger().info(f"Distance to the Goal: {np.sqrt(dist2goal2)}, Goal Tolerance: {goaltol2}, Current target_index: {self.target_index}")
              if dist2goal2 < goaltol2:
                self.isGoalReached = True
              if dist2goal2 > goaltol2:
                self.isGoalReached = False
            self.i += 1

        except Exception as e:
            tb_str = ''.join(traceback.format_exception(None, e, e.__traceback__))
            self.get_logger().warn(f"Could not lookup TF transform: {e}\n{tb_str}")
            return

    def on_shutdown(self):
        self.get_logger().info('MPPI Planner Node shutting down')

def main(args=None):
    rclpy.init(args=args)
    node = MPPIPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.on_shutdown()
        rclpy.shutdown()
if __name__ == '__main__':
    main()