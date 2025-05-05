#!/usr/bin/env python3

import rospy
import actionlib
import numpy as np
import tf
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry, Path
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatus
from std_msgs.msg import Bool, Empty, String
from tf.transformations import euler_from_quaternion

class RecoveryBehaviors:
    def __init__(self):
        rospy.init_node('turtlebot3_recovery_behaviors')
        
        # Parameters
        self.robot_stuck_threshold = rospy.get_param('~robot_stuck_threshold', 0.05)  # m/s
        self.stuck_timeout = rospy.get_param('~stuck_timeout', 5.0)  # seconds
        self.recovery_rotation_speed = rospy.get_param('~recovery_rotation_speed', 0.5)  # rad/s
        self.clear_costmap_on_recovery = rospy.get_param('~clear_costmap_on_recovery', True)
        self.max_recovery_attempts = rospy.get_param('~max_recovery_attempts', 5)
        self.path_progress_timeout = rospy.get_param('~path_progress_timeout', 10.0)  # seconds
        self.min_path_progress = rospy.get_param('~min_path_progress', 0.2)  # meters
        
        # State variables
        self.last_position = None
        self.last_velocity = None
        self.last_time = None
        self.is_stuck = False
        self.stuck_start_time = None
        self.recovery_attempts = 0
        self.current_behavior = None
        self.last_path_position = None
        self.last_path_time = None
        self.recovery_in_progress = False
        self.last_goal = None
        
        # Publishers
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.recovery_status_pub = rospy.Publisher('/recovery_status', String, queue_size=10)
        
        # Subscribers
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.path_sub = rospy.Subscriber('/move_base/NavfnROS/plan', Path, self.path_callback)
        self.goal_sub = rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback)
        
        # Clear costmap service
        if self.clear_costmap_on_recovery:
            self.clear_costmap_pub = rospy.Publisher('/move_base/clear_costmaps', Empty, queue_size=10)
        
        # Create action client for move_base if available
        self.move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.move_base_available = False
        try:
            if self.move_base_client.wait_for_server(rospy.Duration(2.0)):
                self.move_base_available = True
                rospy.loginfo("Connected to move_base action server")
            else:
                rospy.logwarn("Could not connect to move_base action server, will operate in standalone mode")
        except:
            rospy.logwarn("Could not connect to move_base action server, will operate in standalone mode")
        
        # TF listener
        self.tf_listener = tf.TransformListener()
        
        # Recovery behaviors list (ordered by increasing invasiveness)
        self.recovery_behaviors = [
            self.rotate_recovery,
            self.oscillate_recovery,
            self.back_up_recovery,
            self.spin_recovery,
            self.clear_and_rotate_recovery
        ]
        
        # Recovery behavior check timer
        self.check_timer = rospy.Timer(rospy.Duration(0.5), self.check_progress)
        
        rospy.loginfo("TurtleBot3 Recovery Behaviors node initialized")
    
    def goal_callback(self, msg):
        """Track the latest navigation goal"""
        self.last_goal = msg
        self.recovery_attempts = 0  # Reset recovery attempts for new goal
    
    def odom_callback(self, msg):
        """Process odometry data to detect when robot is stuck"""
        # Extract position and velocity
        position = msg.pose.pose.position
        linear_velocity = msg.twist.twist.linear.x
        angular_velocity = msg.twist.twist.angular.z
        
        current_time = rospy.Time.now()
        
        # Initialize state if this is the first odometry message
        if self.last_position is None:
            self.last_position = position
            self.last_velocity = (linear_velocity, angular_velocity)
            self.last_time = current_time
            return
        
        # Update state
        self.last_position = position
        self.last_velocity = (linear_velocity, angular_velocity)
        self.last_time = current_time
    
    def path_callback(self, msg):
        """Track progress along the current path"""
        if not msg.poses:
            return
        
        # Get the first point in the path (closest to robot)
        path_position = msg.poses[0].pose.position
        current_time = rospy.Time.now()
        
        if self.last_path_position is None:
            self.last_path_position = path_position
            self.last_path_time = current_time
            return
        
        # Update state
        self.last_path_position = path_position
        self.last_path_time = current_time
    
    def check_progress(self, event=None):
        """Check if the robot is making progress and execute recovery if not"""
        if self.recovery_in_progress or self.last_position is None or self.last_velocity is None:
            return
        
        # Check if robot is trying to move but has low velocity
        linear_vel, angular_vel = self.last_velocity
        is_moving_command = abs(linear_vel) > 0.01 or abs(angular_vel) > 0.01
        is_actually_moving = abs(linear_vel) > self.robot_stuck_threshold
        
        # If robot is commanded to move but isn't actually moving, it might be stuck
        if is_moving_command and not is_actually_moving:
            if not self.is_stuck:
                self.is_stuck = True
                self.stuck_start_time = rospy.Time.now()
                rospy.loginfo("Robot may be stuck, monitoring...")
            elif (rospy.Time.now() - self.stuck_start_time).to_sec() > self.stuck_timeout:
                rospy.logwarn(f"Robot is stuck for {self.stuck_timeout} seconds, initiating recovery")
                self.execute_recovery()
        else:
            self.is_stuck = False
            
        # Check if we're making progress on the path
        if self.last_path_position is not None and self.last_path_time is not None:
            time_since_path_progress = (rospy.Time.now() - self.last_path_time).to_sec()
            if time_since_path_progress > self.path_progress_timeout:
                rospy.logwarn(f"No path progress for {time_since_path_progress:.1f} seconds, initiating recovery")
                self.execute_recovery()
    
    def execute_recovery(self):
        """Execute the appropriate recovery behavior based on attempt count"""
        if self.recovery_in_progress:
            return
            
        self.recovery_in_progress = True
        
        if self.recovery_attempts >= self.max_recovery_attempts:
            rospy.logerr(f"Maximum recovery attempts ({self.max_recovery_attempts}) reached, giving up")
            self.publish_status("MAX_ATTEMPTS_REACHED")
            
            # If using move_base, abort the current goal
            if self.move_base_available and self.last_goal is not None:
                self.move_base_client.cancel_goal()
                rospy.logwarn("Cancelled current navigation goal due to recovery failure")
                
            self.recovery_in_progress = False
            return
        
        # Select recovery behavior based on attempt count
        behavior_index = min(self.recovery_attempts, len(self.recovery_behaviors) - 1)
        selected_behavior = self.recovery_behaviors[behavior_index]
        
        # Execute the selected behavior
        behavior_name = selected_behavior.__name__
        rospy.loginfo(f"Executing recovery behavior: {behavior_name} (attempt {self.recovery_attempts + 1})")
        self.publish_status(f"EXECUTING_{behavior_name.upper()}")
        
        try:
            selected_behavior()
        except Exception as e:
            rospy.logerr(f"Recovery behavior failed: {e}")
        
        # Increment attempt counter
        self.recovery_attempts += 1
        self.recovery_in_progress = False
        self.publish_status("RECOVERY_COMPLETE")
    
    def rotate_recovery(self):
        """Simple recovery behavior: rotate in place to find a clear path"""
        self.publish_status("ROTATING")
        
        # Stop the robot first
        self.stop_robot()
        rospy.sleep(0.5)
        
        # Rotate 180 degrees
        twist = Twist()
        twist.angular.z = self.recovery_rotation_speed
        
        # Calculate time needed to rotate 180 degrees
        rotation_time = np.pi / self.recovery_rotation_speed
        
        start_time = rospy.Time.now()
        rate = rospy.Rate(10)
        
        while (rospy.Time.now() - start_time).to_sec() < rotation_time and not rospy.is_shutdown():
            self.cmd_vel_pub.publish(twist)
            rate.sleep()
        
        # Stop after rotation
        self.stop_robot()
    
    def oscillate_recovery(self):
        """Oscillate left and right to escape narrow situations"""
        self.publish_status("OSCILLATING")
        
        # Stop the robot first
        self.stop_robot()
        rospy.sleep(0.5)
        
        cycles = 3
        cycle_time = 2.0  # seconds per cycle (1s left, 1s right)
        
        twist = Twist()
        rate = rospy.Rate(20)
        start_time = rospy.Time.now()
        
        while (rospy.Time.now() - start_time).to_sec() < cycles * cycle_time and not rospy.is_shutdown():
            elapsed = (rospy.Time.now() - start_time).to_sec()
            cycle_phase = (elapsed % cycle_time) / cycle_time
            
            if cycle_phase < 0.5:
                # Turn left first half of cycle
                twist.angular.z = self.recovery_rotation_speed
            else:
                # Turn right second half of cycle
                twist.angular.z = -self.recovery_rotation_speed
                
            self.cmd_vel_pub.publish(twist)
            rate.sleep()
        
        # Stop after oscillation
        self.stop_robot()
    
    def back_up_recovery(self):
        """Back up to escape collision"""
        self.publish_status("BACKING_UP")
        
        # Stop the robot first
        self.stop_robot()
        rospy.sleep(0.5)
        
        # Back up for a short time
        twist = Twist()
        twist.linear.x = -0.1  # Slow backward motion
        
        duration = 3.0  # seconds to back up
        
        start_time = rospy.Time.now()
        rate = rospy.Rate(10)
        
        while (rospy.Time.now() - start_time).to_sec() < duration and not rospy.is_shutdown():
            self.cmd_vel_pub.publish(twist)
            rate.sleep()
        
        # Stop after backing up
        self.stop_robot()
    
    def spin_recovery(self):
        """Perform a full 360-degree spin to find a clear path"""
        self.publish_status("SPINNING")
        
        # Clear costmaps first if enabled
        if self.clear_costmap_on_recovery:
            self.clear_costmaps()
            rospy.sleep(0.5)
        
        # Stop the robot first
        self.stop_robot()
        rospy.sleep(0.5)
        
        # Rotate 360 degrees
        twist = Twist()
        twist.angular.z = self.recovery_rotation_speed
        
        # Calculate time needed to rotate 360 degrees
        rotation_time = 2 * np.pi / self.recovery_rotation_speed
        
        start_time = rospy.Time.now()
        rate = rospy.Rate(10)
        
        while (rospy.Time.now() - start_time).to_sec() < rotation_time and not rospy.is_shutdown():
            self.cmd_vel_pub.publish(twist)
            rate.sleep()
        
        # Stop after rotation
        self.stop_robot()
    
    def clear_and_rotate_recovery(self):
        """Clear costmaps and rotate"""
        self.publish_status("CLEAR_AND_ROTATE")
        
        # Clear costmaps first
        self.clear_costmaps()
        rospy.sleep(1.0)
        
        # Then do a 360 spin
        self.spin_recovery()
    
    def clear_costmaps(self):
        """Clear costmaps using ROS service"""
        if not self.clear_costmap_on_recovery:
            return
            
        rospy.loginfo("Clearing costmaps")
        try:
            self.clear_costmap_pub.publish(Empty())
        except Exception as e:
            rospy.logerr(f"Failed to clear costmaps: {e}")
    
    def stop_robot(self):
        """Stop the robot by publishing zero velocity"""
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        
        # Publish multiple times to ensure the command gets through
        for _ in range(3):
            self.cmd_vel_pub.publish(twist)
            rospy.sleep(0.1)
    
    def publish_status(self, status):
        """Publish current recovery status"""
        msg = String()
        msg.data = status
        self.recovery_status_pub.publish(msg)

if __name__ == "__main__":
    try:
        recovery = RecoveryBehaviors()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass