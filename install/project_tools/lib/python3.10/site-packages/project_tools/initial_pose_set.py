import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
from tf_transformations import quaternion_from_euler
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

class InitialPosePublisher(Node):
    def __init__(self):
        super().__init__('initial_pose_publisher')
        # Declare parameters
        self.declare_parameter('x', -2.5)
        self.declare_parameter('y', 1.0)
        self.declare_parameter('yaw', 1.5)  # In radians
        self.declare_parameter('frame_id', 'map')
        # self.declare_parameter('use_sim_time', True)
        
        # Configure QoS to match AMCL
        qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE
        )
        
        self.publisher = self.create_publisher(
            PoseWithCovarianceStamped,
            'initialpose',  # Will respect namespace
            qos
        )
        
        # Publish after 5 seconds (ensure nodes are ready)
        self.timer = self.create_timer(1.0, self.publish_initial_pose)
        
    def publish_initial_pose(self):
        # Get parameters
        x = self.get_parameter('x').value
        y = self.get_parameter('y').value
        yaw = self.get_parameter('yaw').value
        frame_id = self.get_parameter('frame_id').value
        
        # Convert yaw to quaternion
        q = quaternion_from_euler(0, 0, yaw)
        
        # Create message with current time
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.pose.pose.position.x = float(x)
        msg.pose.pose.position.y = float(y)
        msg.pose.pose.orientation.z = q[2]
        msg.pose.pose.orientation.w = q[3]
        msg.pose.covariance = [0.25] * 36  # Default covariance
        
        self.publisher.publish(msg)
        self.get_logger().info(f'Published initial pose at ({x}, {y}, θ={yaw})')
        self.timer.cancel()  # Stop after one publication

def main(args=None):
    rclpy.init(args=args)
    node = InitialPosePublisher()
    rclpy.spin(node)
    rclpy.shutdown()