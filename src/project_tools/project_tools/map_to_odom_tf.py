import rclpy
from rclpy.node import Node
from tf2_ros import StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped
import yaml
import math

class MapToOdomTF(Node):
    def __init__(self):
        super().__init__('map_to_odom_tf_publisher')
        
        # Declare parameter for map file path
        self.declare_parameter('map_file', 'map.yaml')
        map_file = self.get_parameter('map_file').value
        
        # Load map file and extract origin
        try:
            with open(map_file, 'r') as f:
                map_data = yaml.safe_load(f)
                origin = map_data['origin']
                self.get_logger().info(f"Loaded origin from map file: {origin}")
        except Exception as e:
            self.get_logger().error(f"Failed to load map file: {str(e)}")
            rclpy.shutdown()
            return
        
        # Validate origin format (supports [x, y, theta] or [x, y, z, theta])
        if len(origin) not in [3, 4]:
            self.get_logger().error("Invalid origin format. Expected 3 or 4 elements")
            rclpy.shutdown()
            return
        
        # Extract coordinates and orientation
        x = origin[0]
        y = origin[1]
        # z = origin[2] if len(origin) == 4 else 0.0
        theta = 0.0
        
        # Convert yaw to quaternion
        quaternion = self.yaw_to_quaternion(theta)
        
        # Create static transform broadcaster
        self.tf_broadcaster = StaticTransformBroadcaster(self)
        
        # Create transform message
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = 'map'
        transform.child_frame_id = 'odom'
        transform.transform.translation.x = 0.0
        transform.transform.translation.y = 0.0
        # transform.transform.translation.z = z
        transform.transform.rotation = quaternion
        
        # Send static transform
        self.tf_broadcaster.sendTransform(transform)
        self.get_logger().info("Published static transform from map to odom")
        
    def yaw_to_quaternion(self, theta):
        from geometry_msgs.msg import Quaternion
        return Quaternion(
            x=0.0,
            y=0.0,
            z=math.sin(theta / 2),
            w=math.cos(theta / 2)
        )
        
def main(args=None):
    rclpy.init(args=args)
    node = MapToOdomTF()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()