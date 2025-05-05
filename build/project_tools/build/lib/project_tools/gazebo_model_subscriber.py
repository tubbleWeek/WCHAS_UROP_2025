import rclpy
from rclpy.node import Node
from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import GetEntityState
from geometry_msgs.msg import Pose

class ModelTrackerNode(Node):
    def __init__(self):
        super().__init__('model_tracker_node')
        self.declare_parameter('model_name', "turtlebot3_waffle")
        
        # Correct service name to '/gazebo/get_entity_state'
        self.client = self.create_client(GetEntityState, '/get_entity_state')
        self.publisher = self.create_publisher(Pose, "/tracked_model_pose", 10)
        
        # Wait for service to be available
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting...')
        
        self.timer = self.create_timer(0.01, self.timer_callback)

    # def timer_callback(self):
    #     # Retrieve current model_name parameter
    #     model_name = self.get_parameter('model_name').get_parameter_value().string_value
        
    #     # Create service request
    #     request = GetEntityState.Request()
    #     request.name = model_name
        
    #     # Call service asynchronously and wait for response
    #     future = self.client.call_async(request)
    #     rclpy.spin_until_future_complete(self, future)
        
    #     if future.result() is not None:
    #         response = future.result()
    #         if response.success:
    #             # Publish the model's pose
    #             self.publisher.publish(response.state.pose)
    #         else:
    #             self.get_logger().error(f"Failed to get state for model '{model_name}'")
    #     else:
    #         self.get_logger().error("Service call failed")
    def timer_callback(self):
        model_name = self.get_parameter('model_name').value
        # self.get_logger().info(f"Requesting state for: {model_name}")  # DEBUG
        
        request = GetEntityState.Request(name=model_name)
        future = self.client.call_async(request)
        future.add_done_callback(lambda f: self.service_callback(f, model_name))

    def service_callback(self, future, model_name):
        try:
            response = future.result()
            if response.success:
                self.publisher.publish(response.state.pose)
                # self.get_logger().info('Pose published')  # DEBUG
            else:
                self.get_logger().error(f"Service succeeded but model '{model_name}' not found")
        except Exception as e:
            self.get_logger().error(f"Service call failed: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = ModelTrackerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()