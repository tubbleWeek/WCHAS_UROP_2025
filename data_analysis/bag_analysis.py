import sys
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
import matplotlib.pyplot as plt
import numpy as np

def calculate_distance(pose1, pose2):
    """Calculate Euclidean distance between two positions"""
    dx = pose1.x - pose2.x
    dy = pose1.y - pose2.y
    return np.sqrt(dx**2 + dy**2)

def process_bag(bag_path):
    odom_data = []
    model_pose_data = []
    
    with Reader(bag_path) as reader:
        # Create a dictionary to map connections to their topic and data type
        connections = [c for c in reader.connections]
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            if connection.topic == '/odom':
                msg = deserialize_cdr(rawdata, connection.msgtype)
                odom_data.append({
                    'timestamp': timestamp,
                    'x': msg.pose.pose.position.x,
                    'y': msg.pose.pose.position.y
                })
            elif connection.topic == '/tracked_model_pose':
                msg = deserialize_cdr(rawdata, connection.msgtype)
                model_pose_data.append({
                    'timestamp': timestamp,
                    'x': msg.position.x,
                    'y': msg.position.y
                })

    # Time alignment and distance calculation
    distances = []
    odom_idx = 0
    
    for model_pose in model_pose_data:
        # Find closest odom message in time
        while (odom_idx < len(odom_data)-1 and 
               abs(odom_data[odom_idx+1]['timestamp'] - model_pose['timestamp']) < 
               abs(odom_data[odom_idx]['timestamp'] - model_pose['timestamp'])):
            odom_idx += 1
            
        # Calculate distance
        odom_pose = odom_data[odom_idx]
        distances.append(calculate_distance(
            type('', (object,), odom_pose),
            type('', (object,), model_pose)
        ))
    
    # Calculate statistics
    avg_distance = np.mean(distances)
    max_distance = np.max(distances)
    min_distance = np.min(distances)

    # Plotting
    plt.figure(figsize=(12, 6))
    
    # Plot trajectories
    plt.subplot(1, 2, 1)
    plt.plot([p['x'] for p in odom_data], [p['y'] for p in odom_data], label='Odometry')
    plt.plot([p['x'] for p in model_pose_data], [p['y'] for p in model_pose_data], label='Tracked Model')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Trajectory Comparison')
    plt.legend()
    plt.grid(True)

    # Plot distance histogram
    plt.subplot(1, 2, 2)
    plt.hist(distances, bins=50, alpha=0.7)
    plt.xlabel('Euclidean Distance')
    plt.ylabel('Frequency')
    plt.title(f'Distance Distribution\nAvg: {avg_distance:.2f}m, Max: {max_distance:.2f}m')

    plt.tight_layout()
    plt.show()

    print(f"Average Euclidean distance: {avg_distance:.4f} meters")
    print(f"Maximum distance: {max_distance:.4f} meters")
    print(f"Minimum distance: {min_distance:.4f} meters")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python bag_analysis.py <path_to_bag_directory>")
        sys.exit(1)
        
    process_bag(sys.argv[1])
