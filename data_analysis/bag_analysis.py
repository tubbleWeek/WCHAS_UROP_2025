import sys
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

def process_bag(bag_path):
    odom_data = []
    model_pose_data = []
    
    with Reader(bag_path) as reader:
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

    # Convert to numpy arrays
    odom_times = np.array([p['timestamp'] for p in odom_data])
    odom_x = np.array([p['x'] for p in odom_data])
    odom_y = np.array([p['y'] for p in odom_data])
    
    model_times = np.array([p['timestamp'] for p in model_pose_data])
    model_x = np.array([p['x'] for p in model_pose_data])
    model_y = np.array([p['y'] for p in model_pose_data])

    # Create interpolation functions
    interp_x = interp1d(odom_times, odom_x, kind='linear', 
                       bounds_error=False, fill_value="extrapolate")
    interp_y = interp1d(odom_times, odom_y, kind='linear',
                       bounds_error=False, fill_value="extrapolate")

    # Get interpolated odom positions at model pose times
    interp_odom_x = interp_x(model_times)
    interp_odom_y = interp_y(model_times)
    
    # Calculate distances
    dx = model_x - interp_odom_x
    dy = model_y - interp_odom_y
    distances = np.sqrt(dx**2 + dy**2)

    # Statistics
    avg_distance = np.nanmean(distances)
    max_distance = np.nanmax(distances)
    min_distance = np.nanmin(distances)

    # Plotting
    plt.figure(figsize=(12, 6))
    
    # Trajectory comparison
    plt.subplot(1, 2, 1)
    plt.plot(odom_x, odom_y, 'b-', label='Odometry (Interpolated)', alpha=0.5)
    plt.plot(model_x, model_y, 'r-', label='Tracked Model', alpha=0.5)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Trajectory Comparison')
    plt.legend()
    
    # Distance distribution
    plt.subplot(1, 2, 2)
    plt.hist(distances, bins=50, alpha=0.7)
    plt.xlabel('Euclidean Distance (m)')
    plt.ylabel('Frequency')
    plt.title(f'Distance Distribution\nAvg: {avg_distance:.4f}m, Max: {max_distance:.4f}m')
    
    plt.tight_layout()
    plt.show()

    print(f"Average distance: {avg_distance:.6f}m")
    print(f"Max distance: {max_distance:.6f}m")
    print(f"Min distance: {min_distance:.6f}m")

if __name__ == '__main__':
    process_bag(sys.argv[1])