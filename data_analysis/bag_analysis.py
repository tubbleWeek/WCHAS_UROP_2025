import sys
import os
import yaml
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

def process_bag(bag_path, map_yaml_path):
    # Load map data
    with open(map_yaml_path) as f:
        map_info = yaml.safe_load(f)
    map_dir = os.path.dirname(map_yaml_path)
    image_path = os.path.join(map_dir, map_info['image'])
    map_img = plt.imread(image_path)
    resolution = map_info['resolution']
    origin = map_info['origin']  # [x, y, theta]
    
    # Calculate image extent for plotting
    map_height, map_width = map_img.shape[:2]
    left = origin[0]
    right = origin[0] + map_width * resolution
    bottom = origin[1]
    top = origin[1] + map_height * resolution
    extent = [left, right, bottom, top]

    # Load bag data
    odom_data = []
    model_pose_data = []
    
    with Reader(bag_path) as reader:
        connections = [c for c in reader.connections]
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            if connection.topic == '/pf/odom':
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
    
    # Trajectory comparison with map
    plt.subplot(1, 2, 1)
    plt.imshow(map_img, cmap='gray', extent=extent, alpha=0.5)
    plt.plot(odom_x, odom_y, 'b-', label='Odometry', alpha=0.8)
    plt.plot(model_x, model_y, 'r-', label='Tracked Model', alpha=0.8)
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Trajectory Comparison')
    # plt.legend(top=True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
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
    if len(sys.argv) < 3:
        print("Usage: python bag_analysis.py <bag_directory> <map_yaml_file>")
        sys.exit(1)
    process_bag(sys.argv[1], sys.argv[2])