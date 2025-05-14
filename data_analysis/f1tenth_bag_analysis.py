import sys
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import yaml
import os



def read_bag(bag_path, target_topic, is_phasespace=False, is_pf=False):
    """Read a bag file and extract data from the specified topic."""
    data = []
    with Reader(bag_path) as reader:
        connections = [c for c in reader.connections if c.topic == target_topic]
        for connection, _, rawdata in reader.messages(connections=connections):
            msg = deserialize_cdr(rawdata, connection.msgtype)
            
            # Extract timestamp from message header
            stamp = msg.header.stamp
            timestamp_ns = stamp.sec * 10**9 + stamp.nanosec

            # Handle coordinate system conversion for Phasespace
            if is_phasespace:
                # Phasespace uses (X, -Z) convert to standard (X, Y)
                y = -msg.transform.translation.x / 1000.0  # mm 
                x = -msg.transform.translation.z / 1000.0  # mm 
            elif is_pf:
                x = msg.pose.pose.position.x
                y = msg.pose.pose.position.y
            else:  # Odometry 
                y = -msg.pose.pose.position.x
                x = msg.pose.pose.position.y

            data.append({'timestamp': timestamp_ns, 'x': x, 'y': y})
    
    return data

def process_bags(phasespace_bag_path, odom_bag_path, map_yaml_path):
    # Read data from both bags
    # Load map data
    with open(map_yaml_path) as f:
        map_info = yaml.safe_load(f)
    map_dir = os.path.dirname(map_yaml_path)
    image_path = os.path.join(map_dir, map_info['image'])
    map_img = plt.imread(image_path)
    resolution = map_info['resolution']
    origin = map_info['origin']  # [x, y, theta]
    
    # Calculate image extent for plotting (ROS coordinate system)
    map_height, map_width = map_img.shape[:2]
    left = origin[0]
    right = origin[0] + map_width * resolution
    bottom = origin[1]
    top = origin[1] + map_height * resolution
    extent = [left, right, bottom, top]

    # Read data from all topics
    phasespace_data = read_bag(phasespace_bag_path, '/phasespace', is_phasespace=True)
    odom_data = read_bag(odom_bag_path, '/odom')
    pf_odom_data = read_bag(odom_bag_path, '/pf/odom', is_pf=True)  # Added particle filter odometry

    # Convert to numpy arrays
    phasespace_times = np.array([p['timestamp'] for p in phasespace_data])
    phasespace_x = np.array([p['x'] for p in phasespace_data])
    phasespace_y = np.array([p['y'] for p in phasespace_data])

    odom_times = np.array([p['timestamp'] for p in odom_data])
    odom_x = np.array([p['x'] for p in odom_data])
    odom_y = np.array([p['y'] for p in odom_data])

    # Process particle filter odometry data
    pf_odom_times = np.array([p['timestamp'] for p in pf_odom_data])
    pf_odom_x = np.array([p['x'] for p in pf_odom_data])
    pf_odom_y = np.array([p['y'] for p in pf_odom_data])

    # Find overlapping time window for all three data sources
    start_time = max(phasespace_times.min(), odom_times.min(), pf_odom_times.min())
    end_time = min(phasespace_times.max(), odom_times.max(), pf_odom_times.max())

    # Filter data to overlapping region
    phasespace_mask = (phasespace_times >= start_time) & (phasespace_times <= end_time)
    phasespace_times = phasespace_times[phasespace_mask]
    phasespace_x = phasespace_x[phasespace_mask]
    phasespace_y = phasespace_y[phasespace_mask]

    odom_mask = (odom_times >= start_time) & (odom_times <= end_time)
    odom_times = odom_times[odom_mask]
    odom_x = odom_x[odom_mask]
    odom_y = odom_y[odom_mask]

    pf_odom_mask = (pf_odom_times >= start_time) & (pf_odom_times <= end_time)
    pf_odom_times = pf_odom_times[pf_odom_mask]
    pf_odom_x = pf_odom_x[pf_odom_mask]
    pf_odom_y = pf_odom_y[pf_odom_mask]

    # Trim faulty start/end points from Phasespace data
    trim_start = 80  # Number of points to remove from start
    trim_end = 40   # Number of points to remove from end
    
    # Ensure valid trimming
    if len(phasespace_times) > trim_start + trim_end:
        phasespace_times = phasespace_times[trim_start:-trim_end]
        phasespace_x = phasespace_x[trim_start:-trim_end]
        phasespace_y = phasespace_y[trim_start:-trim_end]
    else:
        print("Warning: Insufficient data points for trimming")

    # Interpolate both odometry sources to Phasespace timestamps
    interp_odom_x = interp1d(odom_times, odom_x, kind='linear', 
                            bounds_error=False, fill_value="extrapolate")
    interp_odom_y = interp1d(odom_times, odom_y, kind='linear',
                            bounds_error=False, fill_value="extrapolate")
    
    interp_pf_odom_x = interp1d(pf_odom_times, pf_odom_x, kind='linear', 
                              bounds_error=False, fill_value="extrapolate")
    interp_pf_odom_y = interp1d(pf_odom_times, pf_odom_y, kind='linear',
                              bounds_error=False, fill_value="extrapolate")
    
    odom_x_interp = interp_odom_x(phasespace_times)
    odom_y_interp = interp_odom_y(phasespace_times)
    
    pf_odom_x_interp = interp_pf_odom_x(phasespace_times)
    pf_odom_y_interp = interp_pf_odom_y(phasespace_times)

    # Calculate initial alignment offset using the first valid point
    if len(phasespace_x) >= 1 and len(odom_x_interp) >= 1:
        # Calculate offsets to align odometry with phasespace (ground truth)
        dx_odom_avg = odom_x_interp[0] - phasespace_x[0]
        dy_odom_avg = odom_y_interp[0] - phasespace_y[0]
        
        # Calculate offset for particle filter to align with phasespace
        dx_pf_avg = pf_odom_x_interp[0] - phasespace_x[0]
        dy_pf_avg = pf_odom_y_interp[0] - phasespace_y[0]
        
        # Apply offsets to odometry trajectories to align with phasespace
        odom_x_aligned = odom_x_interp - dx_odom_avg
        odom_y_aligned = odom_y_interp - dy_odom_avg
        
        pf_odom_x_aligned = pf_odom_x_interp - dx_pf_avg
        pf_odom_y_aligned = pf_odom_y_interp - dy_pf_avg
    else:
        print("Warning: Insufficient data for alignment")
        odom_x_aligned = odom_x_interp
        odom_y_aligned = odom_y_interp
        pf_odom_x_aligned = pf_odom_x_interp
        pf_odom_y_aligned = pf_odom_y_interp

    # Calculate Euclidean distances for regular odometry
    dx = phasespace_x - odom_x_aligned
    dy = phasespace_y - odom_y_aligned
    distances_odom = np.sqrt(dx**2 + dy**2)

    # Calculate Euclidean distances for particle filter odometry
    dx_pf = phasespace_x - pf_odom_x_aligned
    dy_pf = phasespace_y - pf_odom_y_aligned
    distances_pf = np.sqrt(dx_pf**2 + dy_pf**2)

    # Statistics for regular odometry
    avg_dist_odom = np.nanmean(distances_odom)
    max_dist_odom = np.nanmax(distances_odom)
    min_dist_odom = np.nanmin(distances_odom)

    # Statistics for particle filter odometry
    avg_dist_pf = np.nanmean(distances_pf)
    max_dist_pf = np.nanmax(distances_pf)
    min_dist_pf = np.nanmin(distances_pf)

    # Plotting
    plt.figure(figsize=(15, 10))
    print(len(odom_x_aligned), len(pf_odom_x_aligned))
    # Trajectory comparison with map
    plt.subplot(2, 2, 1)
    plt.imshow(map_img, extent=extent, cmap='gray', alpha=0.7, origin='lower')
    plt.plot(phasespace_x, phasespace_y, 'r-', label='Phasespace (Ground Truth)', linewidth=2)
    plt.plot(odom_x_aligned, odom_y_aligned, 'g--', label='Regular Odometry')
    plt.plot(pf_odom_x_aligned, pf_odom_y_aligned, 'b-.', label='Particle Filter')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('All Trajectories Comparison')
    plt.legend()
    plt.grid(True)

    # Error distribution for regular odometry
    plt.subplot(2, 2, 2)
    plt.hist(distances_odom, bins=20, alpha=0.7, color='green', label='Regular Odometry Error')
    plt.xlabel('Position Error (m)')
    plt.ylabel('Frequency')
    plt.title(f'Regular Odometry Error\nAvg: {avg_dist_odom:.4f}m | Max: {max_dist_odom:.4f}m')
    plt.legend()
    plt.grid(True)

    # Error distribution for particle filter odometry
    plt.subplot(2, 2, 3)
    plt.hist(distances_pf, bins=20, alpha=0.7, color='blue', label='Particle Filter Error')
    plt.xlabel('Position Error (m)')
    plt.ylabel('Frequency')
    plt.title(f'Particle Filter Error\nAvg: {avg_dist_pf:.4f}m | Max: {max_dist_pf:.4f}m')
    plt.legend()
    plt.grid(True)

    # Comparative error plot over time
    plt.subplot(2, 2, 4)
    rel_times = (phasespace_times - phasespace_times[0]) / 1e9  # Convert to seconds from start
    plt.plot(rel_times, distances_odom, 'g-', alpha=0.7, label='Regular Odometry Error')
    plt.plot(rel_times, distances_pf, 'b-', alpha=0.7, label='Particle Filter Error')
    plt.xlabel('Time (s)')
    plt.ylabel('Error (m)')
    plt.title('Error Comparison Over Time')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Print statistics
    print("\n--- Regular Odometry Statistics ---")
    print(f"Average Error: {avg_dist_odom:.6f} meters")
    print(f"Maximum Error: {max_dist_odom:.6f} meters")
    print(f"Minimum Error: {min_dist_odom:.6f} meters")
    
    print("\n--- Particle Filter Odometry Statistics ---")
    print(f"Average Error: {avg_dist_pf:.6f} meters")
    print(f"Maximum Error: {max_dist_pf:.6f} meters")
    print(f"Minimum Error: {min_dist_pf:.6f} meters")
    
    # Compare the two odometry methods
    error_reduction = ((avg_dist_odom - avg_dist_pf) / avg_dist_odom) * 100 if avg_dist_odom > 0 else 0
    print(f"\n--- Comparison ---")
    print(f"Error reduction with particle filter: {error_reduction:.2f}%")
    print(f"Particle filter performs {'better' if avg_dist_pf < avg_dist_odom else 'worse'} than regular odometry")

if __name__ == '__main__':
    
    if len(sys.argv) != 4:
        print("Usage: python f1tenth_bag_analysis.py <phasespace_bag_dir> <odom_bag_dir> <map.yaml>")
        sys.exit(1)
    process_bags(sys.argv[1], sys.argv[2], sys.argv[3])