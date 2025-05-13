import sys
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

def read_bag(bag_path, target_topic, is_phasespace=False):
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
                # Phasespace uses (X, -Z) → convert to standard (X, Y)
                x = msg.transform.translation.x / 1000.0  # mm → m
                y = -msg.transform.translation.z / 1000.0  # mm → m
            else:  # Odometry (standard X, Y)
                x = msg.pose.pose.position.x
                y = msg.pose.pose.position.y

            data.append({'timestamp': timestamp_ns, 'x': x, 'y': y})
    
    return data

def process_bags(phasespace_bag_path, odom_bag_path):
    # Read data from both bags
    phasespace_data = read_bag(phasespace_bag_path, '/phasespace', is_phasespace=True)
    odom_data = read_bag(odom_bag_path, '/odom')

    # Convert to numpy arrays
    phasespace_times = np.array([p['timestamp'] for p in phasespace_data])
    phasespace_x = np.array([p['x'] for p in phasespace_data])
    phasespace_y = np.array([p['y'] for p in phasespace_data])

    odom_times = np.array([p['timestamp'] for p in odom_data])
    odom_x = np.array([p['x'] for p in odom_data])
    odom_y = np.array([p['y'] for p in odom_data])

    # Find overlapping time window
    start_time = max(phasespace_times.min(), odom_times.min())
    end_time = min(phasespace_times.max(), odom_times.max())

    # Filter data to overlapping region
    phasespace_mask = (phasespace_times >= start_time) & (phasespace_times <= end_time)
    phasespace_times = phasespace_times[phasespace_mask]
    phasespace_x = phasespace_x[phasespace_mask]
    phasespace_y = phasespace_y[phasespace_mask]

    odom_mask = (odom_times >= start_time) & (odom_times <= end_time)
    odom_times = odom_times[odom_mask]
    odom_x = odom_x[odom_mask]
    odom_y = odom_y[odom_mask]

    # Trim faulty start/end points from Phasespace data
    trim_start = 1  # Number of points to remove from start
    trim_end = 54    # Number of points to remove from end

    # Ensure valid trimming
    if len(phasespace_times) > trim_start + trim_end:
        phasespace_times = phasespace_times[trim_start:-trim_end]
        phasespace_x = phasespace_x[trim_start:-trim_end]
        phasespace_y = phasespace_y[trim_start:-trim_end]
    else:
        print("Warning: Insufficient data points for trimming")

    # Interpolate odometry to Phasespace timestamps
    interp_odom_x = interp1d(odom_times, odom_x, kind='linear', 
                            bounds_error=False, fill_value="extrapolate")
    interp_odom_y = interp1d(odom_times, odom_y, kind='linear',
                            bounds_error=False, fill_value="extrapolate")
    
    odom_x_interp = interp_odom_x(phasespace_times)
    odom_y_interp = interp_odom_y(phasespace_times)

    # Calculate initial alignment offset using first 10 points
        # Calculate initial alignment offset using the first valid point
    if len(phasespace_x) >= 1 and len(odom_x_interp) >= 1:
        dx_avg = phasespace_x[0] - odom_x_interp[0]
        dy_avg = phasespace_y[0] - odom_y_interp[0]
        # Apply offset to entire Phasespace trajectory
        phasespace_x -= dx_avg
        phasespace_y -= dy_avg
    else:
        print("Warning: Insufficient data for alignment")

    # Calculate Euclidean distances
    dx = phasespace_x - odom_x_interp
    dy = phasespace_y - odom_y_interp
    distances = np.sqrt(dx**2 + dy**2)

    # Statistics
    avg_dist = np.nanmean(distances)
    max_dist = np.nanmax(distances)
    min_dist = np.nanmin(distances)

    # Plotting
    plt.figure(figsize=(12, 6))
    
    # Trajectory comparison
    plt.subplot(1, 2, 1)
    plt.plot(odom_x, odom_y, 'b-', label='Odometry (Raw)', alpha=0.3)
    plt.plot(phasespace_x, phasespace_y, 'r-', label='Phasespace (Adjusted)', linewidth=2)
    plt.plot(odom_x_interp, odom_y_interp, 'g--', label='Odometry (Interpolated)')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Trajectory Comparison with Initial Alignment')
    plt.legend()

    # Error distribution
    plt.subplot(1, 2, 2)
    plt.hist(distances, bins=20, alpha=0.7, color='purple')
    plt.xlabel('Position Error (m)')
    plt.ylabel('Frequency')
    plt.title(f'Error Distribution\nAvg: {avg_dist:.4f}m | Max: {max_dist:.4f}m')

    plt.tight_layout()
    plt.show()

    print(f"Average Error: {avg_dist:.6f} meters")
    print(f"Maximum Error: {max_dist:.6f} meters")
    print(f"Minimum Error: {min_dist:.6f} meters")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python f1tenth_bag_analysis.py <phasespace_bag_dir> <odom_bag_dir>")
        sys.exit(1)
    process_bags(sys.argv[1], sys.argv[2])