# Improving Controller Accuracy: Increasing Localization Accuracy Using Sensor Fusion
This repository hold the ROS2 nodes and python scripts used for evaluating localization techniques.


ROS Version: Humble


Gazebo Version: Classic

## To Run

Run the particle filter
```
ros2 run particle_filter particle_filter
```

Launch the simulation
```
ros2 launch world_launching tb3_sim.py params_file:=/$PATH TO PARAMS IN CONFIG FOLDER/nav2_params.yaml
```

Run the controller
```
ros2 run controllers mppi_controller
```

If you want to track the model in gazebo run:
```
ros2 run project_tools gazebo_model_tracker
```
This posts a Pose message to the /tracked_model_pose topic.


## Data analytics

3 main scripts to use `bag_analysis.py`, `f1tenth_bag_analysis.py`, and `overlay_analysis.py`

bag_analysis only plots one trajectory

```
python bag_analysis.py <bag_directory> <map_yaml_file>
```

overlay_analysis plots two trajectories, and expects two different odom topics in the ros bag

```
python overlay_analysis.py <bag_directory> <map_yaml_file>

```

f1tenth_bag_analysis expects 2 rosbags. 1 had the phasespace readings. the other has the 2 odom topics.

```
python f1tenth_bag_analysis.py <phasespace_bag_dir> <odom_bag_dir> <map.yaml>
```