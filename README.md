<p align="center">
<a href="https://anaconda.org/conda-forge/black/"><img alt="conda-forge" src="https://img.shields.io/conda/dn/conda-forge/black.svg?label=conda-forge"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>


# Sideslip-Angle-Estimation
My bachelor thesis, which based on ROS/Gazebo as well as Matlab/Simulink.

The video about the Gazebo simulation environment :https://www.youtube.com/watch?v=I3iScs9OWoQ&t=2s

## The simulation steps
Requirements:
Ubuntu 18
Ros Melodic
Gazebo 9
opencv library 
numpy library

refer to: https://github.com/jmscslgroup/catvehicle/

WorkSpace:
```
catkin_ws
```

1. make workspace:
```
cd ~
mkdir -p catkin_ws/src
cd catkin_ws/src
catkin_init_workspace
cd ..
catkin_make
```

2. copy core files
```
copy carbot and catvehicle packages to catkin_ws/src
```

3*. two methods are provided to obtain the other dependent packages:
```
copy other dependent packages from thsi folder to catkin_ws/src
```
or
```
clone other dependent packages from Github:
cd ~/catkin_ws/src
git clone https://github.com/jmscslgroup/obstaclestopper
git clone https://github.com/jmscslgroup/control_toolbox
git clone https://github.com/jmscslgroup/sicktoolbox
git clone https://github.com/jmscslgroup/sicktoolbox_wrapper
git clone https://github.com/jmscslgroup/stepvel
git clone https://github.com/jmscslgroup/cmdvel2gazebo
git clone https://github.com/jmscslgroup/velodyne
```


4. make workspace
```
cd ~/catkin_ws
catkin_make
```

5. start simulation
``` 
cd catkin_ws
source devel/setup.bash
.src/catvehicle/script/run.bash
```


## The images
The mathmatical model:

![fig1_single_track_model](https://user-images.githubusercontent.com/54738414/149680697-e1a9ad2b-51f0-41c3-82a3-653d82654721.png)

The node graph:

![final_rosgraph](https://user-images.githubusercontent.com/54738414/149680726-c0974429-fd69-4f8e-92d2-b8e449f7552a.png)

The running environment:

![image](https://user-images.githubusercontent.com/54738414/149680782-03b44e5d-b346-4cf3-aa70-05f566f54862.png)

The output:

![image](https://user-images.githubusercontent.com/54738414/149680748-3a098709-fa77-4848-b84b-7d3dc3a7ee64.png)
