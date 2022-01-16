#! /bin/bash
gnome-terminal -- bash -c "killall gzserver" &
sleep 1 &&
echo ' Gazeobo starts running ... '

gnome-terminal -- bash -c "roslaunch carbot_gazebo carbot_world.launch" &
sleep 25 &&
echo ' Cat vehicle model is loading ... '

gnome-terminal -- bash -c "roslaunch catvehicle catvehicle_spawn.launch robot:=catvehicle X:=2 Y:=1.5 \yaw:=0" &
sleep 10 &&
echo ' data process node starts running ... '

gnome-terminal -- bash -c "rosrun catvehicle data_process.py" &
sleep 5 &&
echo ' sideslip angle estimation node starts running ... '

gnome-terminal -- bash -c "rosrun catvehicle SideslipAngleEstimation.py" &
sleep 5 &&
echo ' image process & vehicle control node starts running ... '

gnome-terminal -- bash -c "rqt" &
sleep 5 &&

gnome-terminal -- bash -c "rosrun catvehicle vehicle_control.py" &
sleep 5 &&


echo ' '
echo ' finish!'
echo ' '
