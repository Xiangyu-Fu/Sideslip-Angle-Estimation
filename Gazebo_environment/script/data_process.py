#!/usr/bin/env python

import math
import rospy
import message_filters
import numpy as np
from collections import deque
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64, Bool, Float64MultiArray


class DataProcess(object):
    def __init__(self):
        # init node
        self.node_name = 'data_process'
        rospy.init_node(self.node_name)

        # set publish rate
        self.r = rospy.Rate(100)
        self.tyre_radius = 0.3671951254
        self.acc_y_deque = deque(maxlen=20)

        # Subscribers
        imu_data = message_filters.Subscriber("/catvehicle/imu", Imu)
        joint_data = message_filters.Subscriber("/catvehicle/joint_states", JointState)
        steering_data = message_filters.Subscriber("/catvehicle/ste_angle", Float64)

        sync = message_filters.ApproximateTimeSynchronizer([imu_data, joint_data, steering_data], 10, 1, allow_headerless=True)
        sync.registerCallback(self.data_callback)

        # Publishers
        self.pub1 = rospy.Publisher("/SSAE/processed_data", Float64MultiArray, queue_size=1)
        self.pub2 = rospy.Publisher("/SSAE/deltav", Float64, queue_size=10)
        self.pub3 = rospy.Publisher("/SSAE/yawrate", Float64, queue_size=10)

        rospy.loginfo("start data process node ...")

    def data_callback(self, imu_data, joint_states, steering_data):
        try:
            # get sensor data from value
            yaw_rate = imu_data.angular_velocity.z  # read yaw rate from imu sensor
            acc_x = imu_data.linear_acceleration.x  # Read the linear acceleration in two direction
            acc_y_t = imu_data.linear_acceleration.y
            self.acc_y_deque.append(acc_y_t)
            acc_y = np.mean(self.acc_y_deque)

            # calculate the longitudinal velocity
            vel_x = joint_states.velocity[0] * self.tyre_radius
            # get steering angle
            steering_angle = steering_data.data

            # get final output array
            processed_data = Float64MultiArray()
            processed_data.data = [acc_x, acc_y, yaw_rate, vel_x, steering_angle]
            self.pub2.publish(steering_angle)
            self.pub3.publish(yaw_rate)
            self.pub1.publish(processed_data)
            self.r.sleep()

        except(IndexError):
            pass


if __name__ == '__main__':
    try:
        DataProcess()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
