#!/usr/bin/python
# coding: utf8

import roslib
import sys
import rospy
import cv2 
import logging  
import math  
import datetime         
import numpy as np
import matplotlib.pyplot as plt   
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image  
from geometry_msgs.msg import Twist
from collections import deque
from std_msgs.msg import Float64


class Vehicle_Control(object):
    def __init__(self):
        self.node_name = 'vehicle_control'
        rospy.init_node(self.node_name)

        rospy.loginfo('start vehicle control node ...')
        self.steering_angle_deque = deque(maxlen=1)
        self.steering_angle_last = 0.0

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/catvehicle/camera_left/image_raw_left", Image, self.camera_callback)
        self.cmd_vel = rospy.Publisher('/catvehicle/cmd_vel', Twist, queue_size=5000)
        self.move_cmd = Twist()
        self.pub3 = rospy.Publisher("/catvehicle/ste_angle", Float64, queue_size=10)

    def finding_edges(self, input_data_image):
        """
        finding edges
        Aim is to find edges of the lane
        1) convert input image to grayscale to reduce the channel to 1
        2) A median filter is used to remove noise in the grayscaled image
        3) A canny function is used to detect edges
        """
        grayscale = cv2.cvtColor(input_data_image, cv2.COLOR_BGR2GRAY)
        median_filter = cv2.medianBlur(grayscale, 3)
        detected_edge = cv2.Canny(median_filter, 50, 150)
        return grayscale, detected_edge

    def zone_of_concern(self, input_data_image):
        """
        Crop the target region
        Aim is to find the target region
        1) select the polygon points which covers the targeted region
        2) create a black image, similar to input image size
        3) fillpoly function will fill the selected polgon region with white color
        4) a bitwise_and operation is performed between edge image and masked image
        """
        #height = input_data_image.shape[0]
        polygons = np.array([
        [(160, 639), (263, 535), (534, 535), (625, 639)]
        ])
        #[(0, 400), (294, 246), (338, 246), (640, 400)]
        black_image = np.zeros_like(input_data_image)
        masked_image = cv2.fillPoly(black_image, polygons, 255)
        output_image = cv2.bitwise_and(input_data_image, masked_image)
        return output_image, masked_image

    def line_slicing(self, input_data_image):
        """
        Crop the target region
        Aim is to obtain lines from a pixel points
        1) distance is pixel is selected as 2
        2) angle in radians as 1 degree
        3) minimum number of votes in each accumulator
        """
        bin_d = 2  
        bin_theta = np.pi / 180  
        min_votes = 100 
        minLineLength=40 
        maxLineGap=5
        lines = cv2.HoughLinesP(input_data_image, bin_d, bin_theta, min_votes, np.array([]), minLineLength, maxLineGap)
        #print('lines', lines) 
        return lines

    def m_and_c_averaged(self, input_data_image, input_data_lines):
        """
        averaging lines on slope and intercept
        """
        left_line    = []
        right_line   = []
        if input_data_lines is None:
           return None
        for line in input_data_lines:
            for x1, y1, x2, y2 in line:
                fit_values = np.polyfit((x1,x2), (y1,y2), 1)
                #print fit_values
                m_slope = fit_values[0]
                c_intercept = fit_values[1]
                if m_slope < 0: 
                    left_line.append((m_slope, c_intercept))
                else:
                    right_line.append((m_slope, c_intercept))
        left_line_avg  = np.average(left_line, axis=0)
        # print('left_line_avg', left_line_avg)
        right_line_avg = np.average(right_line, axis=0)
        # print('right_line_avg', right_line_avg)
        final_left_line  = self.cropped_points(input_data_image, left_line_avg)
        final_right_line = self.cropped_points(input_data_image, right_line_avg)
        averaged_lines = [final_left_line, final_right_line]
        # print('final_left_line', final_left_line)
        # print('final_right_line', final_right_line)
        # print('averaged_lines', averaged_lines)
        return averaged_lines 

    def cropped_points(self, input_data_image, input_data_lines):
        """
        length to be considered for left and right lane line
        Aim is to get image from bottom of the image till bit lower than the middle
        """
        m, c = input_data_lines
        y1 = int(input_data_image.shape[0])
        y2 = int(y1*63/100)         
        x1 = int((y1 - c)/m)
        x2 = int((y2 - c)/m)
        return [[x1, y1, x2, y2]]

    def navigation(self, input_data_image, input_lane_lines):
        """
        finding steering value
        """
        height, width, _ = input_data_image.shape
        if len(input_lane_lines) == 0:
           return 0
        else:
           _, _, left_x2, _ = input_lane_lines[0][0]
           _, _, right_x2, _ = input_lane_lines[1][0]
        image_mid_x = int(width / 2)
        lateral_x_error = float((left_x2 + right_x2) / 2 - image_mid_x)
           
        image_mid_y = int(height / 2)
        target_angle_rad = math.atan(lateral_x_error / image_mid_y) 
        # print('angle_to_mid_radian', angle_to_mid_radian)
        target_angle_deg = target_angle_rad * 180.0 / math.pi  # delete the int function
        # print('angle_to_mid_deg', angle_to_mid_deg)
        steer_angle_deg = target_angle_deg + 90
        # print('steer_angle_deg', steer_angle_deg)
        return steer_angle_deg

    def display_lines(self, input_data_image, input_data_lines):
        """
        fro displaying in image terminal
        :param self:
        :param input_data_image:
        :param input_data_lines:
        :return:
        """
        black_image = np.zeros_like(input_data_image)
        if input_data_lines is not None:
            for line in input_data_lines:
                for x1, y1, x2, y2 in line:
                    tracked_lines = cv2.line(black_image, (x1, y1), (x2, y2), (0, 255, 0), 20)
        return tracked_lines

    def headline_visualize(self, input_data_image, steer_val, line_color=(0, 0, 255), line_width=5, ):
        heading_image = np.zeros_like(input_data_image)
        height, width, _ = input_data_image.shape
        steer_val_radian = steer_val / 180.0 * math.pi
        x1 = int(width / 2)
        y1 = height
        x2 = int(x1 - height / 2 / math.tan(steer_val_radian))
        y2 = int(height * 2 / 3)

        cv2.line(heading_image, (x1, y1), (x2, y2), line_color, line_width)
        heading_image = cv2.addWeighted(input_data_image, 0.8, heading_image, 1, 1)
        return heading_image

    def camera_callback(self,data):
        """
        callback function
        """
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
            lane_image = np.copy(cv_image)
        except CvBridgeError as error:
            pass

        grayscale, detected_edge_image = self.finding_edges(cv_image)
        output_image, masked_image = self.zone_of_concern(detected_edge_image)
        finding_line = self.line_slicing(output_image)
        averaged_lines = self.m_and_c_averaged(lane_image, finding_line)

        displayed_line = self.display_lines(cv_image, averaged_lines)
        combined_image = cv2.addWeighted(lane_image, 0.8, displayed_line, 1, 1)
        computed_steering_angle = self.navigation(lane_image, averaged_lines)
        # stabilizing_steering_angle = self.stabilize_steering_angle(self.curr_steering_angle, computed_steering_angle, 2)
        displayed_heading_line = self.headline_visualize(lane_image, computed_steering_angle)
        nextcombo_image = cv2.addWeighted(combined_image, 0.8, displayed_heading_line, 1, 1)
        pa = 90 - computed_steering_angle
        ang_vel = (pa * math.pi / 180)

        # steering angle smoothing
        alpha = 0.2
        steering_angle_t = alpha * ang_vel + (1 - alpha) * self.steering_angle_last
        self.steering_angle_deque.append(steering_angle_t)
        steering_angle = np.mean(self.steering_angle_deque)
        self.steering_angle_last = steering_angle
        self.pub3.publish(steering_angle)

        # publish data for vehicle control
        veh_vel = 10
        self.move_cmd.linear.x = veh_vel
        self.move_cmd.angular.z = steering_angle
        self.cmd_vel.publish(self.move_cmd)

        # display the processed image
        cv2.putText(nextcombo_image, 'Linear Speed [m/s]: '+str(veh_vel)[:7],(40,80), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.6,(0,0,0),2,cv2.LINE_AA)
        cv2.putText(nextcombo_image, 'Angular Speed [rad/s]: '+str(ang_vel)[:7],(40,150), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.6,(0,0,0),2,cv2.LINE_AA)
        cv2.imshow('nextcombo_image', nextcombo_image)
        cv2.waitKey(2)


if __name__ == '__main__':
    try:
        Vehicle_Control()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass