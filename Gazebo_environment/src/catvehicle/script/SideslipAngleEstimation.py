#!/usr/bin/env python

import math
import rospy
import numpy as np
from std_msgs.msg import Float64, Bool, Float64MultiArray
import matplotlib.pyplot as plt
from collections import deque


class Side_Slip_Angle_Estimation(object):
    def __init__(self):
        # init node
        self.node_name = 'SideSlipAngle_Estimation'
        rospy.init_node(self.node_name)
        rospy.loginfo('start side slip angle estimation node ...')
        # Set process and sensor noise for EKF of dynamics model
        self.process_noise_v_kd = np.array([[0.001],
                                            [0.001]])

        self.sensor_noise_w_kd = np.array([[0.0001],
                                           [0.0001]])

        self.Q_kd = np.array([[0.01, 0],
                              [0, 0.01]])

        self.R_kd = np.array([[0.005, 0],
                              [0, 0.005]])

        # Set process and sensor noise for EKF of kinematics model
        self.process_noise_v_kk = np.array([[0.1],
                                            [0.1]])

        self.sensor_noise_w_kk = np.array([[0.25],
                                           [0.25]])

        self.Q_kk = np.array([[0.02, 0],
                              [0, 0.02]])

        self.R_kk = np.array([[0.001, 0],
                              [0, 0.001]])
        # Set Constant Parameter of vehicle
        self.C_f = 169265.0
        self.C_r = 249962.5
        self.L_f = 1.55
        self.L_r = 1.05
        self.m_v = 1883.239
        self.I_z = 2529.4827

        # initialisation of EKF
        self.beta = []
        self.v_y = 0
        self.v_x = 0
        self.state_estimate_kd = np.array([[0],
                                           [0]])
        self.P_kd = np.array([[0.1, 0], [0, 0.1]])
        self.state_estimate_dd = np.array([[0],
                                           [0]])
        self.P_dd = np.array([[0.1, 0], [0, 0.1]])

        # trajectory calculation
        self.last_beta = 0
        self.x0 = 0
        self.y0 = 0
        self.x_buffer = [0]
        self.y_buffer = [0]
        self.psi = 0
        self.last_time = rospy.get_time()

        self.pub1 = rospy.Publisher("/SSAE/beta", Float64, queue_size=10)

        self.sub1 = rospy.Subscriber("SSAE/processed_data", Float64MultiArray, self.EstCallback)

    def ekf_update_k(self, A_k, B_k, C_k, z_k_observation_vector, state_estimate_k, control_vector_k, P_k):
        """
        EKF update function of kinematics model
        :param A_k: A_d or A_k
        :param B_k B_d or B_k
        :param C_k: measurement matrix
        :param z_k_observation_vector: measurement data directly from sensor, y_d
        :param state_estimate_k: x_d, state estimation at time k
        :param control_vector_k: u_d, control vector at time k
        :param P_k: covariance estimation at time k
        :return:state estimation [v_x, v_y]**T, covariance
        """

        # Predicted state estimate
        state_estimate_k = np.matmul(A_k, state_estimate_k) + np.matmul(B_k, control_vector_k) + self.process_noise_v_kk
        # print(f'Kinematic State Estimate Before EKF={state_estimate_k.T}')

        # Predicted covariance estimate
        P_k = np.matmul(np.matmul(A_k, P_k), A_k.T) + self.Q_kk

        # reset model if velocity is too small
        if self.v_x < 0.05:
            P_k = np.array([[0.1, 0], [0, 0.1]])
        # print('P_kd={}'.format(P_k))

        # Measurement residual
        measurement_residual_y_k = z_k_observation_vector - (
            ((np.matmul(C_k, state_estimate_k)) + self.sensor_noise_w_kk))

        # print(f'Kinematic Observation={z_k_observation_vector.T}')

        # Residual covariance
        S_k = np.matmul(np.matmul(C_k, P_k), C_k.T) + self.R_kk

        # Kalman gain
        K_k = np.matmul(np.matmul(P_k, C_k.T), np.linalg.pinv(S_k))

        # update state estimate
        state_estimate_k = state_estimate_k + (np.matmul(K_k, measurement_residual_y_k))

        # update covariance of state estimate
        P_k = P_k - np.matmul(np.matmul(K_k, C_k), P_k)

        # print(f'Kinematic State Estimate After EKF={state_estimate_k.T}\n\n')

        return state_estimate_k, P_k

    def ekf_update_d(self, A_k, B_k, C_k, z_k_observation_vector, state_estimate_k, control_vector_k, P_k):
        """
        EKF update function of Dynamics model
        :param A_k: A_dd
        :param B_k: B_dd
        :param C_k: C_dd
        :param z_k_observation_vector: measurement data directly from sensor, y_d
        :param state_estimate_k: state estimation at time k
        :param control_vector_k: control vector at time k
        :param P_k: covariance estimation at time k
        :return:state estimation [v_y, yaw_rate]**T , covariance
        """

        # Predicted state estimate
        state_estimate_k = np.matmul(A_k, state_estimate_k) + (B_k * control_vector_k) + self.process_noise_v_kd
        # print(f'Dynamic State Estimate Before EKF={state_estimate_k.T}')

        # Predicted covariance estimate
        P_k = np.matmul(np.matmul(A_k, P_k), A_k.T) + self.Q_kd

        # reset model if velocity is too small
        if self.v_x < 0.05:
            P_k = np.array([[0.1, 0], [0, 0.1]])

        # Measurement residual
        measurement_residual_y_k = z_k_observation_vector - ((np.matmul(C_k, state_estimate_k)) + self.sensor_noise_w_kd)

        # print(f'Dynamic Observation={z_k_observation_vector.T}')

        # Residual covariance
        S_k = np.matmul(np.matmul(C_k, P_k), C_k.T) + self.R_kd

        # Kalman gain
        K_k = np.matmul(np.matmul(P_k, C_k.T), np.linalg.pinv(S_k))

        # update state estimate
        state_estimate_k = state_estimate_k + (np.matmul(K_k, measurement_residual_y_k))

        # update covariance of state estimate
        P_k = P_k - np.matmul(np.matmul(K_k, C_k), P_k)

        # print(f'Dynamic State Estimate After EKF={state_estimate_k.T}\n\n')

        return state_estimate_k, P_k

    def get_kinematics_param(self, data, Delta_T):
        """
        calculate discretized kinematics system matrix, which be used in EKF
        :param data:input data, acc_x, acc_y, steering_angle, yaw_rate, vel_x
        :param Delta_T: time interval
        :return: system matrix
        """
        # get the required data
        acc_x, acc_y, steering_angle, yaw_rate, vel_x = data

        # calculate discretized transformation matrix A_d
        A_k11 = (Delta_T ** 4 * yaw_rate ** 4) / 24 - (Delta_T ** 2 * yaw_rate ** 2) / 2 + 1
        A_k12 = (Delta_T ** 3 * yaw_rate ** 3) / 6 - Delta_T * yaw_rate
        A_k21 = Delta_T * yaw_rate - (Delta_T ** 3 * yaw_rate ** 3) / 6 
        A_k22 = (Delta_T ** 4 * yaw_rate ** 4) / 24 - (Delta_T ** 2 * yaw_rate ** 2) / 2 + 1
        A_kd = np.array([[A_k11, A_k12], [A_k21, A_k22]])

        # calculate discretized B_d matrix
        B_k11 = np.sin(Delta_T * yaw_rate) / yaw_rate
        B_k12 = np.cos(Delta_T * yaw_rate) / yaw_rate
        B_k21 = -np.cos(Delta_T * yaw_rate) / yaw_rate
        B_k22 = np.sin(Delta_T * yaw_rate) / yaw_rate
        B_kd = np.array([[B_k11, B_k12], [B_k21, B_k22]])

        # calculate measurement matrix
        C_kd = np.array([[1, 0], [0, 1]])

        # set measurement matrix
        y_kd = np.array([[vel_x], [self.v_y]])

        # set control vector
        u_kd = np.array([[acc_x], [acc_y]])

        # print('A_kd={}\nB_kd={}\nC_kd={}\ny_kd={}\nu_kd'.format(A_kd, B_kd, C_kd, y_kd, u_kd))

        return A_kd, B_kd, C_kd, y_kd, u_kd

    def get_dynamics_param(self, processed_data, dt):
        """
        calculate discretized dynamics system matrix, which be used in EKF
        :param processed_data:input data, acc_x, acc_y, steering_angle, yaw_rate, vel_x
        :param dt:time interval
        :return:system matrix
        """
        # get data from topic
        acc_x, acc_y, steering_angle, yaw_rate, vel_x = processed_data

        # prevent vel_x equal to 0
        if vel_x < 0.01:
            vel_x = 0.01

        # calculate transformation matrix
        A_d = np.array([[(-self.C_f - self.C_r) / (self.m_v * vel_x),
                         (-vel_x - (self.L_f * self.C_f - self.L_r * self.C_r) / (self.m_v * vel_x))],
                        [(-self.L_f * self.C_f + self.L_r * self.C_r) / (self.I_z * vel_x),
                         (-(self.L_f ** 2) * self.C_f - (self.L_r ** 2) * self.C_r) / (self.I_z * vel_x)]])
        C_d = np.array([[(-self.C_f - self.C_r) / (self.m_v * vel_x),
                         -(self.L_f * self.C_f - self.L_r * self.C_r) / (self.m_v * vel_x)],
                        [0, 1]])
        D_d = np.array([[self.C_f / self.m_v], [0]])

        # get the measurement output vector
        y_d = np.array([[acc_y, yaw_rate]]).T - D_d * steering_angle

        # get discretized linear continuous dynamics model
        A_dd = np.eye(2) + A_d * dt + 0.5 * A_d ** 2 * dt ** 2 + 1 / 6 * A_d ** 3 * dt ** 3 + 1 / 24 * A_d ** 4 * dt ** 4
        B_dd_0 = (207319096137250381803524174900331*vel_x*np.exp((512*dt*((1122942022011136646878644803407220262924012636719516516972657 - 79924283481264903055262051827173378911198266909432741888*vel_x**2)**(1/2) - 11075153569676092564949274424089))/(23035455126361153533724908322259*vel_x))*(179903239029935360*(1122942022011136646878644803407220262924012636719516516972657 - 79924283481264903055262051827173378911198266909432741888*vel_x**2)**(1/2) - 9340524050321149179789466627555870280588317023*vel_x**2 + 191136591265856894831378235498059356636140569856))/(18446744073709551616*(1122942022011136646878644803407220262924012636719516516972657 - 79924283481264903055262051827173378911198266909432741888*vel_x**2)**(1/2)*((1122942022011136646878644803407220262924012636719516516972657 - 79924283481264903055262051827173378911198266909432741888*vel_x**2)**(1/2) - 11075153569676092564949274424089)) - (207319096137250381803524174900331*vel_x*np.exp(-(512*dt*((1122942022011136646878644803407220262924012636719516516972657 - 79924283481264903055262051827173378911198266909432741888*vel_x**2)**(1/2) + 11075153569676092564949274424089))/(23035455126361153533724908322259*vel_x))*(179903239029935360*(1122942022011136646878644803407220262924012636719516516972657 - 79924283481264903055262051827173378911198266909432741888*vel_x**2)**(1/2) + 9340524050321149179789466627555870280588317023*vel_x**2 - 191136591265856894831378235498059356636140569856))/(18446744073709551616*(1122942022011136646878644803407220262924012636719516516972657 - 79924283481264903055262051827173378911198266909432741888*vel_x**2)**(1/2)*((1122942022011136646878644803407220262924012636719516516972657 - 79924283481264903055262051827173378911198266909432741888*vel_x**2)**(1/2) + 11075153569676092564949274424089))
        B_dd_1 = (207319096137250381803524174900331*vel_x*np.exp((512*dt*((1122942022011136646878644803407220262924012636719516516972657 - 79924283481264903055262051827173378911198266909432741888*vel_x**2)**(1/2) - 11075153569676092564949274424089))/(23035455126361153533724908322259*vel_x))*(405484675648197*(1122942022011136646878644803407220262924012636719516516972657 - 79924283481264903055262051827173378911198266909432741888*vel_x**2)**(1/2) - 429062881814690501191506705002666862454131773))/(36028797018963968*(1122942022011136646878644803407220262924012636719516516972657 - 79924283481264903055262051827173378911198266909432741888*vel_x**2)**(1/2)*((1122942022011136646878644803407220262924012636719516516972657 - 79924283481264903055262051827173378911198266909432741888*vel_x**2)**(1/2) - 11075153569676092564949274424089)) - (207319096137250381803524174900331*vel_x*np.exp(-(512*dt*((1122942022011136646878644803407220262924012636719516516972657 - 79924283481264903055262051827173378911198266909432741888*vel_x**2)**(1/2) + 11075153569676092564949274424089))/(23035455126361153533724908322259*vel_x))*(405484675648197*(1122942022011136646878644803407220262924012636719516516972657 - 79924283481264903055262051827173378911198266909432741888*vel_x**2)**(1/2) + 429062881814690501191506705002666862454131773))/(36028797018963968*(1122942022011136646878644803407220262924012636719516516972657 - 79924283481264903055262051827173378911198266909432741888*vel_x**2)**(1/2)*((1122942022011136646878644803407220262924012636719516516972657 - 79924283481264903055262051827173378911198266909432741888*vel_x**2)**(1/2) + 11075153569676092564949274424089))
        B_dd = np.array([[B_dd_0], [B_dd_1]])

        # control vector
        u_d = steering_angle

        # print('A_dd={}\nB_dd={}\nC_d={}\nD_d={}\ny_d={}\nu_d={}'.format(A_dd, B_dd, C_d, D_d, y_d, u_d))

        return A_dd, B_dd, C_d, D_d, y_d, u_d

    def EstCallback(self, processed_data):
        # get data from topic
        accx, accy, yaw_rate, self.v_x, deltav = processed_data.data
        data = np.array([accx, accy, deltav, yaw_rate, self.v_x])
        Delta_T = 0.1

        # reset model when yaw rate is too small
        if np.abs(yaw_rate) < 0.01:
            Beta = self.last_beta
            self.state_estimate_kd = np.array([[self.v_x],
                                               [0]])
            self.state_estimate_dd = np.array([[0],
                                               [0]])
            self.P_kd = np.array([[0.1, 0], [0, 0.1]])
            self.P_dd = np.array([[0.1, 0], [0, 0.1]])
            self.pub1.publish(Beta)


        else:
            # get kinematics model param
            A_kd, B_kd, C_kd, y_kd, u_kd = self.get_kinematics_param(data, Delta_T)

            # EKF update function of kinematics model
            optimal_state_estimate_kd, self.P_kd = self.ekf_update_k(A_kd, B_kd, C_kd, y_kd, self.state_estimate_kd, u_kd, self.P_kd)
            self.state_estimate_dd[0, 0] = optimal_state_estimate_kd[1]

            # get system matrix of  dynamics model
            A_dd, B_dd, C_dd, D_dd, y_dd, u_dd = self.get_dynamics_param(data, Delta_T)

            # EKF update function of dynamics model
            optimal_state_estimate_dd, self.P_dd = self.ekf_update_d(A_dd, B_dd, C_dd, y_dd, self.state_estimate_dd, u_dd, self.P_dd)

            # update v_y for kinematics model in next time step
            self.v_y = optimal_state_estimate_dd[0, 0]

            # state update
            self.state_estimate_kd[0, 0] = optimal_state_estimate_kd[0, 0]
            self.state_estimate_kd[1, 0] = optimal_state_estimate_dd[0, 0]
            self.state_estimate_dd[1, 0] = optimal_state_estimate_dd[1, 0]
            self.v_y = optimal_state_estimate_dd[0, 0]

            # calculate side slip angle
            Beta = np.arctan(optimal_state_estimate_dd[0, 0] / optimal_state_estimate_kd[0, 0])

            # delete over range noise
            if abs(Beta) > 0.6:
                Beta = self.last_beta

            # record values
            self.beta.append(Beta)
            self.pub1.publish(Beta)
            rospy.loginfo('Beta = {}'.format(Beta))
            self.last_beta = Beta

        # calculate position and trajectory
        self.psi = self.psi + data[3] * Delta_T
        self.x0 = self.x0 + Delta_T * data[4] * np.cos(Beta + self.psi)
        self.y0 = self.y0 + Delta_T * data[4] * np.sin(Beta + self.psi)
        self.x_buffer.append(self.x0)
        self.y_buffer.append(self.y0)

        # store test data as txt file for Matlab
        with open("x0.txt", "a") as f:
            f.write('\r')
            f.write(str(self.x0))
        with open("y0.txt", "a") as f:
            f.write('\r')
            f.write(str(self.y0))
        with open("beta.txt", "a") as f:
            f.write('\r')
            f.write(str(Beta))
        with open("vy.txt", "a") as f:
            f.write('\r')
            f.write(str(self.v_y))
        with open("deltav.txt", "a") as f:
            f.write('\r')
            f.write(str(deltav))


if __name__ == '__main__':
    try:
        Side_Slip_Angle_Estimation()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

