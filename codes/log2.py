#!/usr/bin/env python

import math
import rospy
import numpy as np
from std_msgs.msg import Float64, Bool, Float64MultiArray
import matplotlib.pyplot as plt


class Side_Slip_Angle_Estimation(object):
    def __init__(self):
        self.node_name = 'SideSlipAngle_Estimation'
        rospy.init_node(self.node_name)
        # # Set process and sensor noise for EKF
        self.process_noise_v_k = np.array([[0.1],
                                           [0.1]])

        self.sensor_noise_w_k = np.array([[0.25],
                                          [0.25]])

        self.Q_k = np.array([[1.5, 0],
                             [0, 1.5]])

        self.R_k = np.array([[0.01, 0],
                             [0, 0.01]])

        self.process_noise_v_kk = np.array([[0.001],
                                            [0.001]])

        self.sensor_noise_w_kk = np.array([[0.0001],
                                           [0.0001]])

        self.Q_kk = np.array([[0.1, 0],
                              [0, 0.1]])

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
        :return:state estimation [v_x, v_y]^T, covariance
        """

        # Predicted state estimate
        state_estimate_k = np.matmul(A_k, state_estimate_k) + np.matmul(B_k, control_vector_k) + self.process_noise_v_kk
        # print(f'Kinematic State Estimate Before EKF={state_estimate_k.T}')

        # Predicted covariance estimate
        P_k = np.matmul(np.matmul(A_k, P_k), A_k.T) + self.Q_kk
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

        if self.v_x < 0.05:
            P_k = np.array([[0.1, 0], [0, 0.1]])

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
        :return:state estimation [v_y, yaw_rate]^T , covariance
        """

        # Predicted state estimate
        state_estimate_k = np.matmul(A_k, state_estimate_k) + (B_k * control_vector_k) + self.process_noise_v_k
        # print(f'Dynamic State Estimate Before EKF={state_estimate_k.T}')

        # Predicted covariance estmate
        P_k = np.matmul(np.matmul(A_k, P_k), A_k.T) + self.Q_k
        if self.v_x < 0.05:
            P_k = np.array([[0.1, 0], [0, 0.1]])

        # Measurement residual
        measurement_residual_y_k = z_k_observation_vector - ((np.matmul(C_k, state_estimate_k)) + self.sensor_noise_w_k)

        # print(f'Dynamic Observation={z_k_observation_vector.T}')

        # Residual covariance
        S_k = np.matmul(np.matmul(C_k, P_k), C_k.T) + self.R_k

        # Kalman gain
        K_k = np.matmul(np.matmul(P_k, C_k.T), np.linalg.pinv(S_k))

        # update state estimate
        state_estimate_k = state_estimate_k + (np.matmul(K_k, measurement_residual_y_k))

        # update covariance of state estimate
        P_k = P_k - np.matmul(np.matmul(K_k, C_k), P_k)

        if self.v_x < 0.05:
            P_k = np.array([[0.1, 0], [0, 0.1]])

        # print(f'Dynamic State Estimate After EKF={state_estimate_k.T}\n\n')

        return state_estimate_k, P_k

    def get_kinematics_param(self, data, Delta_T):
        """
        calculate discretized kinematics system matrix, which be used in EKF
        :param data:input data, acc_x, acc_y, steering_angle, yaw_rate, vel_x
        :param Delta_T: time interval
        :return: system matrix
        """
        # calculate discretized transformation matrix A_d
        A_k11 = (Delta_T ** 4 * data[3] ** 4) / 24 - (Delta_T ** 2 * data[3] ** 2) / 2 + 1
        A_k12 = Delta_T * data[3] - (Delta_T ** 3 * data[3] ** 3) / 6
        A_k21 = (Delta_T ** 3 * data[3] ** 3) / 6 - Delta_T * data[3]
        A_k22 = (Delta_T ** 4 * data[3] ** 4) / 24 - (Delta_T ** 2 * data[3] ** 2) / 2 + 1
        A_kd = np.array([[A_k11, A_k12], [A_k21, A_k22]])

        # calculate discretized B_d matrix
        B_k11 = data[3] - (Delta_T ** 2 * data[3] ** 3) / 6
        B_k12 = - (Delta_T ** 3 * data[3] ** 4) / 24 + (Delta_T * data[3] ** 2) / 2 - 1 / Delta_T
        B_k21 = 1 / Delta_T - (Delta_T * data[3] ** 2) / 2 + (Delta_T ** 3 * data[3] ** 4) / 24
        B_k22 = data[3] - (Delta_T ** 2 * data[3] ** 3) / 6
        B_kd = np.array([[B_k11, B_k12], [B_k21, B_k22]])

        # calculate measurement matrix
        C_kd = np.array([[1, 0], [0, 1]])

        # set measurement matrix
        y_kd = np.array([[data[4]], [0]])

        # set control vector
        u_kd = np.array([[data[0], data[1]]]).T

        # print('A_kd={}\nB_kd={}\nC_kd={}\ny_kd={}\nu_kd'.format(A_kd, B_kd, C_kd, y_kd, u_kd))

        return A_kd, B_kd, C_kd, y_kd, u_kd

    def get_dynamics_param(self, processed_data, dk):
        """
        calculate discretized dynamics system matrix, which be used in EKF
        :param processed_data:input data, acc_x, acc_y, steering_angle, yaw_rate, vel_x
        :param dk:time interval
        :return:system matrix
        """
        # get data from topic
        acc_x, acc_y, steering_angle, yaw_rate, vel_x = processed_data

        if vel_x < 0.05:
            vel_x = 0.05

        # calculate transformation matrix
        A_d = np.array([[(-self.C_f - self.C_r) / (self.m_v * vel_x),
                         (-vel_x - (self.L_f * self.C_f - self.L_r * self.C_r) / (self.m_v * vel_x))],
                        [(-self.L_f * self.C_f + self.L_r * self.C_r) / (self.I_z * vel_x),
                         (-(self.L_f ** 2) * self.C_f - (self.L_r ** 2) * self.C_r) / (self.I_z * vel_x)]])
        B_d = np.array([[self.C_f / self.m_v],
                        [self.L_f * self.C_f / self.I_z]])
        C_d = np.array([[(-self.C_f - self.C_r) / (self.m_v * vel_x),
                         -(self.L_f * self.C_f - self.L_r * self.C_r) / (self.m_v * vel_x)],
                        [0, 1]])
        D_d = np.array([[self.C_f / self.m_v, 0]]).T

        # get the measurement output vector
        y_d = np.array([[acc_y, yaw_rate]]).T - D_d * steering_angle

        # get discretized linear continuous dynamics model
        A_dd = np.eye(
            2) + A_d * dk + 0.5 * A_d ** 2 * dk ** 2 + 1 / 6 * A_d ** 3 * dk ** 3 + 1 / 24 * A_d ** 4 * dk ** 4
        B_dd = np.eye(2) * B_d * dk + A_d * B_d * (1 / 2) * dk ** 2 + (1 / 2) * A_d ** 2 * B_d * (1 / 3) * dk ** 3 + (
                    1 / 6) * A_d ** 3 * B_d * (1 / 4) * dk ** 4 + (1 / 24) * A_d ** 4 * B_d * (1 / 5) * dk ** 5

        # control vector
        u_d = steering_angle

        # print('A_dd={}\nB_dd={}\nC_d={}\nD_d={}\ny_d={}\nu_d={}'.format(A_dd, B_dd, C_d, D_d, y_d, u_d))

        return A_dd, B_dd, C_d, D_d, y_d, u_d

    def EstCallback(self, processed_data):
        # get data from topic
        accx, accy, yaw_rate, self.v_x, deltav = processed_data.data
        # rospy.loginfo('vx = {}, acc=[{}, {}]'.format(yaw_rate, accx, accx))
        data = np.array([accx, accy, deltav, yaw_rate, self.v_x])
        Delta_T = 0.1

        # time interval
        time = rospy.get_time()
        Delta_T = time - self.last_time
        self.last_time = time

        if np.abs(yaw_rate) < 0.02:
            Beta = 0
            # rospy.loginfo('Beta = {}'.format(Beta))
            self.beta.append(Beta)
            self.state_estimate_kd = np.array([[self.v_x],
                                               [0]])
            self.state_estimate_dd = np.array([[0],
                                               [0]])
            self.P_kd = np.array([[0.1, 0], [0, 0.1]])
            self.P_dd = np.array([[0.1, 0], [0, 0.1]])


        else:
            # get kinematics model param
            A_kd, B_kd, C_kd, y_kd, u_kd = self.get_kinematics_param(data, Delta_T)

            # EKF update function of kinematics model
            optimal_state_estimate_kd, self.P_kd = self.ekf_update_k(A_kd, B_kd, C_kd, y_kd, self.state_estimate_kd,
                                                                     u_kd, self.P_kd)

            # get system matrix of  dynamics model
            A_dd, B_dd, C_dd, D_dd, y_dd, u_dd = self.get_dynamics_param(data, Delta_T)

            self.state_estimate_dd[0, 0] = optimal_state_estimate_kd[1]

            # EKF update function of dynamics model
            optimal_state_estimate_dd, self.P_dd = self.ekf_update_d(A_dd, B_dd, C_dd, y_dd, self.state_estimate_dd,
                                                                     u_dd, self.P_dd)

            # print(optimal_state_estimate_dd)

            # state update
            self.state_estimate_kd[0, 0] = optimal_state_estimate_kd[0, 0]
            self.state_estimate_kd[1, 0] = optimal_state_estimate_dd[0, 0]
            self.state_estimate_dd[1, 0] = optimal_state_estimate_dd[1, 0]
            self.v_y = optimal_state_estimate_dd[0, 0]

            # calculate side slip angle
            Beta = np.arctan(optimal_state_estimate_dd[0, 0] / optimal_state_estimate_kd[0, 0])
            if abs(Beta) > 0.6:
                Beta = self.last_beta
            # rospy.loginfo('accx = {},accy = {},yaw_rate = {},vx = {},deltav = {}'.format(accx, accy, yaw_rate, vx, deltav))
            # rospy.loginfo('v_y = {}, Beta = {}, delta_v = {}'.format(self.v_y, Beta, deltav))
            self.beta.append(Beta)
            rospy.loginfo('Beta = {}'.format(Beta))
            self.last_beta = Beta

        # calculate position and trajectory
        self.psi = self.psi + data[3] * Delta_T
        self.x0 = self.x0 + Delta_T * data[4] * np.cos(Beta + self.psi)
        self.y0 = self.y0 + Delta_T * data[4] * np.sin(Beta + self.psi)
        self.x_buffer.append(self.x0)
        self.y_buffer.append(self.y0)
        # rospy.loginfo('vx = {}, pos=[{}, {}]'.format(vx,self.x0,self.x0))

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
        rospy.loginfo('---start side slip angle estimation node---')
        Side_Slip_Angle_Estimation()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
