#!/usr/bin/env python

import math
import rospy
import numpy as np
from std_msgs.msg import Float64, Bool, Float64MultiArray


class SSAEstimation(object):
    def __init__(self):
        # Set Constant of vehicle
        self.process_noise_v_k_minus_1 = np.array([[0.01], [0.01]])

        self.sensor_noise_w_k = np.array([[0.11, 0.11]]).T

        self.Q_k = np.array([[1.0, 0],
                        [0, 1.0]])

        self.R_k = np.array([[1.0, 0],
                        [0, 1.0]])

        self.C_f = 160776.0
        self.C_r = 254100.0
        self.L_f = 1.505
        self.L_r = 1.5
        self.m_v = 2300.0
        self.I_z = 4400.0

        self.sub1 = rospy.Subscriber("/catvehicle/processed_data", Float64MultiArray, self.EstCallback)

    def EKFupdate(self, A_k, B_k, H_k, z_k_observation_vector, state_estimate_k_minus_1,
                  control_vector_k_minus_1, P_k_minus_1, dk, D_d=0):
        '''
        EKF update
        :param A_k_minus_1: A_d or A_k
        :param B_k_minus_1: B_d or B_k
        :param H_k: C_d or C_k
        :param z_k_observation_vector: y_d
        :param state_estimate_k_minus_1: x_d
        :param control_vector_k_minus_1: u_d
        :param P_k_minus_1: P_k-1
        :param dk: time interval
        :param D_d: D_d
        :return:state estimation, covariance
        '''

        # Predicted state estimate
        state_estimate_k = np.matmul(A_k, state_estimate_k_minus_1) + (
                    B_k * control_vector_k_minus_1) + self.process_noise_v_k_minus_1
        print('State Estimate Before EKF={}'.format(state_estimate_k.T))

        # Predicted covariance estmate
        P_k = np.matmul(np.matmul(A_k, P_k_minus_1), A_k.T)

        # Measurement residual
        measurement_residual_y_k = z_k_observation_vector - (
                    ((np.matmul(H_k, state_estimate_k)) + (D_d * control_vector_k_minus_1)) + self.sensor_noise_w_k)

        print('Observation={}'.format(z_k_observation_vector.T))

        # Residual covariance
        S_k = np.matmul(np.matmul(H_k, P_k), H_k.T)

        # Kalman gain
        K_k = np.matmul(np.matmul(P_k, H_k.T), np.linalg.pinv(S_k))

        # update state estimate
        state_estimate_k = state_estimate_k + (np.matmul(K_k, measurement_residual_y_k))

        # update covariance of state estimate
        P_k = P_k - (np.matmul(np.matmul(K_k, H_k), P_k))

        print('State Estimate After EKF={}\n'.format(state_estimate_k.T))

        return state_estimate_k, P_k


    def GetDM(self, processed_data, dk, v_y):
        # get data from topic
        acc_x, acc_y, yaw_rate, vel_x, steering_angle = processed_data

        # calculate transformation matrix
        A_d = np.array([[(-self.C_f - self.C_r) / (self.m_v * vel_x), (-vel_x - (self.L_f * self.C_f - self.L_r * self.C_r) / (self.m_v * vel_x))],
                        [(-self.L_f * self.C_f + self.L_r * self.C_r) / (self.I_z * vel_x),
                         (-(self.L_f ** 2) * self.C_f - (self.L_r ** 2) * self.C_r) / (self.I_z * vel_x)]])
        B_d = np.array([[self.C_f / self.m_v],
                        [self.L_f * self.C_f / self.I_z]])
        C_d = np.array([[(-self.C_f - self.C_r) / (self.m_v * vel_x), -((self.L_f * self.C_f - self.L_r * self.C_r) / (self.m_v * vel_x))],
                        [0, 1]])
        D_d = np.array([[self.C_f / self.m_v, 0]]).T

        # get the measurement output vector
        y_d = np.array([[acc_y, yaw_rate]]).T

        # get discretized linear continuous model
        A_d = (A_d * dk + np.eye(2))
        B_d = B_d * dk

        # get velocity in y direction by using kinetic model
        v_y = -yaw_rate * dk * vel_x + v_y + dk * acc_y

        # get the state vector
        x_d = np.array([[v_y, yaw_rate]]).T

        return A_d, B_d, C_d, D_d, y_d, steering_angle, x_d, vel_x

    def EstCallback(self, processed_data):
        data = processed_data.data
        dk = 0.0001
        P_k_minus_1 = np.array([[0.1, 0], [0, 0.1]])
        vel_y = 0
        for k, obs_vector_z_k in enumerate(data, start=1):
            # get parameters for model
            A_d, B_d, C_d, D_d, y_d, u_d, x_d, vel_x = self.GetDM(obs_vector_z_k, dk, vel_y)

            # EKF update
            optimal_state_estimate_k, P_k = self.EKFupdate(A_d, B_d, C_d, y_d, x_d, u_d, P_k_minus_1, dk, D_d=D_d)

            # update values
            x_d = optimal_state_estimate_k
            vel_y = x_d[0, 0]
            Beta = np.arctan(vel_y / vel_x)
            rospy.loginfo('Beta = {} degree'.format(Beta/np.pi*180))
            P_k_minus_1 = P_k


if __name__ == '__main__':
    try:
        SSAEstimation()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


