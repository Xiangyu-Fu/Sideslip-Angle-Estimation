import numpy as np

# Set Constant of vehicle
process_noise_v_k_minus_1 = np.array([[0.01], [0.01]])

sensor_noise_w_k = np.array([[0.11, 0.11]]).T

Q_k = np.array([[1.0, 0],
                [0, 1.0]])

R_k = np.array([[1.0, 0],
                [0, 1.0]])

C_f = 100000.0
C_r = 100000.0
L_f = 1.3
L_r = 1.45
m_v = 1450.0
I_z = 1920.0


def EKFupdate_D(A_k, B_k, H_k, z_k_observation_vector, state_estimate_k_minus_1,
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
    state_estimate_k = np.matmul(A_k, state_estimate_k_minus_1) + (B_k * control_vector_k_minus_1) + process_noise_v_k_minus_1
    # print(f'State Estimate Before EKF={state_estimate_k.T}')

    # Predicted covariance estmate
    P_k = np.matmul(np.matmul(A_k, P_k_minus_1), A_k.T) + Q_k

    # Measurement residual
    measurement_residual_y_k = z_k_observation_vector - (((np.matmul(H_k, state_estimate_k)) + (D_d * control_vector_k_minus_1)) + sensor_noise_w_k)



    # Residual covariance
    S_k = np.matmul(np.matmul(H_k, P_k), H_k.T)


    # Kalman gain
    K_k = np.matmul(np.matmul(P_k, H_k.T), np.linalg.pinv(S_k))
    print(f'K_k={K_k}')

    # update state estimate
    state_estimate_k = state_estimate_k + (np.matmul(K_k, measurement_residual_y_k))

    # update covariance of state estimate
    P_k = P_k - (K_k @ H_k @ P_k)

    # print(f'State Estimate After EKF={state_estimate_k.T}\n\n')

    return state_estimate_k, P_k


def GetDM(processed_data, dk, v_y):
    # get data from topic
    acc_x, acc_y, yaw_rate, vel_x, steering_angle = processed_data

    # calculate transformation matrix
    A_d = np.array([[(-C_f - C_r)/(m_v*vel_x), (-vel_x - (L_f*C_f-L_r*C_r)/(m_v*vel_x))],
                    [(-L_f*C_f+L_r*C_r)/(I_z*vel_x), (-(L_f**2)*C_f-(L_r**2)*C_r)/(I_z*vel_x)]])
    B_d = np.array([[C_f/m_v],
                    [L_f*C_f/I_z]])
    C_d = np.array([[(-C_f-C_r)/(m_v*vel_x), -((L_f*C_f-L_r*C_r)/(m_v*vel_x))],
                    [0, 1]])
    D_d = np.array([[C_f/m_v, 0]]).T

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


def main():
    data = [[0.01, 0.01, 0.1, 0.1, 0.1],
            [0.01, 0.01, 0.1, 0.2, 0.1],
            [0.01, 0.01, 0.1, 0.3, 0.1],
            [0.01, 0.01, 0.1, 0.4, 0.1],
            [0.01, 0.01, 0.1, 0.5, 0.1],
            [0.01, 0.01, 0.1, 0.6, 0.0],
            [0.01, 0.01, 0.0, 0.7, 0.0],
            [0.01, 0.01, 0.0, 0.8, 0.0],
            [0.01, 0.01, 0.0, 0.8, 0.0],
            [0.01, 0.01, 0.0, 0.8, 0.0],
            [0.01, 0.01, 0.1, 0.9, 0.1],
            [0.01, 0.01, 0.1, 0.9, 0.1],
            [0.01, 0.01, 0.1, 0.9, 0.1],
            [0.01, 0.01, 0.1, 0.99, 0.1],
            [0.01, 0.01, 0.1, 0.99, 0.1],
            [0.01, 0.01, 0.1, 0.99, 0.0],
            [0.01, 0.01, 0.0, 0.99, 0.0],
            [0.01, 0.01, 0.0, 0.99, 0.0],
            [0.01, 0.01, 0.0, 0.99, 0.0],
            [0.01, 0.01, 0.0, 0.99, 0.0],
            [0.01, 0.01, 0.1, 1.0, 0.1],
            [0.01, 0.01, 0.1, 1.2, 0.1],
            [0.01, 0.01, 0.1, 1.3, 0.1],
            [0.01, 0.01, 0.1, 1.4, 0.1],
            [0.01, 0.01, 0.1, 1.5, 0.1],
            [0.01, 0.01, 0.1, 1.6, 0.0],
            [0.01, 0.01, 0.0, 1.6, 0.0],
            [0.01, 0.01, 0.0, 1.6, 0.0],
            [0.01, 0.01, 0.0, 1.6, 0.0],
            [0.01, 0.01, 0.0, 1.6, 0.0]
            ]
    dk = 0.00001
    P_k_minus_1 = np.array([[0.1, 0], [0, 0.1]])
    vel_y = 0
    for k, obs_vector_z_k in enumerate(data, start=1):
        # get parameters for model
        A_d, B_d, C_d, D_d, y_d, u_d, x_d, vel_x = GetDM(obs_vector_z_k, dk, vel_y)

        # EKF update
        optimal_state_estimate_k, P_k = EKFupdate_D(A_d, B_d, C_d, y_d, x_d, u_d, P_k_minus_1, dk, D_d=D_d)

        # update values
        x_d = optimal_state_estimate_k
        vel_y = x_d[0, 0]
        Beta = np.arctan(vel_y/vel_x)
        print('Beta = {} degree'.format(Beta/np.pi*180))
        P_k_minus_1 = P_k
        print(P_k)


if __name__ == '__main__':
    main()

    '''try:
        DataProcess()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass'''
