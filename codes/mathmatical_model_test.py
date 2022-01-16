import numpy as np
import mat73
import matplotlib.pyplot as plt


class Side_Slip_Angle_Estimation(object):
    def __init__(self):
        # Set process and sensor noise for EKF of dynamics model
        self.process_noise_v_kd = np.array([[0.1],
                                            [0.1]])

        self.sensor_noise_w_kd = np.array([[0.25],
                                           [0.25]])

        self.Q_kd = np.array([[1.5, 0],
                              [0, 1.5]])

        self.R_kd = np.array([[0.01, 0],
                              [0, 0.01]])

        # Set process and sensor noise for EKF of kinematics model
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

        self.vx = 0

        self.EstCallback()

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
        state_estimate_k = np.matmul(A_k, state_estimate_k) + np.matmul(B_k, control_vector_k) + self.process_noise_v_kd
        # print(f'Kinematic State Estimate Before EKF={state_estimate_k.T}')

        # Predicted covariance estimate
        P_k = np.matmul(np.matmul(A_k, P_k), A_k.T) + self.Q_kd
        # print('P_kd={}'.format(P_k))

        # Measurement residual
        measurement_residual_y_k = z_k_observation_vector - (
        ((np.matmul(C_k, state_estimate_k)) + self.sensor_noise_w_kd))

        # print(f'Kinematic Observation={z_k_observation_vector.T}')

        # Residual covariance
        S_k = np.matmul(np.matmul(C_k, P_k), C_k.T) + self.R_kd

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
        :return:state estimation [v_y, yaw_rate]^T , covariance
        """

        # Predicted state estimate
        state_estimate_k = np.matmul(A_k, state_estimate_k) + (B_k * control_vector_k) + self.process_noise_v_kk
        # print(f'Dynamic State Estimate Before EKF={state_estimate_k.T}')

        # Predicted covariance estmate
        P_k = np.matmul(np.matmul(A_k, P_k), A_k.T) + self.Q_kk

        # Measurement residual
        measurement_residual_y_k = z_k_observation_vector - ((np.matmul(C_k, state_estimate_k)) + self.sensor_noise_w_kk)

        # print(f'Dynamic Observation={z_k_observation_vector.T}')

        # Residual covariance
        S_k = np.matmul(np.matmul(C_k, P_k), C_k.T) + self.R_kk

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
        # calculate discretized transformation matrix A_d
        A_k11 = (Delta_T ** 4 * data[3] ** 4) / 24 - (Delta_T ** 2 * data[3] ** 2) / 2 + 1
        A_k12 = Delta_T * data[3] - (Delta_T ** 3 * data[3] ** 3) / 6
        A_k21 = (Delta_T ** 3 * data[3] ** 3) / 6 - Delta_T * data[3]
        A_k22 = (Delta_T ** 4 * data[3] ** 4) / 24 - (Delta_T ** 2 * data[3] ** 2) / 2 + 1
        A_kd = np.array([[A_k11, A_k12], [A_k21, A_k22]])

        # calculate discretized B_d matrix
        B_k11 = np.sin(Delta_T*data[3]) / data[3]
        B_k12 = -np.cos(Delta_T * data[3])/ data[3]
        B_k21 = np.cos(Delta_T * data[3])/ data[3]
        B_k22 = np.sin(Delta_T*data[3]) / data[3]
        B_kd = np.array([[B_k11, B_k12], [B_k21, B_k22]])

        # calculate measurement matrix
        C_kd = np.array([[1, 0], [0, 0]])

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
        A_dd = np.eye(2) + A_d * dk + 0.5 * A_d ** 2 * dk ** 2 + 1 / 6 * A_d ** 3 * dk ** 3 \
               + 1 / 24 * A_d ** 4 * dk ** 4
        B_dd = np.array([[(23035455126361153533724908322259*vel_x*np.exp((16*dk*(2**(1/2)*(581199167020588720261782802044410150526998028077638549906112128- 3026417631733652526967727289269266641125392407461702372376995*vel_x**2)**(1/2) - 349881927859473263057738141271392))/(23035455126361153533724908322259*vel_x))*(2074896099002280506966796787560292637885429721472*2**(1/2) + 84168962363051684924642293892625849739094337069*2**(1/2)*vel_x**2 + 101321061279085136*(581199167020588720261782802044410150526998028077638549906112128 - 3026417631733652526967727289269266641125392407461702372376995*vel_x**2)**(1/2)))/(36028797018963968*(2**(1/2)*(581199167020588720261782802044410150526998028077638549906112128 - 3026417631733652526967727289269266641125392407461702372376995*vel_x**2)**(1/2) - 349881927859473263057738141271392)*(581199167020588720261782802044410150526998028077638549906112128 - 3026417631733652526967727289269266641125392407461702372376995*vel_x**2)**(1/2)) + (23035455126361153533724908322259*vel_x*np.exp(-(16*dk*(2**(1/2)*(581199167020588720261782802044410150526998028077638549906112128 - 3026417631733652526967727289269266641125392407461702372376995*vel_x**2)**(1/2) + 349881927859473263057738141271392))/(23035455126361153533724908322259*vel_x))*(2074896099002280506966796787560292637885429721472*2**(1/2) + 84168962363051684924642293892625849739094337069*2**(1/2)*vel_x**2 - 101321061279085136*(581199167020588720261782802044410150526998028077638549906112128 - 3026417631733652526967727289269266641125392407461702372376995*vel_x**2)**(1/2)))/(36028797018963968*(2**(1/2)*(581199167020588720261782802044410150526998028077638549906112128 - 3026417631733652526967727289269266641125392407461702372376995*vel_x**2)**(1/2) + 349881927859473263057738141271392)*(581199167020588720261782802044410150526998028077638549906112128 - 3026417631733652526967727289269266641125392407461702372376995*vel_x**2)**(1/2))],
                [(vel_x*np.exp((16*dk*(2**(1/2)*(581199167020588720261782802044410150526998028077638549906112128 - 3026417631733652526967727289269266641125392407461702372376995*vel_x**2)**(1/2) - 349881927859473263057738141271392))/(23035455126361153533724908322259*vel_x))*(22219901467185718230362870761011599865379700791565751436178753385748361553712256*2**(1/2) + 11058189662107399764905708889310451484410369277585845054883138236040842814045*2**(1/2)*vel_x**2 - 1139276341433552854761447047062264747134088440448*(581199167020588720261782802044410150526998028077638549906112128 - 3026417631733652526967727289269266641125392407461702372376995*vel_x**2)**(1/2)))/(70368744177664*(4204187140398613534247367433760*vel_x**2 + 84221450416134952723109741797691392)*(581199167020588720261782802044410150526998028077638549906112128 - 3026417631733652526967727289269266641125392407461702372376995*vel_x**2)**(1/2)) - (vel_x*np.exp(-(16*dk*(2**(1/2)*(581199167020588720261782802044410150526998028077638549906112128 - 3026417631733652526967727289269266641125392407461702372376995*vel_x**2)**(1/2) + 349881927859473263057738141271392))/(23035455126361153533724908322259*vel_x))*(22219901467185718230362870761011599865379700791565751436178753385748361553712256*2**(1/2) + 11058189662107399764905708889310451484410369277585845054883138236040842814045*2**(1/2)*vel_x**2 + 1139276341433552854761447047062264747134088440448*(581199167020588720261782802044410150526998028077638549906112128 - 3026417631733652526967727289269266641125392407461702372376995*vel_x**2)**(1/2)))/(70368744177664*(4204187140398613534247367433760*vel_x**2 + 84221450416134952723109741797691392)*(581199167020588720261782802044410150526998028077638549906112128 - 3026417631733652526967727289269266641125392407461702372376995*vel_x**2)**(1/2))]])

        # control vector
        u_d = steering_angle

        # print('A_dd={}\nB_dd={}\nC_d={}\nD_d={}\ny_d={}\nu_d={}'.format(A_dd, B_dd, C_d, D_d, y_d, u_d))

        return A_dd, B_dd, C_d, D_d, y_d, u_d

    def EstCallback(self):
        # load exported Simulink data
        accx_raw = mat73.loadmat('accx.mat')
        accy_raw = mat73.loadmat('accy.mat')
        deltav_raw = mat73.loadmat('deltav.mat')
        psi_raw = mat73.loadmat('psi.mat')
        vx_raw = mat73.loadmat('vx.mat')
        beta_raw = mat73.loadmat('beta.mat')

        # get required data
        T = accx_raw.accx[0]
        accx = accx_raw.accx[1]
        accy = accy_raw.accy[1]
        deltav = deltav_raw.deltav[1]
        yaw_rate = psi_raw.psi[1]
        vx = vx_raw.vx[1]
        beta_from_simulink = beta_raw.beta[1]
        Delta_T = 0.01
        # initialisation of EKF
        beta = []
        vy = 0
        state_estimate_kd = np.array([[vx[0]],
                                      [vy]])
        P_kd = np.array([[0.1, 0], [0, 0.1]])
        state_estimate_dd = np.array([[vy],
                                      [yaw_rate[0]]])
        P_dd = np.array([[0.1, 0], [0, 0.1]])

        # trajectory calculation
        x0 = 0
        y0 = 0
        x_buffer = [0]
        y_buffer = [0]
        psi = 0
        v_y = []

        # Main Loop
        for data in zip(accx, accy, deltav, yaw_rate, vx):
            if np.abs(data[3]) < 0.02:
                Beta = 0
                beta.append(Beta)
                v_y.append(0)
            else:
                # get system matrix of kinematics model
                data = np.array(data) + np.random.normal(0, 0.0005)  # add Gauss noise
                A_kd, B_kd, C_kd, y_kd, u_kd = self.get_kinematics_param(data, Delta_T)

                # EKF update function of kinematics model
                optimal_state_estimate_kd, P_kd = self.ekf_update_k(A_kd, B_kd, C_kd, y_kd, state_estimate_kd, u_kd, P_kd)

                # get system matrix of  dynamics model
                A_dd, B_dd, C_dd, D_dd, y_dd, u_dd = self.get_dynamics_param(data, Delta_T)

                state_estimate_dd[0, 0] = optimal_state_estimate_kd[1]
                v_y.append(state_estimate_dd[0, 0])

                # EKF update function of dynamics model
                optimal_state_estimate_dd, P_dd = self.ekf_update_d(A_dd, B_dd, C_dd, y_dd, state_estimate_dd, u_dd, P_dd)

                # print(optimal_state_estimate_dd)

                # state update
                state_estimate_kd[0, 0] = optimal_state_estimate_kd[0, 0]
                state_estimate_kd[1, 0] = optimal_state_estimate_dd[0, 0]
                state_estimate_dd[1, 0] = optimal_state_estimate_dd[1, 0]

                # calculate side slip angle
                Beta = np.arctan(optimal_state_estimate_dd[0, 0] / optimal_state_estimate_kd[0, 0])
                beta.append(Beta)

                # calculate position and trajectory
                psi = psi + data[3] * Delta_T
                x0 = x0 + Delta_T * data[4] * np.cos(Beta + psi)
                y0 = y0 + Delta_T * data[4] * np.sin(Beta + psi)
                x_buffer.append(x0)
                y_buffer.append(y0)

        # side slip angle Plot
        fig = plt.figure(1)
        fig.set_size_inches(10, 5)
        plt.subplot(121)
        plt.title('Sideslip Angle beta')
        plt.plot(T, beta)
        plt.plot(T, beta_from_simulink, '--')
        plt.legend(labels=['beta from state observer', 'beta from simulink'], loc='best')
        plt.grid()

        # calculation error plot
        plt.subplot(122)
        plt.title('Estimation Error')
        beta_error = np.array(beta - beta_from_simulink, dtype=float)
        Error = 1 / 200 * np.sum(beta_error[400:600] / beta_from_simulink[400: 600])
        print('Error={}%'.format(Error * 100))
        plt.plot(T, beta_error)
        plt.grid()
        plt.show()


if __name__ == '__main__':
    Side_Slip_Angle_Estimation()
