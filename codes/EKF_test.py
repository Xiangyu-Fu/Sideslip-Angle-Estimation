import numpy as np
# Supress scientific notation when printing NumPy arrays
np.set_printoptions(precision=3, suppress=True)

# A matrix
A_k_minus_1 = np.array([[1.0, 0, 0],
                        [0, 1.0, 0],
                        [0, 0, 1.0]])

# Noise applied to the forward kinematics (calculation
process_noise_v_k_minus_1 = np.array([0.01, 0.01, 0.003])

# State model noise covariance matrix Q_k
Q_k = np.array([[1.0, 0, 0],
                [0, 1.0, 0],
                [0, 0, 1.0]])

# Measurement matrix H_k
H_k = np.array([[1.0, 0, 0],
                [0, 1.0, 0],
                [0, 0, 1.0]])

# Sensor measurement noise covariance matrix R_k
R_k = np.array([[1.0, 0, 0],
                [0, 1.0, 0],
                [0, 0, 1.0]])

# Sensor noise.
sensor_noise_w_k = np.array([0.07, 0.07, 0.04])


def getB(yaw, deltak):
    B = np.array([[np.cos(yaw) * deltak, 0],
                  [np.sin(yaw) * deltak, 0],
                  [0, deltak]])
    return B


def ekf(z_k_observation_vector, state_estimate_k_minus_1,
        control_vector_k_minus_1, P_k_minus_1, dk):
    """
    Extended Kalman Filter. Fuses noisy sensor measurement to
    create an optimal estimate of the state of the robotic system.

    INPUT
        :param z_k_observation_vector The observation from the Odometry
            3x1 NumPy Array [x,y,yaw] in the global reference frame
            in [meters,meters,radians].
        :param state_estimate_k_minus_1 The state estimate at time k-1
            3x1 NumPy Array [x,y,yaw] in the global reference frame
            in [meters,meters,radians].
        :param control_vector_k_minus_1 The control vector applied at time k-1
            3x1 NumPy Array [v,v,yaw rate] in the global reference frame
            in [meters per second,meters per second,radians per second].
        :param P_k_minus_1 The state covariance matrix estimate at time k-1
            3x3 NumPy Array
        :param dk Time interval in seconds

    OUTPUT
        :return state_estimate_k near-optimal state estimate at time k
            3x1 NumPy Array ---> [meters,meters,radians]
        :return P_k state covariance_estimate for time k
            3x3 NumPy Array
    """
    ######################### Predict #############################
    # Predict the state estimate at time k based on the state
    # estimate at time k-1 and the control input applied at time k-1.
    state_estimate_k = A_k_minus_1 @ state_estimate_k_minus_1 \
                       + (getB(state_estimate_k_minus_1[2], dk)) @ control_vector_k_minus_1 \
                       + process_noise_v_k_minus_1

    print(f'State Estimate Before EKF={state_estimate_k}')

    # Predict the state covariance estimate based on the previous
    # covariance and some noise
    P_k = A_k_minus_1 @ P_k_minus_1 @ A_k_minus_1.T + Q_k

    ################### Update (Correct) ##########################
    # Calculate the difference between the actual sensor measurements
    # at time k minus what the measurement model predicted
    # the sensor measurements would be for the current timestep k.
    measurement_residual_y_k = z_k_observation_vector - ((H_k @ state_estimate_k) + sensor_noise_w_k)

    print(f'Observation={z_k_observation_vector}')

    # Calculate the measurement residual covariance
    S_k = H_k @ P_k @ H_k.T + R_k

    # Calculate the near-optimal Kalman gain
    # We use pseudoinverse since some of the matrices might be
    # non-square or singular.
    K_k = P_k @ H_k.T @ np.linalg.pinv(S_k)

    print(measurement_residual_y_k.shape)
    # Calculate an updated state estimate for time k
    state_estimate_k = state_estimate_k + (K_k @ measurement_residual_y_k)

    # Update the state covariance estimate for time k
    P_k = P_k - (K_k @ H_k @ P_k)

    # Print the best (near-optimal) estimate of the current state of the robot
    print(f'State Estimate After EKF={state_estimate_k}')

    # Return the updated state and covariance estimates
    return state_estimate_k, P_k


def main():
    # We start at time k=1
    k = 1

    # Time interval in seconds
    dk = 1

    # Create a list of sensor observations at successive timesteps
    # Each list within z_k is an observation vector.
    z_k = np.array([[4.721, 0.143, 0.006],  # k=1
                    [9.353, 0.284, 0.007],  # k=2
                    [14.773, 0.422, 0.009],  # k=3
                    [18.246, 0.555, 0.011],  # k=4
                    [22.609, 0.715, 0.012]])  # k=5

    # The estimated state vector at time k-1 in the global reference frame.
    # [x_k_minus_1, y_k_minus_1, yaw_k_minus_1]
    # [meters, meters, radians]
    state_estimate_k_minus_1 = np.array([0.0, 0.0, 0.0])

    # The control input vector at time k-1 in the global reference frame.
    # [v, yaw_rate]
    # [meters/second, radians/second]
    # In the literature, this is commonly u.
    # Because there is no angular velocity and the robot begins at the
    # origin with a 0 radians yaw angle, this robot is traveling along
    # the positive x-axis in the global reference frame.
    control_vector_k_minus_1 = np.array([4.5, 0.0])

    P_k_minus_1 = np.array([[0.1, 0, 0],
                            [0, 0.1, 0],
                            [0, 0, 0.1]])


    for k, obs_vector_z_k in enumerate(z_k, start=1):
        # Print the current timestep
        print(f'Timestep k={k}')

        # Run the Extended Kalman Filter and store the
        # near-optimal state and covariance estimates
        optimal_state_estimate_k, covariance_estimate_k = ekf(
            obs_vector_z_k,  # Most recent sensor measurement
            state_estimate_k_minus_1,  # Our most recent estimate of the state
            control_vector_k_minus_1,  # Our most recent control input
            P_k_minus_1,  # Our most recent state covariance matrix
            dk)  # Time interval

        # Get ready for the next timestep by updating the variable values
        state_estimate_k_minus_1 = optimal_state_estimate_k
        P_k_minus_1 = covariance_estimate_k

        # Print a blank line
        print()


main()