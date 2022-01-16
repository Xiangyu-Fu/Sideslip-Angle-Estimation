import mat73
import numpy as np
import matplotlib.pyplot as plt

accx_raw = mat73.loadmat('accx.mat')
accy_raw = mat73.loadmat('accy.mat')
deltav_raw = mat73.loadmat('deltav.mat')
psi_raw = mat73.loadmat('psi.mat')
vx_raw = mat73.loadmat('vx.mat')

T = accx_raw.accx[0]
accx = accx_raw.accx[1]
accy = accy_raw.accy[1]
deltav = deltav_raw.deltav[1]
yaw_rate = psi_raw.psi[1]
vx = vx_raw.vx[1]

# Plot
fig = plt.figure(1)
fig.set_size_inches(10, 5)
plt.subplot(121)
plt.plot(T, accy)  # max acceleration in y = 1 m/s^2
plt.plot(T, deltav)  # max delta_v = 0.035
plt.plot(T, yaw_rate)  # max yaw rate = 0.1
plt.legend(labels=['accy', 'deltav', 'yaw_rate'], loc='best')

plt.subplot(122)
plt.plot(T, accx)  # max acceleration in x = 0
plt.plot(T, vx)  # max longitudinal velocity =10 m/s
plt.legend(labels=['accx', 'vx'], loc='best')
plt.show()


