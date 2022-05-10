import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
import numpy as np
import scipy.io as spio
mat = spio.loadmat('training.mat', squeeze_me=True)
d = mat['d']
Index = mat['Index']
Class = mat['Class']
print(Class)
print(Index)
sf = 25000
data = d
dur_sec = data.shape[0]/sf
#create time vector
time = np.linspace(0, dur_sec, data.shape[0])
#plot first second of the data
fig, ax = plt.subplots(figsize=(15,5))
ax.plot(time[0:sf], data[0:sf])
ax.set_title('Broadband; sampling frequency: {}Hz'.format(sf), fontsize = 23)
ax.set_xlim(0,time[sf])
ax.set_xlabel('time(s)', fontsize=20)
ax.set_ylabel('amplitude [uV]', fontsize = 20)
plt.show

