import numpy as np 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

Y = np.random.rand(3,3)
speech_len = 1

ax.scatter(Y[:speech_len,0],Y[:speech_len,1],Y[:speech_len,2],c='r',marker='o')
ax.scatter(Y[speech_len:2*speech_len,0],Y[speech_len:2*speech_len,1],Y[speech_len:2*speech_len,2],c='b',marker='^')
ax.scatter(Y[2*speech_len:,0],Y[2*speech_len:,1],Y[2*speech_len:,2],c='g',marker='8')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
fig.savefig('test_plot.png')


