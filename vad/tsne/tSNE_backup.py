import numpy as np 
from mfcc_hdf5 import combine_mfcc_id
from sklearn import manifold 
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

#Data 
print('Loading data...')
f = open('baseline_data','rb')
train_speech = pickle.load(f)
train_noise = pickle.load(f)
train_noisy = pickle.load(f)
test_speech = pickle.load(f)
test_noise = pickle.load(f)
test_noisy = pickle.load(f)
f.close()

#data process 
test_speech = test_speech.transpose(2,0,1).reshape(20,-1).T
test_noise = test_noise.transpose(2,0,1).reshape(20,-1).T
test_noisy = test_noisy.transpose(2,0,1).reshape(20,-1).T

np.random.shuffle(test_speech)
np.random.shuffle(test_noise)
np.random.shuffle(test_noisy)

#trim
speech_len = 10000
test_speech = test_speech[:speech_len] 
test_noise = test_noise[:speech_len]
test_noisy = test_noisy[:speech_len]
print(speech_len)

#test 
print(test_speech.shape)
print(test_noise.shape)
print(test_noisy.shape)

#concatenate
con = np.concatenate((test_speech,test_noise,test_noisy),axis=0)
print(con.shape)

#Model 	
print("Reducing dimension...")
model = manifold.TSNE(n_components=3, init='pca', random_state=0, verbose=1) 
Y = model.fit_transform(con)

#Plot 
print('Plotting...')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(Y[:speech_len,0],Y[:speech_len,1],Y[:speech_len,2],c='r',marker='o')
ax.scatter(Y[speech_len:2*speech_len,0],Y[speech_len:2*speech_len,1],Y[speech_len:2*speech_len,2],c='b',marker='^')
ax.scatter(Y[2*speech_len:,0],Y[2*speech_len:,1],Y[2*speech_len:,2],c='g',marker='8')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
fig.savefig('tsne_re.png')

