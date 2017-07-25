from keras.models import Sequential 
from keras.layers.core import Activation, Dropout
from keras.layers.convolutional import Conv1D, UpSampling1D
from keras.layers.normalization import BatchNomalization 
from keras.layers.advanced_activations import LeakyReLU
from keras.initializations import RandomNormal, Orthogonal 
from keras.layers.merge import Concatenate, Add

class AudioNet():
    """
    Modified version of Volodymyr Kuleshov's Audionet
    written with Keras functional API 
    """

    def __init__(self, L=8, X):
        self.n_layer = L #number of downsampling/upsampling layers
        self.X = X #input 

    def create_model(self):
        n_filters = [128,256,512,512,512,512,512,512]
        n_filtersizes = [65,33,17,9,9,9,9,9,9]
        residual = []
        
        print("Building model...")
        #cubic upsampling 
        x = self.X

        #downsampling layers
        for l, nf, fs in zip(range(self.n_layer), n_filters, n_filtersizes):
            x = Conv1D(filters=nf, kernel_size=fs, strides=2, kernel_initializer=Orthogonal, bias_initializer=Orthogonal)(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(0.2)(x)
            residual.append(self.model(x))
            print('D-Block: ', x.get_shape())

        #bottleneck layers
        x = Conv1D(filters=nf[-1], kernel_size=fs[-1], strides=2, kernel_initializer=Orthogonal, bias_initializer=Orthogonal)(x)
        x = Dropout(rate=0.5)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
        
        #upsampling layers
        for l, nf, fs, l_in in reversed(zip(range(self.n_layer), n_filters, n_filtersizes, residual)):
            x = Conv1D(filters=2*nf, kernel_size=fs, kernel_initializer=Orthogonal, bias_initializer=Orthogonal)(x)
            x = BatchNormalization()(x)
            x = Dropout(rate=0.5)(x)
            x = Activation('relu')(x)              
            x = UpSampling1D(size=2)(x)
            x = Concatenate([x, l_in], axis=-1) #residual
            print('U-Block: ', x.get_shape())

        #output layer 
        x = Conv1D(filters=2, kernel_size=9, kernel_initializer=RandomNormal, bias_initializer=RandomNormal)(x)
        x = UpSampling1D(size=2)(x)
        print(x.get_shape())
        g = Add([x, X]) #residual
        
        return g

if __name__ == '__main__':

