from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Activation, Dense, LSTM, TimeDistributed, Bidirectional, Masking, Dropout
from keras.constraints import maxnorm
                        
model = Sequential()
model.add(Masking(mask_value=-1, input_shape=(200,20)))
model.add(TimeDistributed(Dense(128, activation='relu')))
"""
                        17     model = Sequential()
                         18     model.add(Masking(mask_value=-1, input_shape=(200,20)))
                          19     model.add(TimeDistributed(Dense(128, activation='relu')))
                           20     model.add(TimeDistributed(Dense(32, activation='linear')))
                            21     model.add(TimeDistributed(Dense(128, activation='relu')))
                             22     model.add(Bidirectional(LSTM(128, dropout=0.5, recurrent_dropout=0.2, return_sequences=True)))
                              23     model.add(Bidirectional(LSTM(128, dropout=0.5, recurrent_dropout=0.2, return_sequences=True)))
                               24     model.add(TimeDistributed(Dense(128, activation='relu')))
                                25     model.add(Dropout(0.2))
                                 26     model.add(TimeDistributed(Dense(128, activation='relu')))
                                  27     model.add(Bidirectional(LSTM(128, dropout=0.5, recurrent_dropout=0.2, return_sequences=True)))
                                   28         model.add(Bidirectional(LSTM(128, dropout=0.5, recurrent_dropout=0.2, return_sequences=True)))
                                    29     model.add(TimeDistributed(Dense(1024, activation='relu', kernel_constraint=maxnorm(3.))))
                                     30     model.add(Dropout(0.2))
                                      31     model.add(TimeDistributed(Dense(512, activation='relu', kernel_constraint=maxnorm(3.))))
                                       32     model.add(Dropout(0.2))
                                        33     model.add(Dense(1, activation='sigmoid'))
                                         34
                                          35     print('Compiling model...')
                                           36     model.compile(loss='binary_crossentropy',
                                            37               optimizer='adam',
                                             38               metrics=['accuracy'],
                                              39               sample_weight_mode=None)
   """
print(model.summary())
plot_model(model, to_file='model.png')
