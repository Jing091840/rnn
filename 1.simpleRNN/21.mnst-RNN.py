from  tensorflow.keras.datasets import mnist
from  tensorflow.keras.models  import Model
from  tensorflow.keras.layers   import Input, Dense, SimpleRNN, Dropout
from  tensorflow.python.keras.utils   import np_utils

( X_train_data, Y_train_data ), ( X_test_data, Y_test_data ) = mnist.load_data()

X_train = X_train_data.astype( 'float32' ) / 255 
X_test  = X_test_data.astype( 'float32' ) / 255 

OneHot_train = np_utils.to_categorical( Y_train_data, 10)
OneHot_test = np_utils.to_categorical( Y_test_data, 10)

in_tensor = Input( shape=( 28, 28) )

x_tensor = SimpleRNN( 256, unroll=True )( in_tensor )

x_tensor = Dropout(0.1)( x_tensor )

out_tensor = Dense(10,activation='softmax')( x_tensor )

model = Model( inputs= in_tensor , outputs= out_tensor )

model.summary()

import tensorflow.keras.callbacks as callbacks
earlyStopping = callbacks.EarlyStopping(patience=3, restore_best_weights=True, min_delta=1e-3)

model.compile( optimizer= 'rmsprop', loss='categorical_crossentropy', metrics=['accuracy'] )
model.fit( X_train , OneHot_train , batch_size = 128 , epochs = 30 , validation_split=0.2, callbacks=[earlyStopping] )

score = model.evaluate( X_test , OneHot_test , verbose=0 )
print( 'accuracy:{}'.format( score[1] ))
    
