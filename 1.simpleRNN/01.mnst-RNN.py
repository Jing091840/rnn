from  tensorflow.keras.datasets import mnist
from  tensorflow.keras.models  import Sequential
from  tensorflow.keras.layers   import Dense, SimpleRNN, Dropout
from  tensorflow.python.keras.utils   import np_utils

( X_train_data, Y_train_data ), ( X_test_data, Y_test_data ) = mnist.load_data()

X_train = X_train_data.astype( 'float32' ) / 255 
X_test  = X_test_data.astype( 'float32' ) / 255 

OneHot_train = np_utils.to_categorical( Y_train_data, 10)
OneHot_test = np_utils.to_categorical( Y_test_data, 10)

model = Sequential()
model.add( SimpleRNN( input_shape=( 28, 28), units=1024, unroll=True ))   # 參數 unroll=True  表示計算時先展開結構，這樣會用較多的記憶體，但可縮短計算時間

# model.add(Dropout(0.1))  # 防止overfit，拋棄 : 0.1

model.add(Dense(10,activation='softmax'))

model.summary()

import tensorflow.keras.callbacks as callbacks
earlyStopping = callbacks.EarlyStopping(patience=3, restore_best_weights=True, min_delta=1e-3)

model.compile( optimizer= 'rmsprop', loss='categorical_crossentropy', metrics=['accuracy'] )
model.fit( X_train , OneHot_train , batch_size = 128 , epochs = 30 , validation_split=0.2, callbacks=[earlyStopping] )

score = model.evaluate( X_test , OneHot_test , verbose=0 )
print( 'accuracy:{}'.format( score[1] ))

