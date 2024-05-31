from  tensorflow.keras.datasets import cifar10
from  tensorflow.keras.models  import Sequential
from  tensorflow.keras.layers   import Dense, SimpleRNN, Dropout
from  tensorflow.keras.utils   import to_categorical

( X_train_data, Y_train_data ), ( X_test_data, Y_test_data ) = cifar10.load_data()

X_train = X_train_data.reshape( -1, 32, 32*3 ).astype( 'float32' ) / 255.
X_test  = X_test_data.reshape( -1, 32, 32*3 ).astype( 'float32' ) / 255.

OneHot_train = to_categorical( Y_train_data, 10)
OneHot_test  = to_categorical( Y_test_data, 10)

model = Sequential()
model.add( SimpleRNN( input_shape=( 32, 32*3 ), units=256, unroll=True ))   # 參數 unroll=True  表示計算時先展開結構，這樣會用較多的記憶體，但可縮短計算時間

model.add(Dropout(0.1))  # 防止overfit，拋棄 : 0.1

model.add(Dense(10,activation='softmax'))

model.summary()

import tensorflow.keras.callbacks as callbacks
earlyStopping = callbacks.EarlyStopping(patience=3, restore_best_weights=True, min_delta=1e-3)

model.compile( optimizer= 'rmsprop', loss='categorical_crossentropy', metrics=['accuracy'] )
model.fit( X_train , OneHot_train , batch_size = 128 , epochs = 30 , validation_split=0.2, callbacks=[earlyStopping] )

score = model.evaluate( X_test , OneHot_test , verbose=0 )
print( 'accuracy:{}'.format( score[1] ))

