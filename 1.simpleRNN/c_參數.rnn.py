
from  tensorflow.keras.models  import Model
from  tensorflow.keras.layers   import Input, SimpleRNN


RNN_timesteps= 29  #  步數
RNN_i_Dim =     3  #  特徵 features
RNN_o_Dim =     2  #  RNN 輸出的維數

RNN_input  = Input(shape=( RNN_timesteps , RNN_i_Dim ))
RNN_output = SimpleRNN( RNN_o_Dim )( RNN_input )
model_RNN  = Model( inputs=RNN_input , outputs=RNN_output )

model_RNN.summary()

print(  ( RNN_i_Dim + RNN_o_Dim ) * RNN_o_Dim + RNN_o_Dim  )
print(  ( RNN_i_Dim + RNN_o_Dim + 1 ) * RNN_o_Dim          )  # 1 是 bios


W = model_RNN.layers[1].get_weights()[0]
U = model_RNN.layers[1].get_weights()[1]
b = model_RNN.layers[1].get_weights()[2]

print("W", W.size, '： RNN_i_Dim * RNN_o_Dim = ', RNN_i_Dim * RNN_o_Dim )
print("U", U.size, '： RNN_o_Dim * RNN_o_Dim = ', RNN_o_Dim * RNN_o_Dim )
print("bias", b.size , '： RNN_o_Dim = ',  RNN_o_Dim)



RNN_timesteps= 28  #  步數
RNN_i_Dim =    29  #  特徵 features
RNN_o_Dim =   256  #  RNN 輸出的維數

RNN_input  = Input(shape=( RNN_timesteps , RNN_i_Dim ))
RNN_output = SimpleRNN( RNN_o_Dim )( RNN_input )
model_RNN  = Model( inputs=RNN_input , outputs=RNN_output )

model_RNN.summary()

print(  ( RNN_i_Dim + RNN_o_Dim ) * RNN_o_Dim + RNN_o_Dim  )
print(  ( RNN_i_Dim + RNN_o_Dim + 1 ) * RNN_o_Dim          )




