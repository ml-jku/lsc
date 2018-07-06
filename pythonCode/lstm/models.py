#Copyright (C) 2018 Andreas Mayr, Guenter Klambauer
#Licensed under GNU General Public License v3.0 (see http://www.bioinf.jku.at/research/lsc/LICENSE and https://github.com/ml-jku/lsc/blob/master/LICENSE)

import rdkit
import rdkit.Chem
import rdkit.Chem.MACCSkeys
import keras
import keras.backend as K
from keras.backend.tensorflow_backend import set_session
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Conv1D, Dropout, Flatten, AlphaDropout, MaxPooling1D, AveragePooling1D, BatchNormalization, Activation
from keras.metrics import binary_accuracy
from keras import optimizers, regularizers



asym=['C','N','O', 'H', 'F', 'Cl', 'P', 'B', 'Br', 'S', 'I', 'Si']
syms=['#', '(', ')', '+', '-', '1', '2', '3', '4', '5', '6', '7', '8', '=', '[', ']', '@']
arom=['c', 'n', 'o', 's']
other=['X']
endsym=['.']
oneHot=asym+syms+arom+other+endsym
oneHot.sort()
oneHot=np.array(oneHot)
otherInd=np.where(oneHot=="X")[0][0]

def myOneHot(smilesString, oneHot, otherInd, pad_len=-1):
  smilesString=smilesString+"."
  smilesList=list(smilesString)
  smilesArr=np.array(smilesList, dtype='<U2')
  irlInd=np.where(smilesArr=='r')[0].tolist()+np.where(smilesArr=='i')[0].tolist()+np.where(smilesArr=='l')[0].tolist()
  for ind in irlInd:
    smilesArr[ind-1]=smilesArr[ind-1]+smilesArr[ind]
  smilesArr=np.delete(smilesArr, irlInd)

  foundInd=np.searchsorted(oneHot, smilesArr)
  foundInd[foundInd>=len(oneHot)]=0
  foundNotInd=oneHot[foundInd]!=smilesArr
  foundInd[foundNotInd]=otherInd

  if pad_len<0:
    vec=np.zeros((len(smilesArr), len(oneHot)))
  else:
    vec=np.zeros((pad_len, len(oneHot)))

  for ind in range(len(foundInd)):
    if ind<vec.shape[0]:
      vec[ind, foundInd[ind]]=1.0/35.0
  return vec



#randomization idea based on code from Esben Jannik Bjerrum (https://github.com/EBjerrum/SMILES-enumeration)
def myRandomize(m):
  ans=list(range(m.GetNumAtoms()))
  np.random.shuffle(ans)
  nm=rdkit.Chem.RenumberAtoms(m,ans)
  return rdkit.Chem.MolToSmiles(nm, canonical=False, isomericSmiles=False)



def build_masked_loss(loss_function):
  def masked_loss_function(y_true, y_pred):
    mask=K.cast(K.greater(y_true, 0.75), K.floatx())+K.cast(K.greater(0.25, y_true), K.floatx())
    return loss_function(y_true*mask, y_pred*mask)

  return masked_loss_function



if "session" in dir():
  session.close()
session=tf.Session(config=gpu_options)
set_session(session)



data_dim=35
seq_length=350
num_targets=nrOutputTargets+167+128+256



class LSTMParams:
  def __init__(self):
    self.a="selu"
    
    self.embedding="no"
    self.embeddingDim=5.0
    self.poolingFunction="max"
    self.inputDropout=0.1

    self.ConvBlock1="yes"
    self.ConvBlock1Dim=5.0
    self.ConvBlock1Size=4
    self.ConvBlock1Stride=2
    self.ConvBlock1Dropout=0.5
    self.ConvBlock1MaxPooling="no"
    self.ConvBlock1MaxPoolingDim=2

    self.ConvBlock2="no"
    self.ConvBlock2Dim=5.0
    self.ConvBlock2Size=3
    self.ConvBlock2Stride=1
    self.ConvBlock2Dropout=0.5
    self.ConvBlock2MaxPooling="yes"
    self.ConvBlock2MaxPoolingDim=2

    self.ConvBlock3="no"
    self.ConvBlock3Dim=5.0
    self.ConvBlock3Size=3
    self.ConvBlock3Stride=1
    self.ConvBlock3Dropout=0.5
    self.ConvBlock3MaxPooling="yes"
    self.ConvBlock3MaxPoolingDim=2

    self.BatchNorm="no"

    self.LSTM1="yes"
    self.LSTM1Dim=7.0
    self.LSTM1Dir="backward"
    self.LSTM1Dropout=0.1

    self.LSTM2="yes"
    self.LSTM2Dim=9.0
    self.LSTM2Dir="backward"
    self.LSTM2Dropout=0.1

    self.Dense1="no"
    self.Dense1Dim=9.0
    self.Dense1Dropout=0.5

    self.l1=0.0
    self.l2=1e-9



lstmparams=LSTMParams()



if lstmparams.a=='relu':
  kernel_initializer='he_normal'
  dropoutFactor=1.0
  dropoutFunction=Dropout
else:
  kernel_initializer='lecun_uniform'
  dropoutFactor=1.0
  dropoutFunction=Dropout

    
if lstmparams.poolingFunction=="max":
  poolingFunction=MaxPooling1D
else:
  poolingFunction=AveragePooling1D


model = Sequential()


model.add(Dropout(lstmparams.inputDropout, input_shape=(seq_length, data_dim)))
if lstmparams.embedding=='yes':
  model.add(Conv1D(int(2**lstmparams.embeddingDim), 1, strides=1, kernel_initializer='lecun_uniform', activation='linear', padding='same', use_bias=False))


if lstmparams.ConvBlock1=='yes':
  model.add(Conv1D(int(2**lstmparams.ConvBlock1Dim), int(lstmparams.ConvBlock1Size), strides=int(lstmparams.ConvBlock1Stride), kernel_initializer=kernel_initializer, activation=lstmparams.a, padding='same', use_bias=False))
  model.add(dropoutFunction(lstmparams.ConvBlock1Dropout*dropoutFactor))
  model.add(Conv1D(int(2**lstmparams.ConvBlock1Dim), int(lstmparams.ConvBlock1Size), strides=int(lstmparams.ConvBlock1Stride), kernel_initializer=kernel_initializer, activation=lstmparams.a, padding='same', use_bias=False))
  model.add(dropoutFunction(lstmparams.ConvBlock1Dropout*dropoutFactor))
    
if lstmparams.ConvBlock1=='yes' and lstmparams.ConvBlock1MaxPooling=='yes':
  model.add(poolingFunction(int(lstmparams.ConvBlock1MaxPoolingDim)))


if lstmparams.ConvBlock2=='yes':
  model.add(Conv1D(int(2**lstmparams.ConvBlock2Dim), int(lstmparams.ConvBlock2Size), strides=int(lstmparams.ConvBlock2Stride), kernel_initializer=kernel_initializer, activation=lstmparams.a, padding='same', use_bias=False))
  model.add(dropoutFunction(lstmparams.ConvBlock2Dropout*dropoutFactor))
  model.add(Conv1D(int(2**lstmparams.ConvBlock2Dim), int(lstmparams.ConvBlock2Size), strides=int(lstmparams.ConvBlock2Stride), kernel_initializer=kernel_initializer, activation=lstmparams.a, padding='same', use_bias=False))
  model.add(dropoutFunction(lstmparams.ConvBlock2Dropout*dropoutFactor))
    
if lstmparams.ConvBlock2=='yes' and lstmparams.ConvBlock2MaxPooling=='yes':
  model.add(poolingFunction(int(lstmparams.ConvBlock2MaxPoolingDim)))


if lstmparams.ConvBlock3=='yes':
  model.add(Conv1D(int(2**lstmparams.ConvBlock3Dim), int(lstmparams.ConvBlock3Size), strides=int(lstmparams.ConvBlock3Stride), kernel_initializer=kernel_initializer, activation=lstmparams.a, padding='same', use_bias=False))
  model.add(dropoutFunction(lstmparams.ConvBlock3Dropout*dropoutFactor))
  model.add(Conv1D(int(2**lstmparams.ConvBlock3Dim), int(lstmparams.ConvBlock3Size), strides=int(lstmparams.ConvBlock3Stride), kernel_initializer=kernel_initializer, activation=lstmparams.a, padding='same', use_bias=False))
  model.add(dropoutFunction(lstmparams.ConvBlock3Dropout*dropoutFactor))
    
if lstmparams.ConvBlock3=='yes' and lstmparams.ConvBlock3MaxPooling=='yes':
  model.add(poolingFunction(int(lstmparams.ConvBlock3MaxPoolingDim)))


if lstmparams.BatchNorm=="yes":
  model.add(BatchNormalization(center=False, scale=False))


if lstmparams.LSTM1=='yes':
  if lstmparams.LSTM1Dir=="backward":
    model.add(LSTM(int(2**lstmparams.LSTM1Dim), go_backwards=True, return_sequences=True, recurrent_activation='sigmoid'))
  else:
    model.add(LSTM(int(2**lstmparams.LSTM1Dim), go_backwards=False, return_sequences=True, recurrent_activation='sigmoid'))
  model.add(Dropout(lstmparams.LSTM1Dropout))


if lstmparams.LSTM2=='yes':
  if lstmparams.LSTM2Dir=="backward":
    model.add(LSTM(int(2**lstmparams.LSTM2Dim), go_backwards=True, return_sequences=False, recurrent_activation='sigmoid'))
  else:
    model.add(LSTM(int(2**lstmparams.LSTM2Dim), go_backwards=False, return_sequences=False, recurrent_activation='sigmoid'))
  model.add(Dropout(lstmparams.LSTM2Dropout))
if lstmparams.LSTM2=='no':
    model.add(Flatten())


if lstmparams.Dense1=='yes':
  model.add(Dense(int(2**lstmparams.Dense1Dim), activation=lstmparams.a, kernel_initializer=kernel_initializer, bias_initializer='zeros'))
  model.add(dropoutFunction(lstmparams.Dense1Dropout*dropoutFactor))


model.add(Dense(num_targets, activation='sigmoid', kernel_initializer='lecun_uniform', bias_initializer='zeros', kernel_regularizer=regularizers.l1_l2(lstmparams.l1, lstmparams.l2)))


adam=optimizers.Adam(clipvalue=0.1)
model.compile(loss=build_masked_loss(K.binary_crossentropy), optimizer=adam)


