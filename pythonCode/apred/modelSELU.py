#Copyright (C) 2018 Andreas Mayr
#Licensed under GNU General Public License v3.0 (see http://www.bioinf.jku.at/research/lsc/LICENSE and https://github.com/ml-jku/lsc/blob/master/LICENSE)

nrLayers=hyperParams.iloc[paramNr].nrLayers
nrNodes=hyperParams.iloc[paramNr].nrNodes
basicArchitecture=hyperParams.iloc[paramNr].basicArchitecture

nrInputFeatures=nrDenseFeatures+nrSparseFeatures

hiddenLayerSizes=[nrNodes]*nrLayers
layerSizes=[nrInputFeatures]+hiddenLayerSizes+[nrOutputTargets]

if basicArchitecture=="selu":
  activationFunction=actLib.selu
  dropoutFunction=actLib.dropout_stableVariance
  idropoutFunction=actLib.dropout_stableVariance
  initScale=1.0



tf.reset_default_graph()
if "session" in dir():
  session.close()
session=tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))



if nrDenseFeatures>0.5:
  xDenseData=tf.placeholder(tf.float32, [None, nrDenseFeatures])
  sh0=tf.shape(xDenseData)[0]

if nrSparseFeatures>0.5:
  xIndices = tf.placeholder(tf.int64, [None, 2])
  xValues = tf.placeholder(tf.float32, [None])
  xDim = tf.placeholder(tf.int64, [2])
  xSparseData=tf.SparseTensor(indices=xIndices, values=xValues, dense_shape=xDim)
  sparseMeanInit=tf.placeholder(tf.float32, [1, nrSparseFeatures])
  sparseMean=tf.Variable(tf.zeros([1, nrSparseFeatures]), trainable=False, dtype=tf.float32)
  sh0=tf.shape(xSparseData)[0]



yDenseData=tf.placeholder(tf.float32, [None, nrOutputTargets])

yIndices=tf.placeholder(tf.int64, [None, 2])
yValues=tf.placeholder(tf.float32, [None])
yDim=tf.placeholder(tf.int64, [2])
ySparseData=tf.SparseTensor(indices=yIndices, values=yValues, dense_shape=yDim)
ySparseMask=tf.SparseTensor(indices=yIndices, values=tf.ones_like(yValues), dense_shape=yDim)



inputDropout = tf.placeholder(tf.float32)
hiddenDropout = tf.placeholder(tf.float32)
lrGeneral = tf.placeholder(tf.float32)
lrWeight = tf.placeholder(tf.float32)
lrBias = tf.placeholder(tf.float32)
l2PenaltyWeight = tf.placeholder(tf.float32)
l2PenaltyBias = tf.placeholder(tf.float32)
l1PenaltyWeight = tf.placeholder(tf.float32)
l1PenaltyBias = tf.placeholder(tf.float32)
mom = tf.placeholder(tf.float32)
biasInit=tf.placeholder(tf.float32, [nrOutputTargets])
is_training=tf.placeholder(tf.bool)

weightTensors=[]
biasTensors=[]
hidden=[]
hiddenAct=[]
hiddenActMod=[]

with tf.variable_scope('layer_'+str(0)):
  hiddenActl=[]
  hiddenActModl=[]
  if nrDenseFeatures>0.5:
    hiddenActl.append(xDenseData)
    hiddenActModl.append(xDenseData*tf.to_float(tf.random_uniform([sh0, xDenseData.get_shape()[1].value])<(1.0-inputDropout)))
  if nrSparseFeatures>0.5:
    hiddenActl.append(xSparseData)
    if not (normalizeGlobalSparse or normalizeLocalSparse):
      #dropout: expected is real sparse input data
      hiddenActModl.append(tf.sparse_retain(xSparseData, tf.random_uniform([tf.shape(xSparseData.values)[0]])<(1.0-inputDropout)))
    else:
      #dropout: not possible as input is virtually non-sparse
      hiddenActModl.append(xSparseData)
  hiddenActInit=hiddenActl
  hiddenActModInit=hiddenActModl
  
  weightTensors.append(None)
  biasTensors.append(None)
  hidden.append(None)
  hiddenAct.append(hiddenActl)
  hiddenActMod.append(hiddenActModl)

idTensors=[]
hdTensors=[]

layernr=1
with tf.variable_scope('layer_'+str(layernr)):
  wList=[]
  if nrDenseFeatures>0.5:
    WlDense=tf.get_variable("W"+str(layernr)+"_dense", trainable=True, initializer=tf.random_normal([nrDenseFeatures, layerSizes[layernr]], stddev=np.sqrt(initScale/float(layerSizes[layernr-1]))))
    wList.append(WlDense)
  if nrSparseFeatures>0.5:
    WlSparse=tf.get_variable("W"+str(layernr)+"_sparse", trainable=True, initializer=tf.random_normal([nrSparseFeatures, layerSizes[layernr]], stddev=np.sqrt(initScale/float(layerSizes[layernr-1]))))
    wList.append(WlSparse)
    #sparseMeanWSparse=tf.Variable(tf.zeros([1, layerSizes[layernr]]), trainable=False, dtype=tf.float32)
    sparseMeanWSparse=tf.matmul(sparseMean, WlSparse)
  bl=tf.get_variable('b'+str(layernr), shape=[layerSizes[layernr]], trainable=True, initializer=tf.zeros_initializer())
  
  regRaw=l2PenaltyBias*tf.nn.l2_loss(bl)+l1PenaltyBias*tf.reduce_sum(tf.abs(bl))
  if nrDenseFeatures>0.5:
    if nrSparseFeatures>0.5:
      regRaw=regRaw+l2PenaltyWeight*(tf.nn.l2_loss(WlSparse)+tf.nn.l2_loss(WlDense))+l1PenaltyWeight*(tf.reduce_sum(tf.abs(WlSparse))+tf.reduce_sum(tf.abs(WlDense)))
      hiddenl=tf.matmul(hiddenActModl[0], WlDense)+tf.sparse_tensor_dense_matmul(hiddenActModl[1], WlSparse)+(bl+sparseMeanWSparse)
    else:
      regRaw=regRaw+l2PenaltyWeight*tf.nn.l2_loss(WlDense)+l1PenaltyWeight*tf.reduce_sum(tf.abs(WlDense))
      hiddenl=tf.matmul(hiddenActModl[0], WlDense)+bl
  else:
    if nrSparseFeatures>0.5:
      regRaw=regRaw+l2PenaltyWeight*tf.nn.l2_loss(WlSparse)+l1PenaltyWeight*tf.reduce_sum(tf.abs(WlSparse))
      hiddenl=tf.sparse_tensor_dense_matmul(hiddenActModl[0], WlSparse)+(bl+sparseMeanWSparse)
  hiddenActModl=hiddenl*tf.to_float(tf.random_uniform([sh0, hiddenl.get_shape()[1].value])<(1.0-hiddenDropout))
  hiddenActl=activationFunction(hiddenActModl)
  
  weightTensors.append(wList)
  biasTensors.append(bl)
  hidden.append(hiddenl)
  hiddenAct.append(hiddenActl)
  hiddenActMod.append(hiddenActModl)
  
  if nrDenseFeatures>0.5:
    hdTensors.append(WlDense)
    idTensors.append(WlDense)
  if nrSparseFeatures>0.5:
    hdTensors.append(WlSparse)
    #hdTensors.append(sparseMeanWSparse)
    if not (normalizeGlobalSparse or normalizeLocalSparse):
      idTensors.append(WlSparse)



for layernr in range(2, len(layerSizes)-1):
  with tf.variable_scope('layer_'+str(layernr)):
    Wl=tf.get_variable("W"+str(layernr), trainable=True, initializer=tf.random_normal([layerSizes[layernr-1], layerSizes[layernr]], stddev=np.sqrt(initScale/float(layerSizes[layernr-1]))))
    bl=tf.get_variable('b'+str(layernr), shape=[layerSizes[layernr]], trainable=True, initializer=tf.zeros_initializer())
    
    regRaw=regRaw+l2PenaltyWeight*tf.nn.l2_loss(Wl)+l1PenaltyWeight*tf.reduce_sum(tf.abs(Wl))+l2PenaltyBias*tf.nn.l2_loss(bl)+l1PenaltyBias*tf.reduce_sum(tf.abs(bl))
    hiddenl=tf.matmul(hiddenActl, Wl) + bl
    hiddenActModl=hiddenl*tf.to_float(tf.random_uniform([sh0, hiddenl.get_shape()[1].value])<(1.0-hiddenDropout))
    hiddenActl=activationFunction(hiddenActModl)
    
    weightTensors.append(Wl)
    biasTensors.append(bl)
    hidden.append(hiddenl)
    hiddenAct.append(hiddenActl)
    hiddenActMod.append(hiddenActModl)
    
    hdTensors.append(Wl)

layernr=len(layerSizes)-1
with tf.variable_scope('layer_'+str(layernr)):
  Wl=tf.get_variable("W"+str(layernr), trainable=True, initializer=tf.random_normal([layerSizes[layernr-1], layerSizes[layernr]], stddev=np.sqrt(initScale/float(layerSizes[layernr-1]))))
  bl=tf.get_variable('b'+str(layernr), shape=[layerSizes[layernr]], trainable=True, initializer=tf.zeros_initializer())
  
  regRaw=regRaw+l2PenaltyWeight*tf.nn.l2_loss(Wl)+l1PenaltyWeight*tf.reduce_sum(tf.abs(Wl))+l2PenaltyBias*tf.nn.l2_loss(bl)+l1PenaltyBias*tf.reduce_sum(tf.abs(bl))
  hiddenl=tf.matmul(hiddenActl, Wl) + bl
  
  weightTensors.append(Wl)
  biasTensors.append(bl)
  hidden.append(hiddenl)
  hiddenAct.append(None)
  hiddenActMod.append(None)
  
  #hdTensors.append(Wl)



naMat=tf.where(tf.abs(yDenseData) < 0.5, tf.zeros_like(yDenseData), tf.ones_like(yDenseData))
lossRawDense=tf.nn.sigmoid_cross_entropy_with_logits(labels=(yDenseData+1.0)/2.0, logits=hiddenl)*naMat
errOverallDense=tf.reduce_mean(tf.reduce_sum(lossRawDense,1))+regRaw
predNetworkDense=tf.nn.sigmoid(hiddenl)
optimizerDense=tf.train.MomentumOptimizer(momentum=mom, learning_rate=lrGeneral).minimize(errOverallDense)

hiddenlSelected=tf.gather_nd(hiddenl, yIndices)
lossRawSelected=tf.nn.sigmoid_cross_entropy_with_logits(labels=(yValues+1.0)/2.0, logits=hiddenlSelected)
lossRawSparse=tf.SparseTensor(indices=yIndices, values=lossRawSelected, dense_shape=yDim)
errOverallSparse=tf.reduce_mean(tf.sparse_reduce_sum(lossRawSparse, 1))+regRaw
predNetworkSparse=tf.nn.sigmoid(hiddenlSelected)
optimizerSparse=tf.train.MomentumOptimizer(momentum=mom, learning_rate=lrGeneral).minimize(errOverallSparse)

predNetwork=tf.nn.sigmoid(hiddenl)

class MyNoOp:
  op=tf.no_op()

init=tf.global_variables_initializer()
biasInitOp=biasTensors[-1].assign(biasInit)
if nrSparseFeatures>0.5:
  #sparseMeanWSparseOp=sparseMeanWSparse.assign(tf.matmul(sparseMean, WlSparse))
  sparseMeanWSparseOp=MyNoOp()
  sparseMeanInitOp=sparseMean.assign(sparseMeanInit)

scaleTrainHd=[tf.assign(hdTensors[i], hdTensors[i]/tf.sqrt(1.0-hiddenDropout)).op for i in range(0, len(hdTensors))]
scalePredictHd=[tf.assign(hdTensors[i], hdTensors[i]*tf.sqrt(1.0-hiddenDropout)).op for i in range(0, len(hdTensors))]
scaleTrainId=[tf.assign(idTensors[i], idTensors[i]/(1.0-inputDropout)).op for i in range(0, len(idTensors))]
scalePredictId=[tf.assign(idTensors[i], idTensors[i]*(1.0-inputDropout)).op for i in range(0, len(idTensors))]

checkNA=[tf.reduce_any(tf.is_nan(x)) for x in weightTensors[1]+weightTensors[2:]+biasTensors[1:]]
