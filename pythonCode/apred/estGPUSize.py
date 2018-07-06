#Copyright (C) 2018 Andreas Mayr
#Licensed under GNU General Public License v3.0 (see http://www.bioinf.jku.at/research/lsc/LICENSE and https://github.com/ml-jku/lsc/blob/master/LICENSE)

from __future__ import print_function
from __future__ import division
import numpy as np
import scipy
import scipy.sparse
import pandas as pd
import itertools
import pickle
import imp
import os
import pathlib
from multiprocessing import Process, Manager, Array
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = ''
gpu_options = tf.GPUOptions(allow_growth=True)
import time
import gc
import argparse
basePath=os.getenv("HOME")+"/mycode/pythonCode/"

#np.set_printoptions(threshold='nan')
np.set_printoptions(threshold=1000)
np.set_printoptions(linewidth=160)
np.set_printoptions(precision=4)
np.set_printoptions(edgeitems=15)
np.set_printoptions(suppress=True)
pd.set_option('display.width', 160)
pd.options.display.float_format = '{:.2f}'.format



parser = argparse.ArgumentParser()
parser.add_argument("-availableGPU", help="GPU for Test", type=int, default=0)
parser.add_argument("-methodName", help="Method name", type=str, default="apred")
parser.add_argument("-originalData", help="Path for original data in python Format", type=str, default=os.getenv("HOME")+"/mydata/trgpred/chembl20/dataPythonReduced/")
parser.add_argument("-datasetNames", help="DatasetNames", nargs='+', type=str, default=["static", "semi", "ecfp", "dfs", "ecfpTox"])
parser.add_argument("-saveBasePath", help="saveBasePath", type=str, default=os.getenv("HOME")+"/mydata/trgpred/chembl20/resPython/")
args = parser.parse_args()



availableGPU=args.availableGPU

methodName=args.methodName
methodPath=basePath+methodName+"/"
utilsLib=imp.load_source(basePath+'utilsLib.py', basePath+"utilsLib.py")
actLib=imp.load_source(basePath+'actLib.py', basePath+"actLib.py")

dataPathSave=args.originalData

datasetNames=args.datasetNames

saveBasePath=args.saveBasePath
if not os.path.exists(saveBasePath):
  os.makedirs(saveBasePath)



os.environ['CUDA_VISIBLE_DEVICES']=str(availableGPU)



for datasetName in datasetNames:
  
  
  
  savePath=saveBasePath+datasetName+"/"
  if not os.path.exists(savePath):
    os.makedirs(savePath)

  batchSize=128



  exec(open(methodPath+'hyperparams.py').read(), globals())



  denseInputData=None
  sparseInputData=None
  denseOutputData=None
  sparseOutputData=None
  exec(open(methodPath+'loadData.py').read(), globals())

  if not denseInputData is None:
    nrDenseFeatures=denseInputData.shape[1]
  else:
    nrDenseFeatures=0
  if not sparseInputData is None:
    if datasetName=="dfs":
      sparsenesThr=0.02
    else:
      sparsenesThr=0.0025
    featSel=sparseInputData.getnnz(axis=0)/float(sparseInputData.shape[0]) > sparsenesThr
    
    nrSparseFeatures=sparseInputData.shape[1]
    estNonZFeatures=int(np.percentile(sparseInputData.getnnz(1), 90)+0.5)
    
    nrSparseFeatures=np.sum(featSel)
    nrDenseFeatures=nrSparseFeatures
    nrSparseFeatures=0
    
  else:
    nrSparseFeatures=0
  if not denseOutputData is None:
    nrOutputTargets=denseOutputData.shape[1]
  if not sparseOutputData is None:
    nrOutputTargets=sparseOutputData.shape[1]



  manager=Manager()
  sizeDict=manager.dict()
  sizeArray = Array("l", [0]*hyperParams.shape[0])

  def myfuncHyper():
    import pynvml
    
    pynvml.nvmlInit()
    deviceCount = pynvml.nvmlDeviceGetCount()
    handle=pynvml.nvmlDeviceGetHandleByIndex(int(os.environ['CUDA_VISIBLE_DEVICES']))
    gpuMem=pynvml.nvmlDeviceGetMemoryInfo(handle)
    print("Init")
    print(gpuMem.used)
    
    
    
    nrLayers=hyperParams.iloc[paramNr].nrLayers
    nrNodes=hyperParams.iloc[paramNr].nrNodes
    basicArchitecture=hyperParams.iloc[paramNr].basicArchitecture
    dictionary = {
      'nrLayers': [nrLayers], 
      'nrNodes': [nrNodes],
      'basicArchitecture': [basicArchitecture]
    }
    
    
    
    if not str(dictionary) in sizeDict:
      if basicArchitecture=="selu":
        exec(open(methodPath+'modelSELU.py').read(), globals())
      elif basicArchitecture=="relu":
        exec(open(methodPath+'modelReLU.py').read(), globals())      
      
      my_yDenseData=np.zeros((batchSize, nrOutputTargets))
      
      my_inputDropout=0.2
      my_hiddenDropout=0.5
      my_lrGeneral=0.1
      my_lrWeight=0.0
      my_lrBias=0.0
      my_l2PenaltyWeight=0.1
      my_l2PenaltyBias=0.1
      my_l1PenaltyWeight=0.1
      my_l1PenaltyBias=0.1
      my_mom=0.0
      my_biasInit=np.zeros(nrOutputTargets)
      my_is_training=True
      
      if nrDenseFeatures>0:
        my_xDenseData=np.zeros((batchSize, nrDenseFeatures))
      
      if nrSparseFeatures>0:
        indices=np.random.random_integers(0, nrSparseFeatures-1, size=batchSize*estNonZFeatures)
        indptr=np.random.random_integers(0, len(indices)-1, size=batchSize)
        indptr.sort()
        indptr[0]=0
        indptr=np.append(indptr, len(indices))
        data=np.random.random_integers(0, 430000, size=batchSize*estNonZFeatures).astype(np.float32)/430000
        mycsr=scipy.sparse.csr_matrix((data, indices, indptr), (batchSize, nrSparseFeatures))
        mycsr.sort_indices()
        nonzx=mycsr.nonzero()
        valnonzx=(mycsr)[nonzx[0],nonzx[1]]
        
        my_xIndices=np.int64(np.vstack(nonzx).T)
        my_xValues=valnonzx.A.flatten()
        my_xDim=[mycsr.shape[0], mycsr.shape[1]]
        
        my_sparseMeanInit=np.zeros((1, nrSparseFeatures))
      
      
      myfeed={
        yDenseData: my_yDenseData,
        inputDropout: my_inputDropout,
        hiddenDropout: my_hiddenDropout,
        lrGeneral: my_lrGeneral,
        lrWeight: my_lrWeight,
        lrBias: my_lrBias,
        l2PenaltyWeight: my_l2PenaltyWeight,
        l2PenaltyBias: my_l2PenaltyBias,
        l1PenaltyWeight: my_l1PenaltyWeight,
        l1PenaltyBias: my_l1PenaltyBias,
        mom: my_mom,
        biasInit: my_biasInit,
        is_training: my_is_training
      }
      
      if nrDenseFeatures>0:
        myfeed.update({
          xDenseData: my_xDenseData
        })
      
      if nrSparseFeatures>0:
        myfeed.update({
          xIndices: my_xIndices,
          xValues: my_xValues,
          xDim: my_xDim,
          sparseMeanInit: my_sparseMeanInit,
        })
      
      
      
      _=session.run([init])
      if nrSparseFeatures>0:
        _=session.run([sparseMeanInitOp], feed_dict=myfeed)
        _=session.run([sparseMeanWSparseOp])
      _=session.run([optimizerDense], feed_dict=myfeed)
      _=session.run([predNetwork], feed_dict=myfeed)

      print("GPU")
      gpuMem=pynvml.nvmlDeviceGetMemoryInfo(handle)
      sizeDict[str(dictionary)]=gpuMem.used
      sizeArray[paramNr]=gpuMem.used
      print(gpuMem.used)
    else:
      sizeArray[paramNr]=sizeDict[str(dictionary)]

  for paramNr in range(0, hyperParams.shape[0]):
    p = Process(target=myfuncHyper)
    p.start()
    p.join()
    print(sizeDict)

  sizeArr=np.array(sizeArray)
  sizeArr.tofile(savePath+"hyperSize.npy")
  
  
  
  totalSize = Array("l", [0])

  def myfuncTotal():
    import pynvml

    pynvml.nvmlInit()
    deviceCount = pynvml.nvmlDeviceGetCount()
    handle=pynvml.nvmlDeviceGetHandleByIndex(int(os.environ['CUDA_VISIBLE_DEVICES']))
    gpuMem=pynvml.nvmlDeviceGetMemoryInfo(handle)
    totalMem=gpuMem.total
    totalSize[0]=totalMem

  p = Process(target=myfuncTotal)
  p.start()
  p.join()
    
  totalSize=np.array(totalSize)
  totalSize.tofile(savePath+"totalSize.npy")
  
  
  
  print(datasetName)
  print(sizeArr/totalSize[0])
  print(np.max(sizeArr/totalSize[0]))
