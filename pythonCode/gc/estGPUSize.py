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
gpu_options=tf.ConfigProto()
gpu_options.gpu_options.allow_growth=True
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
parser.add_argument("-methodName", help="Method name", type=str, default="gc")
parser.add_argument("-originalData", help="Path for original data in python Format", type=str, default=os.getenv("HOME")+"/mydata/trgpred/chembl20/dataPythonReduced/")
parser.add_argument("-datasetNames", help="DatasetNames", nargs='+', type=str, default=["graphConv"])
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



  graphInputData=None
  denseOutputData=None
  sparseOutputData=None
  exec(open(methodPath+'loadData.py').read(), globals())

  if not denseOutputData is None:
    nrOutputTargets=denseOutputData.shape[1]
  if not sparseOutputData is None:
    nrOutputTargets=sparseOutputData.shape[1]



  manager=Manager()
  sizeArray = Array("l", [0]*hyperParams.shape[0])
  

  def myfuncHyper():
    import pynvml
    
    pynvml.nvmlInit()
    deviceCount = pynvml.nvmlDeviceGetCount()
    handle=pynvml.nvmlDeviceGetHandleByIndex(int(os.environ['CUDA_VISIBLE_DEVICES']))
    gpuMem=pynvml.nvmlDeviceGetMemoryInfo(handle)
    print("Init")
    print(gpuMem.used)
    
    
    
    exec(open(methodPath+'models.py').read(), globals())
    
    
    
    randSamples=np.random.random_integers(0, len(graphInputData), batchSize)
    batchGraphX=graphInputData[randSamples]
    batchDenseY=denseOutputData[randSamples]
    
    batchInputSingle=[mychemblConvertedMols[molX] for molX in batchGraphX]
    batchInput=batchFunc(model, batchInputSingle)
    myfeedDict=batchInput
    myfeedDict[model.label]=(batchDenseY>0.5).astype(np.integer)
    myfeedDict[model.weights]=(np.abs(batchDenseY)>0.5).astype(np.integer)
    myfeedDict[model._training_placeholder]=1.0
    with model._get_tf("Graph").as_default():
      try:
        model.session.run([model._get_tf('train_op'), updateOps], feed_dict=myfeedDict)
      except:
        print("Error in Training!")
    myfeedDict=batchInput
    myfeedDict[model._training_placeholder]=0.0
    with model._get_tf("Graph").as_default():
      myres=model.session.run(model.outputs[0], feed_dict=myfeedDict)
    
    
    print("GPU")
    gpuMem=pynvml.nvmlDeviceGetMemoryInfo(handle)
    sizeArray[paramNr]=gpuMem.used
    print(gpuMem.used)

  for paramNr in range(0, hyperParams.shape[0]):
    p = Process(target=myfuncHyper)
    p.start()
    p.join()

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
