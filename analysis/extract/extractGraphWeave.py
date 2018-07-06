#Copyright (C) 2018 Andreas Mayr
#Licensed under GNU General Public License v3.0 (see http://www.bioinf.jku.at/research/lsc/LICENSE and https://github.com/ml-jku/lsc/blob/master/LICENSE)

from __future__ import print_function
from __future__ import division
import math
import itertools
import numpy as np
import pandas as pd
import scipy
import scipy.io
import scipy.sparse
import sklearn
import sklearn.feature_selection
import sklearn.model_selection
import sklearn.metrics
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import imp
import os
import sys
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = ''
gpu_options=tf.ConfigProto()
gpu_options.gpu_options.allow_growth=True

import time
import gc
import argparse
basePath=os.getenv("HOME")+"/mycode/pythonCode/"
methodName="gc"
methodPath=basePath+methodName+"/"
utilsLib=imp.load_source(basePath+'utilsLib.py', basePath+"utilsLib.py")
actLib=imp.load_source(basePath+'actLib.py', basePath+"actLib.py")

#np.set_printoptions(threshold='nan')
np.set_printoptions(threshold=1000)
np.set_printoptions(linewidth=160)
np.set_printoptions(precision=4)
np.set_printoptions(edgeitems=15)
np.set_printoptions(suppress=True)
pd.set_option('display.width', 160)
pd.options.display.float_format = '{:.2f}'.format



saveBasePath=os.getenv("HOME")+"/mydata/trgpred/chembl20/resPython/"
dataPathSave=dataPathSave=os.getenv("HOME")+"/mydata/trgpred/chembl20/dataPythonReduced/"
writeDir=os.getenv("HOME")+"/mydata/trgpred/chembl20/results/dnnRes/"



f=open(dataPathSave+'folds0.pckl', "rb")
folds=pickle.load(f)
f.close()

f=open(dataPathSave+'labelsHard.pckl', "rb")
targetMat=pickle.load(f)
sampleAnnInd=pickle.load(f)
targetAnnInd=pickle.load(f)
f.close()

targetMat=targetMat
targetMat=targetMat.copy().tocsr()
targetMat.sort_indices()
targetAnnInd=targetAnnInd
targetAnnInd=targetAnnInd-targetAnnInd.min()

folds=[np.intersect1d(fold, sampleAnnInd.index.values).tolist() for fold in folds]



datasetName="graphWeave"
savePath=saveBasePath+datasetName+"/"
rocAUC=[]
prAUC=[]

for outerFold in [0,1,2]:
  savePrefix0="o"+'{0:04d}'.format(outerFold+1)
  savePrefix=savePath+savePrefix0

  saveFilename=savePrefix+".test.auc.pckl"
  if os.path.isfile(saveFilename):
    saveFile=open(saveFilename, "rb")
    reportTestAUC=pickle.load(saveFile)
    saveFile.close()

  saveFilename=savePrefix+".train.auc.pckl"
  if os.path.isfile(saveFilename):
    saveFile=open(saveFilename, "rb")
    reportTrainAUC=pickle.load(saveFile)
    saveFile.close()
  
  saveFilename=savePrefix+".finaltest.auc.pckl"
  if os.path.isfile(saveFilename):
    saveFile=open(saveFilename, "rb")
    testAUC=pickle.load(saveFile)
    saveFile.close()

  saveFilename=savePrefix+".finaltest.ap.pckl"
  if os.path.isfile(saveFilename):
    saveFile=open(saveFilename, "rb")
    testAP=pickle.load(saveFile)
    saveFile.close()
  
  saveFilename=savePrefix+".finaltrain.auc.pckl"
  if os.path.isfile(saveFilename):
    saveFile=open(saveFilename, "rb")
    trainAUC=pickle.load(saveFile)
    saveFile.close()
  
  indTrainDecrease=np.min(np.where((np.nanmean(reportTrainAUC, 1)[1:]/np.nanmean(reportTrainAUC, 1)[:-1])<0.95)[0].tolist()+[999999])
  if indTrainDecrease < 999998:
    rocAUCRes=np.array(reportTestAUC)[(indTrainDecrease+1)*20-1]
    prAUCRes=np.array([np.nan]*len(rocAUCRes))
  else:
    rocAUCRes=testAUC
    prAUCRes=testAP
  
  rocAUC.append(rocAUCRes)
  prAUC.append(prAUCRes)
  
  print(rocAUCRes)



dfROC=pd.DataFrame(data={'fold0': rocAUC[0], 'fold1': rocAUC[1], 'fold2': rocAUC[2]}, index=targetAnnInd.index.values)
dfPR=pd.DataFrame(data={'fold0': prAUC[0], 'fold1': prAUC[1], 'fold2': prAUC[2]}, index=targetAnnInd.index.values)

dfROC.to_csv(writeDir+datasetName+".roc.csv")
dfPR.to_csv(writeDir+datasetName+".pr.csv")