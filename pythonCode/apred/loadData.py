#Copyright (C) 2018 Andreas Mayr
#Licensed under GNU General Public License v3.0 (see http://www.bioinf.jku.at/research/lsc/LICENSE and https://github.com/ml-jku/lsc/blob/master/LICENSE)

import itertools
import numpy as np
import pandas as pd
import scipy
import scipy.io
import scipy.sparse
import pickle
from sklearn.feature_selection import VarianceThreshold



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
targetMatTransposed=targetMat[sampleAnnInd[list(itertools.chain(*folds))]].T.tocsr()
targetMatTransposed.sort_indices()
trainPosOverall=np.array([np.sum(targetMatTransposed[x].data > 0.5) for x in range(targetMatTransposed.shape[0])])
trainNegOverall=np.array([np.sum(targetMatTransposed[x].data < -0.5) for x in range(targetMatTransposed.shape[0])])



#denseOutputData=targetMat.A
denseOutputData=None
sparseOutputData=targetMat



if datasetName=="static":
  f=open(dataPathSave+'static.pckl', "rb")
  staticMat=pickle.load(f)
  sampleStaticInd=pickle.load(f)
  # featureStaticInd=pickle.load(f)
  f.close()
  
  denseInputData=staticMat
  denseSampleIndex=sampleStaticInd
  sparseInputData=None
  sparseSampleIndex=None
  
  del staticMat
  del sampleStaticInd
elif datasetName=="semi":
  f=open(dataPathSave+'semi.pckl', "rb")
  semiMat=pickle.load(f)
  sampleSemiInd=pickle.load(f)
  # featureSemiInd=pickle.load(f)
  f.close()

  denseInputData=semiMat.A
  denseSampleIndex=sampleSemiInd
  sparseInputData=None
  sparseSampleIndex=None
  
  del semiMat
  del sampleSemiInd
elif datasetName=="ecfp":
  f=open(dataPathSave+'ecfp6.pckl', "rb")
  ecfpMat=pickle.load(f)
  sampleECFPInd=pickle.load(f)
  # featureECFPInd=pickle.load(f)
  f.close()

  denseInputData=None
  denseSampleIndex=None
  sparseInputData=ecfpMat
  sparseSampleIndex=sampleECFPInd
  sparseInputData.eliminate_zeros()
  sparseInputData=sparseInputData.tocsr()
  sparseInputData.sort_indices()
  
  del ecfpMat
  del sampleECFPInd
  
  sparsenesThr=0.0025
elif datasetName=="dfs":
  f=open(dataPathSave+'dfs8.pckl', "rb")
  dfsMat=pickle.load(f)
  sampleDFSInd=pickle.load(f)
  # featureDFSInd=pickle.load(f)
  f.close()

  denseInputData=None
  denseSampleIndex=None
  sparseInputData=dfsMat
  sparseSampleIndex=sampleDFSInd
  sparseInputData.eliminate_zeros()
  sparseInputData=sparseInputData.tocsr()
  sparseInputData.sort_indices()
  
  del dfsMat
  del sampleDFSInd
  
  sparsenesThr=0.02
elif datasetName=="ecfpTox":
  f=open(dataPathSave+'ecfp6.pckl', "rb")
  ecfpMat=pickle.load(f)
  sampleECFPInd=pickle.load(f)
  # featureECFPInd=pickle.load(f)
  f.close()
  
  f=open(dataPathSave+'tox.pckl', "rb")
  toxMat=pickle.load(f)
  sampleToxInd=pickle.load(f)
  # featureToxInd=pickle.load(f)
  f.close()

  denseInputData=None
  denseSampleIndex=None
  sparseInputData=scipy.sparse.hstack([ecfpMat, toxMat])
  sparseSampleIndex=sampleECFPInd
  sparseInputData.eliminate_zeros()
  sparseInputData=sparseInputData.tocsr()
  sparseInputData.sort_indices()
  
  del ecfpMat
  del sampleECFPInd
  del toxMat
  del sampleToxInd
  
  sparsenesThr=0.0025

gc.collect()



allSamples=np.array([], dtype=np.int64)
if not (denseInputData is None):
  allSamples=np.union1d(allSamples, denseSampleIndex.index.values)
if not (sparseInputData is None):
  allSamples=np.union1d(allSamples, sparseSampleIndex.index.values)
if not (denseInputData is None):
  allSamples=np.intersect1d(allSamples, denseSampleIndex.index.values)
if not (sparseInputData is None):
  allSamples=np.intersect1d(allSamples, sparseSampleIndex.index.values)
allSamples=allSamples.tolist()



if not (denseInputData is None):
  folds=[np.intersect1d(fold, denseSampleIndex.index.values).tolist() for fold in folds]
if not (sparseInputData is None):
  folds=[np.intersect1d(fold, sparseSampleIndex.index.values).tolist() for fold in folds]