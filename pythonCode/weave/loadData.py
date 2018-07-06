#Copyright (C) 2018 Andreas Mayr
#Licensed under GNU General Public License v3.0 (see http://www.bioinf.jku.at/research/lsc/LICENSE and https://github.com/ml-jku/lsc/blob/master/LICENSE)

import itertools
import numpy as np
import pandas as pd
import scipy
import scipy.io
import scipy.sparse
import pickle
import rdkit
import rdkit.Chem
import deepchem
import deepchem.feat
import pickle



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



denseOutputData=targetMat.A
sparseOutputData=None
#denseOutputData=None
#sparseOutputData=targetMat



if datasetName=="graphWeave":
  f=open(dataPathSave+"chembl20Weave.pckl",'rb')
  mychemblConvertedMols=pickle.load(f)
  chemblMolsArr=np.arange(len(mychemblConvertedMols))
  f.close()
  
  denseInputData=None
  denseSampleIndex=None
  sparseInputData=None
  sparseSampleIndex=None
  graphInputData=chemblMolsArr
  graphSampleIndex=sampleAnnInd



gc.collect()



allSamples=np.array([], dtype=np.int64)
if not (graphInputData is None):
  allSamples=np.union1d(allSamples, graphSampleIndex.index.values)
if not (graphInputData is None):
  allSamples=np.intersect1d(allSamples, graphSampleIndex.index.values)
allSamples=allSamples.tolist()



if not (graphSampleIndex is None):
  folds=[np.intersect1d(fold, graphSampleIndex.index.values).tolist() for fold in folds]