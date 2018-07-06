#Copyright (C) 2018 Andreas Mayr
#Licensed under GNU General Public License v3.0 (see http://www.bioinf.jku.at/research/lsc/LICENSE and https://github.com/ml-jku/lsc/blob/master/LICENSE)

import pickle
import numpy as np
import pandas as pd
import itertools
import os
import shutil

np.random.seed(54321)


dataPathSave=os.getenv("HOME")+"/mydata/trgpred/chembl20/dataPythonReduced/"
dataPathSaveNew=os.getenv("HOME")+"/mydata/trgpred/chembl20/dataPythonPermReduced/"
if not os.path.exists(dataPathSaveNew):
  os.makedirs(dataPathSaveNew)



f=open(dataPathSave+'folds0.pckl', "rb")
folds=pickle.load(f)
f.close()

f=open(dataPathSave+'labelsHard.pckl', "rb")
targetMat=pickle.load(f)
sampleAnnInd=pickle.load(f)
f.close()

targetMat=targetMat.copy().tocsr()
targetMat.sort_indices()

folds=[np.intersect1d(fold, sampleAnnInd.index.values).tolist() for fold in folds]



f1=targetMat[sampleAnnInd[folds[0]].values]
f1Pos=np.sum(f1.A > 0.5, 0)
f1Neg=np.sum(f1.A < -0.5, 0)

f2=targetMat[sampleAnnInd[folds[1]].values]
f2Pos=np.sum(f2.A > 0.5, 0)
f2Neg=np.sum(f2.A < -0.5, 0)

f3=targetMat[sampleAnnInd[folds[2]].values]
f3Pos=np.sum(f3.A > 0.5, 0)
f3Neg=np.sum(f3.A < -0.5, 0)



myarr=np.array([f1Pos, f1Neg, f2Pos, f2Neg, f3Pos, f3Neg]).T
myarr.sort(axis=1)
sortOrder=np.lexsort(np.flip(myarr, axis=1).T)



foldVec=np.zeros(len(folds[0]+folds[1]+folds[2]), dtype=np.int64)
foldVec[sampleAnnInd[folds[0]].values]=0
foldVec[sampleAnnInd[folds[1]].values]=1
foldVec[sampleAnnInd[folds[2]].values]=2

foldNewVec=np.repeat(-1, len(foldVec))
permMark=np.repeat(False, len(foldVec))



for tInd in sortOrder:
  mylabels=targetMat[sampleAnnInd.values, tInd].A.reshape(-1)

  mylabelsToAssign0=mylabels[foldVec==0]
  mylabelsToAssign1=mylabels[foldVec==1]
  mylabelsToAssign2=mylabels[foldVec==2]

  mylabelsAssigned0=mylabels[np.logical_and(foldNewVec==0, permMark)]
  mylabelsAssigned1=mylabels[np.logical_and(foldNewVec==1, permMark)]
  mylabelsAssigned2=mylabels[np.logical_and(foldNewVec==2, permMark)]

  nrLabels0Pos=max(np.sum(mylabelsToAssign0 > 0.5)-np.sum(mylabelsAssigned0 > 0.5), 0)
  nrLabels1Pos=max(np.sum(mylabelsToAssign1 > 0.5)-np.sum(mylabelsAssigned1 > 0.5), 0)
  nrLabels2Pos=max(np.sum(mylabelsToAssign2 > 0.5)-np.sum(mylabelsAssigned2 > 0.5), 0)
  biggerZero=np.where(np.logical_and(mylabels > 0.5, np.logical_not(permMark)))[0]

  nrLabels0Neg=max(np.sum(mylabelsToAssign0 < -0.5)-np.sum(mylabelsAssigned0 < -0.5), 0)
  nrLabels1Neg=max(np.sum(mylabelsToAssign1 < -0.5)-np.sum(mylabelsAssigned1 < -0.5), 0)
  nrLabels2Neg=max(np.sum(mylabelsToAssign2 < -0.5)-np.sum(mylabelsAssigned2 < -0.5), 0)
  smallerZero=np.where(np.logical_and(mylabels < -0.5, np.logical_not(permMark)))[0]



  nrPosLab=[0, nrLabels0Pos, nrLabels1Pos, nrLabels2Pos]
  nrNegLab=[0, nrLabels0Neg, nrLabels1Neg, nrLabels2Neg]
  posCum=np.cumsum(nrPosLab)
  negCum=np.cumsum(nrNegLab)

  if np.max(posCum)>0.5:
    posCum=((posCum.astype(np.float)/float(np.max(posCum)))*float(len(biggerZero))+0.5).astype(np.integer)
  if np.max(negCum)>0.5:
    negCum=((negCum.astype(np.float)/float(np.max(negCum)))*float(len(smallerZero))+0.5).astype(np.integer)



  permBigger=np.random.permutation(biggerZero)
  permSmaller=np.random.permutation(smallerZero)

  p0=permBigger[posCum[0]:posCum[1]]
  p1=permBigger[posCum[1]:posCum[2]]
  p2=permBigger[posCum[2]:posCum[3]]

  n0=permSmaller[negCum[0]:negCum[1]]
  n1=permSmaller[negCum[1]:negCum[2]]
  n2=permSmaller[negCum[2]:negCum[3]]

  foldNewVec[p0]=0
  foldNewVec[p1]=1
  foldNewVec[p2]=2
  foldNewVec[n0]=0
  foldNewVec[n1]=1
  foldNewVec[n2]=2
  permMark[permBigger]=True
  permMark[permSmaller]=True

newFolds=[np.sort(sampleAnnInd.index.values[foldNewVec==0]), np.sort(sampleAnnInd.index.values[foldNewVec==1]), np.sort(sampleAnnInd.index.values[foldNewVec==2])]




newFoldsFile=dataPathSaveNew+"folds0.pckl"
shutil.copytree(dataPathSave, dataPathSaveNew)
os.remove(newFoldsFile)

f=open(newFoldsFile, "wb")
pickle.dump(newFolds, f, -1)
f.close()
