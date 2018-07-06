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
import pickle
import imp
import os
import sys
import time
import gc

dataPathSave=os.getenv("HOME")+"/mydata/trgpred/chembl20/dataPython/"
dataPathSaveNew=os.getenv("HOME")+"/mydata/trgpred/chembl20/dataPythonReduced/"

f=open(dataPathSave+'folds0.pckl', "rb")
folds=pickle.load(f)
f.close()

f=open(dataPathSave+'labelsHard.pckl', "rb")
targetMat=pickle.load(f)
sampleAnnInd=pickle.load(f)
targetAnnInd=pickle.load(f)
f.close()

f=open(dataPathSave+'labelsWeakHard.pckl', "rb")
targetMatWeak=pickle.load(f)
sampleAnnIndWeak=pickle.load(f)
targetAnnIndWeak=pickle.load(f)
f.close()

f=open(dataPathSave+'static.pckl', "rb")
staticMat=pickle.load(f)
sampleStaticInd=pickle.load(f)
# featureStaticInd=pickle.load(f)
f.close()

f=open(dataPathSave+'semi.pckl', "rb")
semiMat=pickle.load(f)
sampleSemiInd=pickle.load(f)
# featureSemiInd=pickle.load(f)
f.close()

f=open(dataPathSave+'ecfp6.pckl', "rb")
ecfpMat=pickle.load(f)
sampleECFPInd=pickle.load(f)
# featureECFPInd=pickle.load(f)
f.close()

f=open(dataPathSave+'dfs8.pckl', "rb")
dfsMat=pickle.load(f)
sampleDFSInd=pickle.load(f)
# featureDFSInd=pickle.load(f)
f.close()

f=open(dataPathSave+'tox.pckl', "rb")
toxMat=pickle.load(f)
sampleToxInd=pickle.load(f)
# featureToxInd=pickle.load(f)
f.close()



take=(targetMat.getnnz(1)>0.5)
newNr=-np.ones(len(take), dtype=np.int64)
myNr=0
for i in range(len(newNr)):
  if take[i]:
    newNr[i]=myNr
    myNr=myNr+1


folds=[np.sort(newNr[folds[0][take[folds[0]]]]), np.sort(newNr[folds[1][take[folds[1]]]]), np.sort(newNr[folds[2][take[folds[2]]]])]
f=open(dataPathSaveNew+'folds0.pckl', "wb")
pickle.dump(folds, f, -1)
f.close()

f=open(dataPathSaveNew+'labelsHard.pckl', "wb")
mymat=targetMat[take].copy()
mymat.sort_indices()
pickle.dump(mymat, f, -1)
pickle.dump(pd.Series(data=np.arange(take.sum()), index=np.arange(take.sum())), f, -1)
pickle.dump(targetAnnInd, f, -1)
f.close()

f=open(dataPathSaveNew+'static.pckl', "wb")
mymat=staticMat[take].copy()
pickle.dump(mymat, f, -1)
pickle.dump(pd.Series(data=np.arange(take.sum()), index=np.arange(take.sum())), f, -1)
f.close()

f=open(dataPathSaveNew+'semi.pckl', "wb")
mymat=semiMat[take].copy()
mymat.sort_indices()
pickle.dump(mymat, f, -1)
pickle.dump(pd.Series(data=np.arange(take.sum()), index=np.arange(take.sum())), f, -1)
f.close()

f=open(dataPathSaveNew+'ecfp6.pckl', "wb")
mymat=ecfpMat[take].copy()
mymat.sort_indices()
pickle.dump(mymat, f, -1)
pickle.dump(pd.Series(data=np.arange(take.sum()), index=np.arange(take.sum())), f, -1)
f.close()

f=open(dataPathSaveNew+'dfs8.pckl', "wb")
mymat=dfsMat[take].copy()
mymat.sort_indices()
pickle.dump(mymat, f, -1)
pickle.dump(pd.Series(data=np.arange(take.sum()), index=np.arange(take.sum())), f, -1)
f.close()

f=open(dataPathSaveNew+'tox.pckl', "wb")
mymat=toxMat[take].copy()
mymat.sort_indices()
pickle.dump(mymat, f, -1)
pickle.dump(pd.Series(data=np.arange(take.sum()), index=np.arange(take.sum())), f, -1)
f.close()



projectPathname=os.getenv("HOME")+"/mydata/trgpred/chembl20"
chemPathname=os.path.join(projectPathname, "chemFeatures");
sampleIdFilename=os.path.join(chemPathname, "SampleIdTable.txt")
tpSampleIdToName=np.genfromtxt(sampleIdFilename, dtype=str)
tpSampleIdToName=tpSampleIdToName[take]
f=open(dataPathSaveNew+'samples.pckl', "wb")
pickle.dump(tpSampleIdToName, f, -1)
f.close()



f=open(dataPathSaveNew+'labelsWeakHard.pckl', "wb")
mymat=targetMatWeak[take].copy()
mymat.sort_indices()
pickle.dump(mymat, f, -1)
pickle.dump(pd.Series(data=np.arange(take.sum()), index=np.arange(take.sum())), f, -1)
pickle.dump(targetAnnIndWeak, f, -1)
f.close()

scipy.io.mmwrite(dataPathSaveNew+'labelsWeakHard.mtx', mymat)
np.savetxt(dataPathSaveNew+"labelsWeakHard.cmpNames", tpSampleIdToName, fmt="%s")
np.savetxt(dataPathSaveNew+"labelsWeakHard.targetNames", targetAnnIndWeak.index.values, fmt="%s")