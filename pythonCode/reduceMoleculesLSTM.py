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
import rdkit
import rdkit.Chem
import deepchem
import deepchem.feat

dataPathSave=os.getenv("HOME")+"/mydata/trgpred/chembl20/dataPython/"
dataPathSaveNew=os.getenv("HOME")+"/mydata/trgpred/chembl20/dataPythonReduced/"

f=open(dataPathSave+'labelsHard.pckl', "rb")
targetMat=pickle.load(f)
sampleAnnInd=pickle.load(f)
targetAnnInd=pickle.load(f)
f.close()

f=open(dataPathSave+"chembl20LSTM.pckl",'rb')
chemblMolsArr=pickle.load(f)
f.close()



take=(targetMat.getnnz(1)>0.5)



f=open(dataPathSaveNew+'chembl20LSTM.pckl', "wb")
myarr=chemblMolsArr[take].copy()
pickle.dump(myarr, f)
f.close()
