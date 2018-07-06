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
import rdkit.Chem.MACCSkeys
import pickle
import os

dataPathSave=os.getenv("HOME")+"/mydata/trgpred/chembl20/dataPythonReduced/"

f=open(dataPathSave+"chembl20LSTM.pckl",'rb')
chemblMolsArr=pickle.load(f)
f.close()



chemblSmilesArr=[]
for ind in range(len(chemblMolsArr)):
  if chemblMolsArr[ind] is not None:
    chemblSmilesArr.append(rdkit.Chem.MolToSmiles(chemblMolsArr[ind]))
  else:
    chemblSmilesArr.append(rdkit.Chem.MolToSmiles(rdkit.Chem.MolFromSmiles("")))

f=open(dataPathSave+'chembl20Smiles.pckl', "wb")
pickle.dump(chemblSmilesArr, f)
f.close()



chemblMACCSArr=[]
for ind in range(len(chemblMolsArr)):
  if chemblMolsArr[ind] is not None:
    chemblMACCSArr.append(rdkit.Chem.MACCSkeys.GenMACCSKeys(chemblMolsArr[ind]))
  else:
    chemblMACCSArr.append(np.zeros(167, dtype=np.int64))
chemblMACCSMat=np.array(chemblMACCSArr)

f=open(dataPathSave+'chembl20MACCS.pckl', "wb")
pickle.dump(chemblMACCSMat, f)
f.close()
