#Copyright (C) 2018 Andreas Mayr
#Licensed under GNU General Public License v3.0 (see http://www.bioinf.jku.at/research/lsc/LICENSE and https://github.com/ml-jku/lsc/blob/master/LICENSE)

import h5py
import numpy as np
import pickle
import os



datasetName="ecfpTox"
saveBasePath=os.getenv("HOME")+"/mydata/trgpred/chembl20/resPython/"
savePath=saveBasePath+datasetName+"/"

saveFilename=savePath+"o0001.evalPredict.hdf5"
saveFile=h5py.File(saveFilename, "r")
fold1=saveFile["predictions"][:]
saveFile.close()

saveFilename=savePath+"o0002.evalPredict.hdf5"
saveFile=h5py.File(saveFilename, "r")
fold2=saveFile["predictions"][:]
saveFile.close()

saveFilename=savePath+"o0003.evalPredict.hdf5"
saveFile=h5py.File(saveFilename, "r")
fold3=saveFile["predictions"][:]
saveFile.close()

fold1Ind=np.loadtxt(savePath+"o0001.eval.cmpNames", dtype=np.int64)
fold2Ind=np.loadtxt(savePath+"o0002.eval.cmpNames", dtype=np.int64)
fold3Ind=np.loadtxt(savePath+"o0003.eval.cmpNames", dtype=np.int64)
np.all(np.sort(np.hstack([fold1Ind,fold2Ind,fold3Ind]))==np.arange(456331))
targetNames=np.loadtxt(savePath+"o0001.eval.targetNames", dtype=np.str)

fullMat=np.zeros((456331, 1310))
fullMat[fold1Ind,:]=fold1
fullMat[fold2Ind,:]=fold2
fullMat[fold3Ind,:]=fold3

f=open(os.getenv("HOME")+"/mydata/trgpred/chembl20/dataPythonReduced/samples.pckl", "rb")
tpSampleIdToName=pickle.load(f)
f.close()

saveFilename=os.getenv("HOME")+"/mydata/trgpred/chembl20/results/allRes/"+datasetName+".h5"
saveFile=h5py.File(saveFilename, "w")
saveFile.create_dataset('mymatrix', data=fullMat)
saveFile.create_dataset('samples', data=[np.string_(x) for x in tpSampleIdToName], dtype=h5py.special_dtype(vlen=str))
saveFile.create_dataset('targets', data=[np.string_(x) for x in targetNames], dtype=h5py.special_dtype(vlen=str))
saveFile.close()

