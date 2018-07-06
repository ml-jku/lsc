#Copyright (C) 2018 Andreas Mayr
#Licensed under GNU General Public License v3.0 (see http://www.bioinf.jku.at/research/lsc/LICENSE and https://github.com/ml-jku/lsc/blob/master/LICENSE)

import math
import numpy as np
import pandas as pd
import scipy
import scipy.io
import scipy.sparse
import sklearn
import sklearn.feature_selection
import sklearn.model_selection
import sklearn.metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import sys
import argparse

#np.set_printoptions(threshold='nan')
np.set_printoptions(threshold=1000)
np.set_printoptions(linewidth=160)
np.set_printoptions(precision=4)
np.set_printoptions(edgeitems=15)
np.set_printoptions(suppress=True)
pd.set_option('display.width', 160)
pd.options.display.float_format = '{:.2f}'.format



parser = argparse.ArgumentParser()
parser.add_argument("-destPath", help="Dest Path for original data: python Format", type=str, default=os.getenv("HOME")+"/mydata/trgpred/chembl20/dataPython/")
args = parser.parse_args()




dataPathSave=args.destPath
if not os.path.exists(dataPathSave):
  os.makedirs(dataPathSave)



trainInfo="train";
clusterInfo="cl1";
#targetInfo="trg1";

targetSampleInfo=trainInfo
clusterInfo=clusterInfo
#predictTargetInfo=targetInfo



projectPathname=os.getenv("HOME")+"/mydata/trgpred/chembl20"
chemPathname=os.path.join(projectPathname, "chemFeatures");
clusterPathname=os.path.join(chemPathname, "cl");
dchemPathname=os.path.join(chemPathname, "d");
schemPathname=os.path.join(chemPathname, "s");
trainPathname=os.path.join(projectPathname, "train");
runPathname=os.path.join(projectPathname, "run");
sampleIdFilename=os.path.join(chemPathname, "SampleIdTable.txt")

targetSampleFilename=os.path.join(trainPathname, targetSampleInfo+".info")
clusterSampleFilename=os.path.join(clusterPathname, clusterInfo+".info")
#predictTargetFilename=os.path.join(trainPathname, predictTargetInfo+".info")



sampleIdFilename=os.path.join(chemPathname, 'SampleIdTable.txt')
tpSampleIdToName=np.genfromtxt(sampleIdFilename, dtype=str)



clusterTab=pd.read_csv(clusterSampleFilename, header=None, index_col=False, sep=" ")
folds=[clusterTab.loc[clusterTab.iloc[:,0]==0].iloc[:,1].values, clusterTab.loc[clusterTab.iloc[:,0]==1].iloc[:,1].values, clusterTab.loc[clusterTab.iloc[:,0]==2].iloc[:,1].values]
f=open(dataPathSave+'folds0.pckl', "wb")
pickle.dump(folds, f, -1)
f.close()
lookup=clusterTab.set_index(1)



trainTab=pd.read_csv(targetSampleFilename, header=None, index_col=False, sep=" ")
trainTab=trainTab.drop_duplicates()

classTypes=np.array([ 0,  1,  2,  3, 10, 11, 13])
clusterNames=np.sort(np.unique(lookup))
targetNames=np.sort(np.unique(trainTab.iloc[:,2].values))

classNumbers=[]
for clusterName in clusterNames:
  classNumbersCluster=[]
  for classType in classTypes:
    trainTabSel=trainTab.loc[(lookup.iloc[trainTab.iloc[:,1].values]==clusterName).values[:,0]]
    eq=trainTabSel.loc[trainTabSel.iloc[:,0]==classType].iloc[:,2].value_counts()
    ser=pd.Series(np.zeros(len(targetNames), dtype=np.int64))
    ser.index=targetNames
    ser[eq.index.values]=eq.values
    classNumbersCluster.append(ser)
  classNumbers.append(np.array(classNumbersCluster))
classNumbers=np.array(classNumbers)



ovTab=[classNumbers[0][np.where(classTypes==0)[0][0]], #0
classNumbers[1][np.where(classTypes==0)[0][0]], #1
classNumbers[2][np.where(classTypes==0)[0][0]], #2

classNumbers[0][np.where(classTypes==1)[0][0]], #3
classNumbers[1][np.where(classTypes==1)[0][0]], #4
classNumbers[2][np.where(classTypes==1)[0][0]], #5

classNumbers[0][np.where(classTypes==2)[0][0]], #6
classNumbers[1][np.where(classTypes==2)[0][0]], #7
classNumbers[2][np.where(classTypes==2)[0][0]], #8

classNumbers[0][np.where(classTypes==3)[0][0]], #9
classNumbers[1][np.where(classTypes==3)[0][0]], #10
classNumbers[2][np.where(classTypes==3)[0][0]], #11

classNumbers[0][np.where(classTypes==11)[0][0]], #12
classNumbers[1][np.where(classTypes==11)[0][0]], #13
classNumbers[2][np.where(classTypes==11)[0][0]], #14

classNumbers[0][np.where(classTypes==13)[0][0]], #15
classNumbers[1][np.where(classTypes==13)[0][0]], #16
classNumbers[2][np.where(classTypes==13)[0][0]]] #17

ovTab=np.array(ovTab).T
nrOfActivities=ovTab[:,[3,4,5,   9,10,11,   12,13,14,   15,16,17,   6,7,8]].sum(1)
hasInactives=(ovTab[:,[3,4,5]].sum(1)>0.5)
hasActives=(ovTab[:,[9,10,11]].sum(1)>0.5)
allFoldsHaveInActives=(np.sum((ovTab[:,[3,4,5]])>0.5, 1)>2.5)
allFoldsHaveActives=(np.sum((ovTab[:,[9,10,11]])>0.5, 1)>2.5)



selTargetNames=targetNames[np.logical_and((nrOfActivities>=100), np.logical_and(allFoldsHaveInActives, allFoldsHaveActives))]

trainTabSel=trainTab.loc[np.in1d(trainTab.iloc[:,0].values, [1,3])]
trainTabSel=trainTabSel.loc[np.in1d(trainTabSel.iloc[:,2].values, selTargetNames)]

sampleAnnInd=pd.Series(data=np.arange(len(tpSampleIdToName)), index=np.arange(len(tpSampleIdToName)))
targetAnnInd=pd.Series(data=np.arange(len(selTargetNames)), index=selTargetNames)
annMat=scipy.sparse.coo_matrix((trainTabSel.iloc[:,0], (sampleAnnInd[trainTabSel.iloc[:,1]], targetAnnInd[trainTabSel.iloc[:,2]])), shape=(sampleAnnInd.max()+1, targetAnnInd.max()+1))
annMat.data[annMat.data<2]=-1
annMat.data[annMat.data>2]=1
annMat.eliminate_zeros()
annMat=annMat.tocsr()
annMat.sort_indices()
f=open(dataPathSave+'labelsHard.pckl', "wb")
pickle.dump(annMat, f, -1)
pickle.dump(sampleAnnInd, f, -1)
pickle.dump(targetAnnInd, f, -1)
f.close()

trainTabSel=trainTab.loc[np.in1d(trainTab.iloc[:,0].values, [1,11,3,13])]
trainTabSel=trainTabSel.loc[np.in1d(trainTabSel.iloc[:,2].values, selTargetNames)]

sampleAnnInd=pd.Series(data=np.arange(len(tpSampleIdToName)), index=np.arange(len(tpSampleIdToName)))
targetAnnInd=pd.Series(data=np.arange(len(selTargetNames)), index=selTargetNames)
annMat=scipy.sparse.coo_matrix((trainTabSel.iloc[:,0], (sampleAnnInd[trainTabSel.iloc[:,1]], targetAnnInd[trainTabSel.iloc[:,2]])), shape=(sampleAnnInd.max()+1, targetAnnInd.max()+1))
annMat.eliminate_zeros()
annMat=annMat.tocsr()
annMat.sort_indices()
f=open(dataPathSave+'labelsWeakHard.pckl', "wb")
pickle.dump(annMat, f, -1)
pickle.dump(sampleAnnInd, f, -1)
pickle.dump(targetAnnInd, f, -1)
f.close()

scipy.io.mmwrite(dataPathSave+'labelsWeakHard.mtx', annMat)
np.savetxt(dataPathSave+"labelsWeakHard.cmpNames", tpSampleIdToName[sampleAnnInd.index.values], fmt="%s")
np.savetxt(dataPathSave+"labelsWeakHard.targetNames", targetAnnInd.index.values, fmt="%s")












def readSFeature(schemPathname, sfeature):
  featureIntFilename=os.path.join(schemPathname, sfeature, 'fpFeatureTableInt.bin')
  featureExtFilename=os.path.join(schemPathname, sfeature, 'fpFeatureTableExt.bin')
  intToExtMappingFilename=os.path.join(schemPathname, sfeature, 'IntToExtMappingTable.bin')
  sampleFilename=os.path.join(schemPathname, sfeature, 'fpSampleTable.bin')
  featureCountFilename=os.path.join(schemPathname, sfeature, 'fpFeatureCountTableD.bin')
  
  featuresInt=np.fromfile(featureIntFilename, dtype=np.long, count=-1)
  featuresExt=np.fromfile(featureExtFilename, dtype=np.long, count=-1)
  featureMap=np.fromfile(intToExtMappingFilename, dtype=np.long, count=-1)
  samples=np.fromfile(sampleFilename, dtype=np.long, count=-1)
  samples=np.append(samples, featuresInt.size)
  featureCounts=np.fromfile(featureCountFilename, dtype=np.float64, count=-1)
  
  if (np.all(featureMap[featuresInt]==featuresExt))==False:
    print("An error occured!")
    return []
  
  smat=scipy.sparse.csr_matrix((featureCounts, featuresInt, samples))
  sampleInd=pd.Series(data=np.arange(smat.shape[0]), index=np.arange(smat.shape[0]))
  #featureInd=pd.Series(data=np.arange(smat.shape[1]), index=np.arange(smat.shape[1]))
  featureInd=pd.Series([sfeature+":"]*len(featureMap))+featureMap.astype(np.str)
  return (smat, sampleInd, featureInd)

def readDFeature(dchemPathname, dfeature):
  propertyFilename = os.path.join(dchemPathname, dfeature, 'properties.bin')
  propertyIdFilename = os.path.join(dchemPathname, dfeature, 'PropertyIdTable.txt')
  
  dmat=np.fromfile(propertyFilename, dtype=np.float64, count=-1)
  propertyIndexToPropertyId=np.genfromtxt(propertyIdFilename, dtype=str)
  dmat=dmat.reshape(-1, len(propertyIndexToPropertyId))
  
  sampleInd=pd.Series(data=np.arange(dmat.shape[0]), index=np.arange(dmat.shape[0]))
  #featureInd=pd.Series(data=np.arange(dmat.shape[1]), index=np.arange(dmat.shape[1]))
  featureInd=pd.Series([dfeature+":"]*len(propertyIndexToPropertyId))+propertyIndexToPropertyId
  
  return (dmat, sampleInd, featureInd)



dMat, dMatSampleInd, dMatSFeatureInd=readDFeature(dchemPathname, "dense")
f=open(dataPathSave+'static.pckl', "wb")
pickle.dump(dMat, f, -1)
pickle.dump(dMatSampleInd, f, -1)
pickle.dump(dMatSFeatureInd, f, -1)
f.close()

sMat, sMatSampleInd, sMatFeatureInd=readSFeature(schemPathname, "semisparse")
f=open(dataPathSave+'semi.pckl', "wb")
pickle.dump(sMat, f, -1)
pickle.dump(sMatSampleInd, f, -1)
pickle.dump(sMatFeatureInd, f, -1)
f.close()

sMat, sMatSampleInd, sMatFeatureInd=readSFeature(schemPathname, "ECFC6_ES")
f=open(dataPathSave+'ecfp6.pckl', "wb")
pickle.dump(sMat, f, -1)
pickle.dump(sMatSampleInd, f, -1)
pickle.dump(sMatFeatureInd, f, -1)
f.close()

sMat, sMatSampleInd, sMatFeatureInd=readSFeature(schemPathname, "DFS8_ES")
f=open(dataPathSave+'dfs8.pckl', "wb")
pickle.dump(sMat, f, -1)
pickle.dump(sMatSampleInd, f, -1)
pickle.dump(sMatFeatureInd, f, -1)
f.close()

sMat, sMatSampleInd, sMatFeatureInd=readSFeature(schemPathname, "toxicophores")
f=open(dataPathSave+'tox.pckl', "wb")
pickle.dump(sMat, f, -1)
pickle.dump(sMatSampleInd, f, -1)
pickle.dump(sMatFeatureInd, f, -1)
f.close()


