#Copyright (C) 2018 Andreas Mayr
#Licensed under GNU General Public License v3.0 (see http://www.bioinf.jku.at/research/lsc/LICENSE and https://github.com/ml-jku/lsc/blob/master/LICENSE)

import sys

trainLSTMOutput=np.hstack([(trainDenseOutput+1)/2.0, trainLSTMSideOutput])



predDenseTrain=[]
idxSamplesEval=[arr[1] for arr in sklearn.model_selection.KFold(n_splits=int(math.ceil(len(trainSamples)/batchSize)), shuffle=False).split(np.arange(len(trainSamples)))]
for j in range(len(idxSamplesEval)):
  batchX=np.array([myOneHot(x, oneHot, otherInd, pad_len=seq_length) for x in trainSmilesLSTMInput[idxSamplesEval[j]]])
  batchY=trainLSTMOutput[idxSamplesEval[j]]
  predDenseTrain.append(model.predict_on_batch(batchX))

predDenseTrain=np.vstack(predDenseTrain)
if compPerformanceTrain:
  sumTrainAUC=np.array(utilsLib.calculateAUCs(trainDenseOutput, predDenseTrain[:,0:nrOutputTargets]))
  sumTrainAUCCheck=np.array(utilsLib.calculateAUCs(trainLSTMSideOutput*2-1, predDenseTrain[:,nrOutputTargets:]))
  sumTrainAP=np.array(utilsLib.calculateAPs(trainDenseOutput, predDenseTrain[:,0:nrOutputTargets]))



predDenseTest=[]
idxSamplesEval=[arr[1] for arr in sklearn.model_selection.KFold(n_splits=int(math.ceil(len(testSamples)/batchSize)), shuffle=False).split(np.arange(len(testSamples)))]
for j in range(len(idxSamplesEval)):
  batchX=np.array([myOneHot(x, oneHot, otherInd, pad_len=seq_length) for x in testSmilesLSTMInput[idxSamplesEval[j]]])
  predDenseTest.append(model.predict_on_batch(batchX))

predDenseTest=np.vstack(predDenseTest)
if compPerformanceTest:
  sumTestAUC=np.array(utilsLib.calculateAUCs(testDenseOutput, predDenseTest[:,0:nrOutputTargets]))
  sumTestAUCCheck=np.array(utilsLib.calculateAUCs(testLSTMSideOutput*2-1, predDenseTest[:,nrOutputTargets:]))
  sumTestAP=np.array(utilsLib.calculateAPs(testDenseOutput, predDenseTest[:,0:nrOutputTargets]))



print("\n", file=dbgOutput)
print("Train Mean AUC: ", file=dbgOutput)
print(np.nanmean(sumTrainAUC), file=dbgOutput)
print("\n", file=dbgOutput)

print("\n", file=dbgOutput)
print("Test Mean AUC: ", file=dbgOutput)
print(np.nanmean(sumTestAUC), file=dbgOutput)
print("\n", file=dbgOutput)
