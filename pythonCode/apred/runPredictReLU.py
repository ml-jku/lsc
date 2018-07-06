#Copyright (C) 2018 Andreas Mayr
#Licensed under GNU General Public License v3.0 (see http://www.bioinf.jku.at/research/lsc/LICENSE and https://github.com/ml-jku/lsc/blob/master/LICENSE)

if nrSparseFeatures>0.5:
  session.run(sparseMeanWSparseOp.op)



session.run(scalePredict, feed_dict={ inputDropout: currentIDropout, hiddenDropout: currentDropout })



predDenseTrain=[]
if not (sparseOutputData is None):
  predSparseTrain=trainSparseOutput.copy().astype(np.float32)
  predSparseTrain.data[:]=-1

idxSamplesEval=[arr[1] for arr in sklearn.model_selection.KFold(n_splits=int(math.ceil(len(trainSamples)/batchSize)), shuffle=False).split(np.arange(len(trainSamples)))]
for j in range(len(idxSamplesEval)):
  
  myfeedDict={
    inputDropout: 0.0,
    hiddenDropout: 0.0,
    is_training: False
  }
  
  if nrDenseFeatures>0.5:
    batchDenseX=trainDenseInput[idxSamplesEval[j]]
    myfeedDict.update({
      xDenseData: batchDenseX
    })
  
  if nrSparseFeatures>0.5:
    batchSparseX=trainSparseInput[idxSamplesEval[j],:].copy()
    batchSparseX.sort_indices()
    nonzx=batchSparseX.nonzero()
    valnonzx=(batchSparseX)[nonzx[0],nonzx[1]]
    myfeedDict.update({
      xIndices: np.int64(np.vstack(nonzx).T), 
      xValues: valnonzx.A.flatten(), 
      xDim: [len(idxSamplesEval[j]), batchSparseX.shape[1]]
    })
  
  if useDenseOutputNetPred:
    if compPerformanceTrain:
      batchDenseY=trainDenseOutput[idxSamplesEval[j]]
    predDenseTrain.append(session.run(predNetwork, feed_dict=myfeedDict))
  
  if not useDenseOutputNetPred:
    if compPerformanceTrain:
      batchSparseY=trainSparseOutput[idxSamplesEval[j]].copy()
      batchSparseY.sort_indices()
      nonzy=batchSparseY.nonzero()
      valnonzy=(batchSparseY)[nonzy[0],nonzy[1]]
      myfeedDict.update({
        yIndices: np.int64(np.vstack(nonzy).T)
      })
      predSparseTrain[min(idxSamplesEval[j])+nonzy[0],nonzy[1]]=session.run(predNetworkSparse, feed_dict=myfeedDict)

if useDenseOutputNetPred:
  predDenseTrain=np.vstack(predDenseTrain)
  if compPerformanceTrain:
    sumTrainAUC=np.array(utilsLib.calculateAUCs(trainDenseOutput, predDenseTrain))
    sumTrainAP=np.array(utilsLib.calculateAPs(trainDenseOutput, predDenseTrain))

if not useDenseOutputNetPred:
  if compPerformanceTrain:
    predSparseTrainTransposed=predSparseTrain.T.tocsr()
    predSparseTrain=None
    predSparseTrainTransposed.sort_indices()
    sumTrainAUC=np.array(utilsLib.calculateSparseAUCs(trainSparseOutputTransposed, predSparseTrainTransposed))
    sumTrainAP=np.array(utilsLib.calculateSparseAPs(trainSparseOutputTransposed, predSparseTrainTransposed))
    predSparseTrainTransposed=None



predDenseTest=[]
if not (sparseOutputData is None):
  predSparseTest=testSparseOutput.copy().astype(np.float32)
  predSparseTest.data[:]=-1

idxSamplesEval=[arr[1] for arr in sklearn.model_selection.KFold(n_splits=int(math.ceil(len(testSamples)/batchSize)), shuffle=False).split(np.arange(len(testSamples)))]
for j in range(len(idxSamplesEval)):
  
  myfeedDict={
    inputDropout: 0.0,
    hiddenDropout: 0.0,
    is_training: False
  }
  
  if nrDenseFeatures>0.5:
    batchDenseX=testDenseInput[idxSamplesEval[j]]
    myfeedDict.update({
      xDenseData: batchDenseX
    })
  
  if nrSparseFeatures>0.5:
    batchSparseX=testSparseInput[idxSamplesEval[j],:].copy()
    batchSparseX.sort_indices()
    nonzx=batchSparseX.nonzero()
    valnonzx=(batchSparseX)[nonzx[0],nonzx[1]]
    myfeedDict.update({
      xIndices: np.int64(np.vstack(nonzx).T), 
      xValues: valnonzx.A.flatten(), 
      xDim: [len(idxSamplesEval[j]), batchSparseX.shape[1]]
    })
  
  predDenseTest.append(session.run(predNetwork, feed_dict=myfeedDict))
  
  if not useDenseOutputNetPred:
    if compPerformanceTest:
      batchSparseY=testSparseOutput[idxSamplesEval[j]].copy()
      batchSparseY.sort_indices()
      nonzy=batchSparseY.nonzero()
      valnonzy=(batchSparseY)[nonzy[0],nonzy[1]]
      myfeedDict.update({
        yIndices: np.int64(np.vstack(nonzy).T)
      })
      predSparseTest[min(idxSamplesEval[j])+nonzy[0],nonzy[1]]=session.run(predNetworkSparse, feed_dict=myfeedDict)

predDenseTest=np.vstack(predDenseTest)
if useDenseOutputNetPred:
  if compPerformanceTest:
    sumTestAUC=np.array(utilsLib.calculateAUCs(testDenseOutput, predDenseTest))
    sumTestAP=np.array(utilsLib.calculateAPs(testDenseOutput, predDenseTest))

if not useDenseOutputNetPred:
  if compPerformanceTest:
    predSparseTestTransposed=predSparseTest.copy().T.tocsr()
    predSparseTest=None
    predSparseTestTransposed.sort_indices()
    sumTestAUC=np.array(utilsLib.calculateSparseAUCs(testSparseOutputTransposed, predSparseTestTransposed))
    sumTestAP=np.array(utilsLib.calculateSparseAPs(testSparseOutputTransposed, predSparseTestTransposed))
    predSparseTestTransposed=None



session.run(scaleTrain, feed_dict={ inputDropout: currentIDropout, hiddenDropout: currentDropout })



print("\n", file=dbgOutput)
print("Train Mean AUC: ", file=dbgOutput)
print(np.nanmean(sumTrainAUC), file=dbgOutput)
print("\n", file=dbgOutput)

print("\n", file=dbgOutput)
print("Test Mean AUC: ", file=dbgOutput)
print(np.nanmean(sumTestAUC), file=dbgOutput)
print("\n", file=dbgOutput)