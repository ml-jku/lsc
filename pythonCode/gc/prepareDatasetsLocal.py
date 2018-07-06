#Copyright (C) 2018 Andreas Mayr
#Licensed under GNU General Public License v3.0 (see http://www.bioinf.jku.at/research/lsc/LICENSE and https://github.com/ml-jku/lsc/blob/master/LICENSE)

if not (denseOutputData is None):
  trainDenseOutput=denseOutputData[sampleAnnInd[trainSamples].values].copy()
  testDenseOutput=denseOutputData[sampleAnnInd[testSamples].values].copy()
  trainPos=(trainDenseOutput > 0.5).sum(axis=0)
  trainNeg=(trainDenseOutput < -0.5).sum(axis=0)

if not (sparseOutputData is None):
  trainSparseOutput=sparseOutputData[sampleAnnInd[trainSamples].values].copy()
  trainSparseOutputTransposed=trainSparseOutput.copy().T.tocsr()
  trainSparseOutputTransposed.sort_indices()
  testSparseOutput=sparseOutputData[sampleAnnInd[testSamples].values].copy()
  testSparseOutputTransposed=testSparseOutput.copy().T.tocsr()
  testSparseOutputTransposed.sort_indices()
  trainPos=np.array([np.sum(trainSparseOutputTransposed[x].data > 0.5) for x in range(trainSparseOutputTransposed.shape[0])])
  trainNeg=np.array([np.sum(trainSparseOutputTransposed[x].data < -0.5) for x in range(trainSparseOutputTransposed.shape[0])])

trainProp=trainPos/(trainPos+trainNeg)
trainBias=np.log(trainProp/(1.0-trainProp))
trainBias[np.logical_not(np.logical_and(trainPos>10, trainNeg>10))]=0.0



if savePredictionsAtBestIter:
  if computeTestPredictions:
    if useDenseOutputNetPred:
      predDenseBestIter=-np.ones((len(testSamples), nrOutputTargets))
    else:
      predSparseBestIter=testSparseOutput.copy().astype(np.float32)
      predSparseBestIter.data[:]=-1

if logPerformanceAtBestIter:
  if computeTestPredictions:
    reportAUCBestIter=np.zeros(nrOutputTargets)
    reportAPBestIter=np.zeros(nrOutputTargets)



nrDenseFeatures=0
if not (denseInputData is None):
  trainDenseInput=denseInputData[denseSampleIndex[trainSamples].values].copy()
  testDenseInput=denseInputData[denseSampleIndex[testSamples].values].copy()
  nrDenseFeatures=trainDenseInput.shape[1]

  if normalizeLocalDense:
    trainDenseMean1=np.nanmean(trainDenseInput, 0)
    trainDenseStd1=np.nanstd(trainDenseInput, 0)+0.0001
    trainDenseInput=(trainDenseInput-trainDenseMean1)/trainDenseStd1
    trainDenseInput=np.tanh(trainDenseInput)
    trainDenseMean2=np.nanmean(trainDenseInput, 0)
    trainDenseStd2=np.nanstd(trainDenseInput, 0)+0.0001
    trainDenseInput=(trainDenseInput-trainDenseMean2)/trainDenseStd2
    
    testDenseInput=(testDenseInput-trainDenseMean1)/trainDenseStd1
    testDenseInput=np.tanh(testDenseInput)
    testDenseInput=(testDenseInput-trainDenseMean2)/trainDenseStd2
  
  trainDenseInput=np.nan_to_num(trainDenseInput)
  testDenseInput=np.nan_to_num(testDenseInput)



nrSparseFeatures=0
if not (sparseInputData is None):
  trainSparseInput=sparseInputData[sparseSampleIndex[trainSamples].values].copy()
  testSparseInput=sparseInputData[sparseSampleIndex[testSamples].values].copy()
  
  featSel=trainSparseInput.getnnz(axis=0)/float(trainSparseInput.shape[0])> sparsenesThr
  trainSparseInput=trainSparseInput[:, featSel].tocsr()
  trainSparseInput.sort_indices()
  testSparseInput=testSparseInput[:, featSel].tocsr()
  testSparseInput.sort_indices()
  nrSparseFeatures=np.sum(featSel)
  
  if not (normalizeGlobalSparse or normalizeLocalSparse):
    trainSparseDiv1=np.zeros(nrSparseFeatures)
    trainSparseDiv2=np.zeros(nrSparseFeatures)
  
  if normalizeGlobalSparse:
    trainSparseDiv1=sparseDiv1[featSel]
    trainSparseDiv2=sparseDiv2[featSel]
  
  if normalizeLocalSparse:
    trainSparseMean1=trainSparseInput.mean(0).A[0]
    trainSparseStd1=np.sqrt(trainSparseInput.multiply(trainSparseInput).mean(0).A[0]-trainSparseMean1**2)
    trainSparseInputNorm1=trainSparseInput.copy()
    trainSparseInputNorm1.data=np.tanh((trainSparseInputNorm1.data-trainSparseMean1[trainSparseInputNorm1.indices])/trainSparseStd1[trainSparseInputNorm1.indices])
    trainSparseDiv1=np.tanh(-trainSparseMean1/trainSparseStd1)
    trainSparseInputHelp1=trainSparseInputNorm1.copy()
    trainSparseInputHelp1.data=trainSparseInputHelp1.data-trainSparseDiv1[trainSparseInputHelp1.indices]
    trainSparseInputHelp2=trainSparseInputNorm1.copy()
    trainSparseInputHelp2.data=trainSparseInputHelp2.data**2-trainSparseDiv1[trainSparseInputHelp2.indices]**2
    trainSparseMean2=trainSparseInputHelp1.mean(0).A[0]+trainSparseDiv1
    trainSparseStd2=np.sqrt(trainSparseInputHelp2.mean(0).A[0]+trainSparseDiv1**2-trainSparseMean2**2)
    trainSparseInputNorm2=trainSparseInputNorm1.copy()
    trainSparseInputNorm2.data=(trainSparseInputNorm2.data-trainSparseMean2[trainSparseInputNorm2.indices])/trainSparseStd2[trainSparseInputNorm2.indices]
    trainSparseDiv2=(trainSparseDiv1-trainSparseMean2)/trainSparseStd2
    trainSparseInput=trainSparseInputNorm2.copy().tocsr()
    trainSparseInput.sort_indices()
    trainSparseInput.data=trainSparseInput.data-trainSparseDiv2[trainSparseInput.indices]
    
    testSparseInputNorm1=testSparseInput.copy()
    testSparseInputNorm1.data=np.tanh((testSparseInputNorm1.data-trainSparseMean1[testSparseInputNorm1.indices])/trainSparseStd1[testSparseInputNorm1.indices])
    testSparseInputNorm2=testSparseInputNorm1.copy()
    testSparseInputNorm2.data=(testSparseInputNorm2.data-trainSparseMean2[testSparseInputNorm2.indices])/trainSparseStd2[testSparseInputNorm2.indices]
    testSparseInput=testSparseInputNorm2.copy().tocsr()
    testSparseInput.sort_indices()
    testSparseInput.data=testSparseInput.data-trainSparseDiv2[testSparseInput.indices]

if ("graphInputData" in globals()) and (not (graphInputData is None)):
  trainGraphInput=graphInputData[graphSampleIndex[trainSamples].values].copy()
  testGraphInput=graphInputData[graphSampleIndex[testSamples].values].copy()