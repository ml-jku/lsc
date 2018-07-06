#Copyright (C) 2018 Andreas Mayr
#Licensed under GNU General Public License v3.0 (see http://www.bioinf.jku.at/research/lsc/LICENSE and https://github.com/ml-jku/lsc/blob/master/LICENSE)

if computeTestPredictions:
  if compPerformanceTest:
    saveFilename=savePrefix+".test.auc.pckl"
    saveFile=open(saveFilename, "wb")
    pickle.dump(reportTestAUC, saveFile)
    saveFile.close()

    saveFilename=savePrefix+".test.ap.pckl"
    saveFile=open(saveFilename, "wb")
    pickle.dump(reportTestAP, saveFile)
    saveFile.close()

if computeTrainPredictions:
  if compPerformanceTrain:
    saveFilename=savePrefix+".train.auc.pckl"
    saveFile=open(saveFilename, "wb")
    pickle.dump(reportTrainAUC, saveFile)
    saveFile.close()

    saveFilename=savePrefix+".train.ap.pckl"
    saveFile=open(saveFilename, "wb")
    pickle.dump(reportTrainAP, saveFile)
    saveFile.close()

if logPerformanceAtBestIter:
  saveFilename=savePrefix+".eval.auc"
  np.save(saveFilename, reportAUCBestIter)
  saveFilename=savePrefix+".eval.ap"
  np.save(saveFilename, reportAPBestIter)

if savePredictionsAtBestIter:
  if useDenseOutputNetPred:
    saveFilename=savePrefix+".evalPredict.pckl"
    saveFile=open(saveFilename, "wb")
    pickle.dump(predDenseBestIter, saveFile)
    saveFile.close()
    
    saveFilename=savePrefix+".evalPredict.hdf5"
    saveFile=h5py.File(saveFilename, "w")
    saveFile.create_dataset('predictions', data=predDenseBestIter)
    saveFile.close()
  else:
    saveFilename=savePrefix+".evalPredict.pckl"
    saveFile=open(saveFilename, "wb")
    pickle.dump(predSparseBestIter, saveFile)
    saveFile.close()
    
    saveFilename=savePrefix+".evalPredict.mtx"
    scipy.io.mmwrite(saveFilename, predSparseBestIter)
  
  saveFilename=savePrefix+".eval"
  np.savetxt(saveFilename+".cmpNames", np.array(testSamples), fmt="%s")
  np.savetxt(saveFilename+".targetNames", np.array(targetAnnInd.index.values), fmt="%s")
  
  if outerFold>=0:
    if not (denseOutputData is None):
      saveFilename=savePrefix+".evalTrue.hdf5"
      saveFile=h5py.File(saveFilename, "w")
      saveFile.create_dataset('true', data=testDenseOutput)
      saveFile.close()
    
    if not (sparseOutputData is None):
      saveFilename=savePrefix+".evalTrue.mtx"
      scipy.io.mmwrite(saveFilename, testSparseOutput)

saveFilename=savePrefix+".trainInfo.pckl"
saveFile=open(saveFilename, "wb")
pickle.dump(epoch, saveFile)
pickle.dump(minibatchCounterTrain, saveFile)
pickle.dump(minibatchCounterTest, saveFile)
pickle.dump(minibatchReportNr, saveFile)
saveFile.close()

saveFilename=savePrefix+".trainModel"
model.save(saveFilename)

saveFilename=savePrefix+"."+finMark+".pckl"
saveFile=open(saveFilename, "wb")
finNr=0
pickle.dump(finNr, saveFile)
saveFile.close()