#Copyright (C) 2018 Andreas Mayr
#Licensed under GNU General Public License v3.0 (see http://www.bioinf.jku.at/research/lsc/LICENSE and https://github.com/ml-jku/lsc/blob/master/LICENSE)

if continueComputations:
  if computeTestPredictions:
    if compPerformanceTest:
      saveFilename=savePrefix+".test.auc.pckl"
      if os.path.isfile(saveFilename):
        saveFile=open(saveFilename, "rb")
        reportTestAUC=pickle.load(saveFile)
        saveFile.close()

      saveFilename=savePrefix+".test.ap.pckl"
      if os.path.isfile(saveFilename):
        saveFile=open(saveFilename, "rb")
        reportTestAP=pickle.load(saveFile)
        saveFile.close()

  if computeTrainPredictions:
    if compPerformanceTrain:
      saveFilename=savePrefix+".train.auc.pckl"
      if os.path.isfile(saveFilename):
        saveFile=open(saveFilename, "rb")
        reportTrainAUC=pickle.load(saveFile)
        saveFile.close()

      saveFilename=savePrefix+".train.ap.pckl"
      if os.path.isfile(saveFilename):
        saveFile=open(saveFilename, "rb")
        reportTrainAP=pickle.load(saveFile)
        saveFile.close()

if logPerformanceAtBestIter:
  saveFilename=savePrefix+".eval.auc.npy"
  if os.path.isfile(saveFilename):
    reportAUCBestIter=np.load(saveFilename)
  
  saveFilename=savePrefix+".eval.ap.npy"
  if os.path.isfile(saveFilename):
    reportAPBestIter=np.load(saveFilename)  

if savePredictionsAtBestIter:
  if useDenseOutputNetPred:
    saveFilename=savePrefix+".evalPredict.pckl"
    if os.path.isfile(saveFilename):
      saveFile=open(saveFilename, "rb")
      predDenseBestIter=pickle.load(saveFile)
      saveFile.close()
  else:
    saveFilename=savePrefix+".evalPredict.pckl"
    if os.path.isfile(saveFilename):
      saveFile=open(saveFilename, "rb")
      predSparseBestIter=pickle.load(saveFile)
      saveFile.close()

saveFilename=savePrefix+".trainInfo.pckl"
if os.path.isfile(saveFilename):
  saveFile=open(saveFilename, "rb")
  startEpoch=pickle.load(saveFile)
  minibatchCounterTrain=pickle.load(saveFile)
  minibatchCounterTest=pickle.load(saveFile)
  minibatchReportNr=pickle.load(saveFile)
  saveFile.close()

saveFilename=savePrefix+".trainModel.meta"
if os.path.isfile(saveFilename):
  saveFilename=savePrefix+".trainModel"
  tf.train.Saver().restore(session, saveFilename)
