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

saveFilename=savePrefix+".trainInfo.pckl"
saveFile=open(saveFilename, "wb")
pickle.dump(epoch, saveFile)
pickle.dump(minibatchCounterTrain, saveFile)
pickle.dump(minibatchCounterTest, saveFile)
pickle.dump(minibatchReportNr, saveFile)
saveFile.close()

saveFilename=savePrefix+".trainModel"
with model._get_tf("Graph").as_default():
  tf.train.Saver().save(model.session, saveFilename)

saveFilename=savePrefix+"."+finMark+".pckl"
saveFile=open(saveFilename, "wb")
finNr=0
pickle.dump(finNr, saveFile)
saveFile.close()