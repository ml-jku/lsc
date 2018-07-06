#Copyright (C) 2018 Andreas Mayr
#Licensed under GNU General Public License v3.0 (see http://www.bioinf.jku.at/research/lsc/LICENSE and https://github.com/ml-jku/lsc/blob/master/LICENSE)



indMod=np.where(np.array([type(mychemblConvertedMols[x])==np.ndarray for x in trainGraphInput]))
trainDenseOutput[indMod,:]=0.0
uranium=singleFunc(rdkit.Chem.MolFromSmiles("[U]"))



predDenseTrain=[]
idxSamplesEval=[arr[1] for arr in sklearn.model_selection.KFold(n_splits=int(math.ceil(len(trainSamples)/batchSize)), shuffle=False).split(np.arange(len(trainSamples)))]
for j in range(len(idxSamplesEval)):
  batchGraphX=mychemblConvertedMols[trainGraphInput[idxSamplesEval[j]]]
  batchGraphX=np.array([uranium if type(x)==np.ndarray else x for x in batchGraphX])
  if compPerformanceTrain:
    batchDenseY=trainDenseOutput[idxSamplesEval[j]]
  extendSize=0
  if len(batchGraphX)<model.batch_size:
    extendSize=model.batch_size-len(batchGraphX)
    batchGraphX=np.append(batchGraphX, batchGraphX[0:extendSize])
  
  #batchInputSingle=[singleFunc(molX) for molX in batchGraphX]
  batchInput=batchFunc(model, batchGraphX)
  myfeedDict=batchInput
  myfeedDict[model._training_placeholder]=0.0
  with model._get_tf("Graph").as_default():
    predDenseTrain.append(model.session.run(model.outputs[0], feed_dict=myfeedDict)[0:(model.batch_size-extendSize)])

predDenseTrain=np.vstack(predDenseTrain)
if compPerformanceTrain:
  sumTrainAUC=np.array(utilsLib.calculateAUCs(trainDenseOutput, predDenseTrain))
  sumTrainAP=np.array(utilsLib.calculateAPs(trainDenseOutput, predDenseTrain))



predDenseTest=[]
idxSamplesEval=[arr[1] for arr in sklearn.model_selection.KFold(n_splits=int(math.ceil(len(testSamples)/batchSize)), shuffle=False).split(np.arange(len(testSamples)))]
for j in range(len(idxSamplesEval)):
  batchGraphX=mychemblConvertedMols[testGraphInput[idxSamplesEval[j]]]
  batchGraphX=np.array([uranium if type(x)==np.ndarray else x for x in batchGraphX])
  if compPerformanceTrain:
    batchDenseY=testDenseOutput[idxSamplesEval[j]]
  extendSize=0
  if len(batchGraphX)<model.batch_size:
    extendSize=model.batch_size-len(batchGraphX)
    batchGraphX=np.append(batchGraphX, batchGraphX[0:extendSize])
  
  #batchInputSingle=[singleFunc(molX) for molX in batchGraphX]
  batchInput=batchFunc(model, batchGraphX)
  myfeedDict=batchInput
  myfeedDict[model._training_placeholder]=0.0
  with model._get_tf("Graph").as_default():
    predDenseTest.append(model.session.run(model.outputs[0], feed_dict=myfeedDict)[0:(model.batch_size-extendSize)])

predDenseTest=np.vstack(predDenseTest)
if compPerformanceTest:
  sumTestAUC=np.array(utilsLib.calculateAUCs(testDenseOutput, predDenseTest))
  sumTestAP=np.array(utilsLib.calculateAPs(testDenseOutput, predDenseTest))



print("\n", file=dbgOutput)
print("Train Mean AUC: ", file=dbgOutput)
print(np.nanmean(sumTrainAUC), file=dbgOutput)
print("\n", file=dbgOutput)

print("\n", file=dbgOutput)
print("Test Mean AUC: ", file=dbgOutput)
print(np.nanmean(sumTestAUC), file=dbgOutput)
print("\n", file=dbgOutput)