#Copyright (C) 2018 Andreas Mayr
#Licensed under GNU General Public License v3.0 (see http://www.bioinf.jku.at/research/lsc/LICENSE and https://github.com/ml-jku/lsc/blob/master/LICENSE)

if denseOutputData is not None:
  nrOutputTargets=denseOutputData.shape[1]
if sparseOutputData is not None:
  nrOutputTargets=sparseOutputData.shape[1]

if normalizeGlobalDense:
  denseInput=denseInputData.copy()
  denseMean1=np.nanmean(denseInput, 0)
  denseStd1=np.nanstd(denseInput, 0)+0.0001
  denseInput=(denseInput-denseMean1)/denseStd1
  denseInput=np.tanh(denseInput)
  denseMean2=np.nanmean(denseInput, 0)
  denseStd2=np.nanstd(denseInput, 0)+0.0001
  denseInput=(denseInput-denseMean2)/denseStd2
  denseInputData=denseInput

if normalizeGlobalSparse:
  sparseInput=sparseInputData.copy()
  sparseMean1=sparseInput.mean(0).A[0]
  sparseStd1=np.sqrt(sparseInput.multiply(sparseInput).mean(0).A[0]-sparseMean1**2)
  sparseInputNorm1=sparseInput.copy()
  sparseInputNorm1.data=np.tanh((sparseInputNorm1.data-sparseMean1[sparseInputNorm1.indices])/sparseStd1[sparseInputNorm1.indices])
  sparseDiv1=np.tanh(-sparseMean1/sparseStd1)
  sparseInputHelp1=sparseInputNorm1.copy()
  sparseInputHelp1.data=sparseInputHelp1.data-sparseDiv1[sparseInputHelp1.indices]
  sparseInputHelp2=sparseInputNorm1.copy()
  sparseInputHelp2.data=sparseInputHelp2.data**2-sparseDiv1[sparseInputHelp2.indices]**2
  sparseMean2=sparseInputHelp1.mean(0).A[0]+sparseDiv1
  sparseStd2=np.sqrt(sparseInputHelp2.mean(0).A[0]+sparseDiv1**2-sparseMean2**2)
  sparseInputNorm2=sparseInputNorm1.copy()
  sparseInputNorm2.data=(sparseInputNorm2.data-sparseMean2[sparseInputNorm2.indices])/sparseStd2[sparseInputNorm2.indices]
  sparseDiv2=(sparseDiv1-sparseMean2)/sparseStd2
  sparseInput=sparseInputNorm2.copy().tocsr()
  sparseInput.sort_indices()
  sparseInput.data=sparseInput.data-sparseDiv2[sparseInput.indices]
  sparseInputData=sparseInput