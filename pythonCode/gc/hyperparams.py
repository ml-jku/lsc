#Copyright (C) 2018 Andreas Mayr
#Licensed under GNU General Public License v3.0 (see http://www.bioinf.jku.at/research/lsc/LICENSE and https://github.com/ml-jku/lsc/blob/master/LICENSE)

dictionary0 = {
  'basicArchitecture': ['GraphConv'],
  'learningRate': [0.001, 0.0001],
  'dropout': [0.0, 0.5],
  'graphLayers': [[128]*2, [128]*3, [128]*4],
  'denseLayers': [[1024], [2048]]
}

hyperParams=pd.DataFrame(list(itertools.product(*dictionary0.values())), columns=dictionary0.keys())
hyperParams.index=np.arange(len(hyperParams.index.values))
