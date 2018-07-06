#Copyright (C) 2018 Guenter Klambauer, Andreas Mayr
#Licensed under GNU General Public License v3.0 (see http://www.bioinf.jku.at/research/lsc/LICENSE and https://github.com/ml-jku/lsc/blob/master/LICENSE)

import os
import numpy as np
import scipy
import pandas as pd


resDir=os.getenv("HOME")+"/mydata/trgpred/chembl20/results/allRes/"



titleMap=pd.Series(["FNN", "SVM", "RF", "KNN", "NB", "SEA", "GC", "Weave", "SmilesLSTM"], index=['dnn', 'svm', 'rf', 'knn', 'nb', 'sea', 'conv', 'weave', 'lstm'])



features=['static', 'semi', 'ecfp6', 'dfs8', 'ecfp6_tox', 'graph', 'smiles']
algs=['dnn', 'svm', 'rf', 'knn', 'nb', 'sea', 'conv', 'weave', 'lstm']

auc=dict()
for feat in features:
  auc[feat]=dict()
  for alg in algs:
    filename=resDir+"/"+feat+"/"+alg+".csv"
    if os.path.exists(filename):
      auc[feat][alg]=pd.read_csv(filename, index_col=0)



showFeatures=["static", "semi", "ecfp6", "dfs8", "ecfp6_tox", "graph", "smiles"]
titleFeatures=["StaticF", "SemiF", "ECFP6", "DFS8", "ECFP6+ToxF", "Graph", "SMILES"]
showAlgorithms=["dnn", "svm", "rf", "knn", "nb", "sea", "conv", "weave", "lstm"]
titleAlgorithms=titleMap[showAlgorithms].values

tabOverview=pd.DataFrame([(alg, feat, auc[feat][alg].mean(1).mean(), auc[feat][alg].mean(1).std()) if alg in auc[feat] else (alg, feat, float('nan'), float('nan')) for alg in algs for feat in features], columns=["alg", "feat", "mymean", "mystd"])
tabOverview["myval"]=zip(tabOverview.mymean, tabOverview.mystd)
tabOverview["mystr"]=["\\footnotesize"+"%.3f" % x[0]+"\\tiny$\\pm$"+"%.3f" % x[1] for x in zip(tabOverview.mymean.tolist(), tabOverview.mystd.tolist())]
tabOverview=pd.pivot_table(tabOverview, index='feat', columns='alg', values='mystr', aggfunc=lambda x: x, dropna=False)
tabOverview=tabOverview.loc[showFeatures, showAlgorithms]
tabOverview.index=titleFeatures
tabOverview.columns=titleAlgorithms

print(tabOverview.to_latex(escape=False))  