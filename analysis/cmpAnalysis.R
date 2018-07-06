#Copyright (C) 2018 Andreas Mayr
#Licensed under GNU General Public License v3.0 (see http://www.bioinf.jku.at/research/lsc/LICENSE and https://github.com/ml-jku/lsc/blob/master/LICENSE)

resDir=paste0(Sys.getenv("HOME"), "/mydata/trgpred/chembl20/results/allRes")




titleMap=c("DNN", "SVM", "RF", "KNN", "NB", "SEA", "GC", "Weave", "LSTM")
names(titleMap)=c('dnn', 'svm', 'rf', 'knn', 'nb', 'sea', 'conv', 'weave', 'lstm')



features=c('static', 'semi', 'ecfp6', 'dfs8', 'ecfp6_tox', 'graph', 'smiles')
algs=c('dnn', 'svm', 'rf', 'knn', 'nb', 'sea', 'conv', 'weave', 'lstm')

auc=list()
for(feat in features) {
  auc[[feat]]=list()
  for(alg in algs) {
    filename=paste0(resDir, "/", feat, "/", alg, ".csv")
    if(file.exists(filename)) {
      auc[[feat]][[alg]]=read.csv(filename, row.names=1)
    }
    else {
    }
  }
}




takeFeature='ecfp6'
showAlgorithms=c('dnn', 'svm', 'rf', 'knn', 'nb', 'sea', 'conv', 'weave', 'lstm')

cmpAlgs=c('dnn', 'svm', 'rf', 'knn', 'nb', 'sea', 'conv', 'weave', 'lstm')
cmpFeatures=c(takeFeature, takeFeature, takeFeature, takeFeature, takeFeature, takeFeature, 'graph', 'graph', 'smiles', takeFeature)

pval=matrix(NA, nrow=length(cmpAlgs), ncol=length(cmpAlgs))
for(i in 1L:length(cmpAlgs)) {
  for(j in 1L:length(cmpAlgs)) {
    tab1=auc[[cmpFeatures[i]]][[cmpAlgs[i]]]
    tab2=auc[[cmpFeatures[j]]][[cmpAlgs[j]]]
    
    if(!(is.null(tab1)||is.null(tab2))) {
      pval[j,i]=wilcox.test(rowMeans(tab1), rowMeans(tab2), alternative = "less", paired=TRUE)$p.value
    }
  }
}
rownames(pval)=algs
colnames(pval)=algs






