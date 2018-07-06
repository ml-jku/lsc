#Copyright (C) 2018 Andreas Mayr
#Licensed under GNU General Public License v3.0 (see http://www.bioinf.jku.at/research/lsc/LICENSE and https://github.com/ml-jku/lsc/blob/master/LICENSE)

projectName="chembl20";
rawDataFoldername="chembl20";

basePathname=paste0(Sys.getenv("HOME"), "/mydata/trgpred/");
rawDataPathname=paste0(Sys.getenv("HOME"), "/mydata/raw/");
rawDataPathname=file.path(rawDataPathname, rawDataFoldername);

projectPathname=file.path(basePathname, projectName);
chemPathname=file.path(projectPathname, "chemFeatures");
clusterPathname=file.path(chemPathname, "cl");
dchemPathname=file.path(chemPathname, "d");
schemPathname=file.path(chemPathname, "s");
trainPathname=file.path(projectPathname, "train");
runPathname=file.path(projectPathname, "run");

sampleIdFilename=file.path(chemPathname, "SampleIdTable.txt")
targetSampleFilename=file.path(trainPathname, "t1.info")
predictTargetFilename=file.path(trainPathname, "trg1.info")
clusterSampleFilename=file.path(clusterPathname, "cl1.info")
predictSampleFilename=file.path(trainPathname, "p1.info")

targetsFilename=paste0(rawDataPathname, "/", "tocompute.info")

mychemblSort=readLines(targetsFilename)



writeDir=file.path(projectPathname, "results/allRes/");
dir.create(writeDir, recursive=TRUE)

dir.create(paste0(writeDir, "static"))
dir.create(paste0(writeDir, "semi"))
dir.create(paste0(writeDir, "ecfp6"))
dir.create(paste0(writeDir, "dfs8"))
dir.create(paste0(writeDir, "ecfp6_tox"))
dir.create(paste0(writeDir, "graph"))
dir.create(paste0(writeDir, "smiles"))



# SVM
settings=list()
 
for(j in 0L:4L) {
  resTab=c()
  setting=paste0("/", as.character(j), "/")
  for(i in 1L:length(mychemblSort)) {
    resTab=rbind(resTab, as.double(strsplit(readLines(paste0(file.path(runPathname, "svm", mychemblSort[i]), setting, "clustEvalPredAUC.out")),",")[[1]]))
  }
  rownames(resTab)=mychemblSort
  settings[[j+1]]=resTab[mychemblSort,]
}
resSVM=settings
for(i in 1L:length(resSVM)) {
  colnames(resSVM[[i]])=c("fold0", "fold1", "fold2")
}

write.csv(resSVM[[1]], file=paste0(writeDir, "static/svm.csv"))
write.csv(resSVM[[2]], file=paste0(writeDir, "semi/svm.csv"))
write.csv(resSVM[[3]], file=paste0(writeDir, "ecfp6/svm.csv"))
write.csv(resSVM[[4]], file=paste0(writeDir, "dfs8/svm.csv"))
write.csv(resSVM[[5]], file=paste0(writeDir, "ecfp6_tox/svm.csv"))



#KNN
settings=list()
 
for(j in 0L:4L) {
  resTab=c()
  setting=paste0("/", as.character(j), "/")
  for(i in 1L:length(mychemblSort)) {
    resTab=rbind(resTab, as.double(strsplit(readLines(paste0(file.path(runPathname, "knn", mychemblSort[i]), setting, "clustEvalPredAUC.out")),",")[[1]]))
  }
  rownames(resTab)=mychemblSort
  settings[[j+1]]=resTab[mychemblSort,]
}
resKNN=settings
for(i in 1L:length(resKNN)) {
  colnames(resKNN[[i]])=c("fold0", "fold1", "fold2")
}

write.csv(resKNN[[1]], file=paste0(writeDir, "static/knn.csv"))
write.csv(resKNN[[2]], file=paste0(writeDir, "semi/knn.csv"))
write.csv(resKNN[[3]], file=paste0(writeDir, "ecfp6/knn.csv"))
write.csv(resKNN[[4]], file=paste0(writeDir, "dfs8/knn.csv"))
write.csv(resKNN[[5]], file=paste0(writeDir, "ecfp6_tox/knn.csv"))



#SEA
settings=list()
 
for(j in c(0L)) {
  resTab=c()
  setting=paste0("/", as.character(j), "/")
  for(i in 1L:length(mychemblSort)) {
    resTab=rbind(resTab, as.double(strsplit(readLines(paste0(file.path(runPathname, "sea", mychemblSort[i]), setting, "clustEvalPredAUC.out")),",")[[1]]))
  }
  rownames(resTab)=mychemblSort
  settings[[j+1]]=resTab[mychemblSort,]
}
resSEA=settings
for(i in 1L:length(resSEA)) {
  colnames(resSEA[[i]])=c("fold0", "fold1", "fold2")
}

write.csv(resSEA[[1]], file=paste0(writeDir, "ecfp6/sea.csv"))



#NB
settings=list()
 
for(j in 0L:3L) {
  resTab=c()
  setting=paste0("/", as.character(j), "/")
  for(i in 1L:length(mychemblSort)) {
    resTab=rbind(resTab, as.double(strsplit(readLines(paste0(file.path(runPathname, "nb", mychemblSort[i]), setting, "clustEvalPredAUC.out")),",")[[1]]))
  }
  rownames(resTab)=mychemblSort
  settings[[j+1]]=resTab[mychemblSort,]
}
resNB=settings
for(i in 1L:length(resNB)) {
  colnames(resNB[[i]])=c("fold0", "fold1", "fold2")
}

write.csv(resNB[[1]], file=paste0(writeDir, "semi/nb.csv"))
write.csv(resNB[[2]], file=paste0(writeDir, "ecfp6/nb.csv"))
write.csv(resNB[[3]], file=paste0(writeDir, "dfs8/nb.csv"))
write.csv(resNB[[4]], file=paste0(writeDir, "ecfp6_tox/nb.csv"))



#Deep Learning
dnnResPathname=file.path(projectPathname, "results/dnnRes/")

resDNN=list()
resDNN[[1]]=read.csv(paste0(dnnResPathname, "static.roc.csv"), stringsAsFactors=FALSE, row.names=1)[mychemblSort,c("fold0", "fold1", "fold2")]
resDNN[[2]]=read.csv(paste0(dnnResPathname, "semi.roc.csv"), stringsAsFactors=FALSE, row.names=1)[mychemblSort,c("fold0", "fold1", "fold2")]
resDNN[[3]]=read.csv(paste0(dnnResPathname, "ecfp.roc.csv"), stringsAsFactors=FALSE, row.names=1)[mychemblSort,c("fold0", "fold1", "fold2")]
resDNN[[4]]=read.csv(paste0(dnnResPathname, "dfs.roc.csv"), stringsAsFactors=FALSE, row.names=1)[mychemblSort,c("fold0", "fold1", "fold2")]
resDNN[[5]]=read.csv(paste0(dnnResPathname, "ecfpTox.roc.csv"), stringsAsFactors=FALSE, row.names=1)[mychemblSort,c("fold0", "fold1", "fold2")]
for(i in 1L:length(resDNN)) {
  colnames(resDNN[[i]])=c("fold0", "fold1", "fold2")
}

write.csv(resDNN[[1]], file=paste0(writeDir, "static/dnn.csv"))
write.csv(resDNN[[2]], file=paste0(writeDir, "semi/dnn.csv"))
write.csv(resDNN[[3]], file=paste0(writeDir, "ecfp6/dnn.csv"))
write.csv(resDNN[[4]], file=paste0(writeDir, "dfs8/dnn.csv"))
write.csv(resDNN[[5]], file=paste0(writeDir, "ecfp6_tox/dnn.csv"))



#Graph Conv
resGraph=list()
resGraph[[1]]=read.csv(paste0(dnnResPathname, "graphConv.roc.csv"), stringsAsFactors=FALSE, row.names=1)[mychemblSort,c("fold0", "fold1", "fold2")]
resGraph[[2]]=read.csv(paste0(dnnResPathname, "graphWeave.roc.csv"), stringsAsFactors=FALSE, row.names=1)[mychemblSort,c("fold0", "fold1", "fold2")]

for(i in 1L:length(resGraph)) {
  colnames(resGraph[[i]])=c("fold0", "fold1", "fold2")
}

write.csv(resGraph[[1]], file=paste0(writeDir, "graph/conv.csv"))
write.csv(resGraph[[2]], file=paste0(writeDir, "graph/weave.csv"))



#LSTM
resLSTM=list()
resLSTM[[1]]=read.csv(paste0(dnnResPathname, "lstm.roc.csv"), stringsAsFactors=FALSE, row.names=1)[mychemblSort,c("fold0", "fold1", "fold2")]

for(i in 1L:length(resLSTM)) {
  colnames(resLSTM[[i]])=c("fold0", "fold1", "fold2")
}

write.csv(resLSTM[[1]], file=paste0(writeDir, "smiles/lstm.csv"))
