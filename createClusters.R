#Copyright (C) 2018 Andreas Mayr
#Licensed under GNU General Public License v3.0 (see http://www.bioinf.jku.at/research/lsc/LICENSE and https://github.com/ml-jku/lsc/blob/master/LICENSE)

projectName="chembl20";
rawDataFoldername="chembl20";
clusterInfo="cl1"
clusterVersion="clusterMinFull"



basePathname=paste0(Sys.getenv("HOME"), "/mydata/trgpred/");

projectPathname=file.path(basePathname, projectName);
chemPathname=file.path(projectPathname, "chemFeatures");
clusterPathname=file.path(chemPathname, "cl");
dchemPathname=file.path(chemPathname, "d");
schemPathname=file.path(chemPathname, "s");
trainPathname=file.path(projectPathname, "train");
runPathname=file.path(projectPathname, "run");

sampleIdFilename=file.path(chemPathname, "SampleIdTable.txt")

clusterSampleFilename=file.path(clusterPathname, paste0(clusterInfo, ".info"))



set.seed(12345L)
i=70L
rawClusterTable=read.table(file.path(clusterPathname, clusterVersion, paste0("clustering_", i, ".txt")), stringsAsFactors=FALSE, sep=",")
clusterSizes=table(rawClusterTable[,1])
clusterSizes=clusterSizes[sample(length(clusterSizes))]
clusterSizes=clusterSizes[order(clusterSizes)]
if((length(clusterSizes)%%3L)>0L) {
  clusterAssignment=c(rep(0L:2L, length(clusterSizes)%/%3L), (0L:2L)[1L:(length(clusterSizes)%%3L)])
} else {
  clusterAssignment=rep(0L:2L, length(clusterSizes)%/%3L)
}
names(clusterAssignment)=names(clusterSizes)
newclusterAssignment=clusterAssignment[as.character(rawClusterTable[,1])]
names(newclusterAssignment)=NULL
newClusterTable=rawClusterTable
newClusterTable[,1]=newclusterAssignment
write.table(newClusterTable, file=clusterSampleFilename, row.names=FALSE, col.names=FALSE, quote=FALSE)







