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



chemblAssays=readLines(targetsFilename)

assayInfo=read.table(file.path(rawDataPathname, "assayInfo.txt"), header=FALSE, sep="\t", na.strings="\\N", stringsAsFactors=FALSE, quote="", comment.char="")
assayTypeInfo=read.table(file.path(rawDataPathname, "assayTypeInfo.txt"), header=FALSE, sep="\t", na.strings="\\N", stringsAsFactors=FALSE, quote="", comment.char="")

dir.create(file.path(rawDataPathname, "types"))
for(i in 1:nrow(assayTypeInfo)) {
  mychembl=assayInfo$V2[assayInfo$V3==assayTypeInfo[i,1]]
  mychembl=mychembl[mychembl%in%chemblAssays]
  writeLines(mychembl, file.path(rawDataPathname, "types", paste0(assayTypeInfo[i,2], ".txt")))
}
