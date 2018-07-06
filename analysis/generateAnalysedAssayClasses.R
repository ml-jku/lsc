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
trgInfo=read.table(file.path(rawDataPathname, "trgInfo.txt"), header=FALSE, sep="\t", na.strings="\\N", stringsAsFactors=FALSE, quote="", comment.char="")
pcInfo=read.table(file.path(rawDataPathname, "pcInfo.txt"), header=FALSE, sep="\t", na.strings="\\N", stringsAsFactors=FALSE, quote="", comment.char="", row.names=1)



trgHier=cbind(as.integer(rownames(pcInfo)), pcInfo[,1])

myvals=trgHier[trgHier[,2]==0L,1]
myvals=myvals[-1L]
trgHier2=trgHier[-1,]
trgHier3=trgHier2[!(trgHier2[,1L]%in%myvals),]

parentMap=trgHier3[,2L]
names(parentMap)=as.character(trgHier3[,1L])

parents0=trgHier3[,2L]
parents1=ifelse(parents0%in%myvals, parents0, parentMap[as.character(parents0)])
parents2=ifelse(parents1%in%myvals, parents1, parentMap[as.character(parents1)])
parents3=ifelse(parents2%in%myvals, parents2, parentMap[as.character(parents2)])
parents4=ifelse(parents3%in%myvals, parents3, parentMap[as.character(parents3)])
parents5=ifelse(parents4%in%myvals, parents4, parentMap[as.character(parents4)])
all(parents4==parents5)

finalMap=c(0, myvals, parents5)
names(finalMap)=c(0, myvals, trgHier3[,1L])



dir.create(file.path(rawDataPathname, "cats"))
for(mycat in c("Enzyme", "Epigenetic regulator", "Ion channel", "Membrane receptor", "Transcription factor", "Transporter")) {
  mychembl=unique(chemblAssays[chemblAssays%in%trgInfo[trgInfo[,2L]%in%as.integer(names(finalMap)[as.integer(finalMap)==as.integer(rownames(pcInfo[pcInfo[,2L]==mycat,]))]),1]])
  writeLines(mychembl, file.path(rawDataPathname, "cats", paste0(mycat, ".txt")))
}



