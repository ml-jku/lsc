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



actFileName=paste0(rawDataPathname, "/", "chemblActivities.txt")
trainFileName=paste0(rawDataPathname, "/", "train.info")
targetsFilename=paste0(rawDataPathname, "/", "computeTargets.in")
trainWSName=paste0(rawDataPathname, "/", "trainTable.RData")



sampleIdToName=read.table(sampleIdFilename, row.names=NULL, stringsAsFactors=FALSE)[,1]
sampleNameToId=seq(length(sampleIdToName))
names(sampleNameToId)=sampleIdToName



actTab=read.table(actFileName, header=FALSE, sep="\t", na.strings="\\N", stringsAsFactors=FALSE, quote="", comment.char="")
colnames(actTab)=c("molId", "assId", "relation", "value", "unit", "type", "comment", "docId", "tId", "ptype", "ttype", "cscore")
#actTab=actTab[actTab$cscore>=4L,]
actTab$value=as.numeric(actTab$value)

inactCommentTypes=c("Not Active (inhibition < 50% @ 10 uM and thus dose-reponse curve not measured)", "Not Active", "inactive")
actCommentTypes=c("active", "Active")
unknownTypes=c("inconclusive", "Inconclusive", "Not Determined")
relationTypes=c("=", "<", "<=", "~", ">", ">=")

sel1=actTab$comment%in%actCommentTypes
sel2=actTab$comment%in%inactCommentTypes

remTab=actTab[(!(sel1|sel2)),]
remTab=remTab[remTab$unit=="nM",]
remTab=remTab[!is.na(remTab$value),]
#remTab=remTab[remTab$value>0.0,]
remTab=remTab[remTab$relation%in%relationTypes,]

biggerRow=(remTab$relation==">")|(remTab$relation==">=")
lowerRow=(remTab$relation=="<")|(remTab$relation=="<=")
equalRow=(remTab$relation=="=")|(remTab$relation=="~")

indefFilter=((remTab$value>10.0^(9.0-5.5))&(remTab$value<10.0^(9.0-4.5)))
inactiveFilter1=((remTab$value>=10.0^(9.0-4.5))&(equalRow))|((remTab$value>=10.0^(9.0-4.5))&(biggerRow))
inactiveFilter2=((remTab$value>10.0^(9.0-5.0))&(equalRow))|((remTab$value>10.0^(9.0-5.0))&(biggerRow))
activeFilter1=((remTab$value<=10.0^(9.0-5.5))&(equalRow))|((remTab$value<=10.0^(9.0-5.5))&(lowerRow))
activeFilter2=((remTab$value<10.0^(9.0-5.0))&(equalRow))|((remTab$value<10.0^(9.0-5.0))&(lowerRow))




actClass=rep(0L, nrow(remTab))
actClass[indefFilter]=2L
actClass[activeFilter2]=13L
actClass[inactiveFilter2]=11L
actClass[activeFilter1]=3L
actClass[inactiveFilter1]=1L

newTab1=data.frame(actTab[sel1, c("assId", "molId")], actInfo=3L, row.names=NULL, stringsAsFactors=FALSE)
newTab2=data.frame(actTab[sel2, c("assId", "molId")], actInfo=1L, row.names=NULL, stringsAsFactors=FALSE)
newTab3=data.frame(remTab[, c("assId", "molId")], actInfo=actClass, row.names=NULL, stringsAsFactors=FALSE)

actTabNew=rbind(newTab1, newTab2, newTab3)
actTabNew=actTabNew[actTabNew$molId%in%sampleIdToName,]
tpSampleId=sampleNameToId[actTabNew$molId]-1L
actTabNew=data.frame(assId=actTabNew[, "assId"], tpSampleId=tpSampleId, actInfo=actTabNew[, "actInfo"], row.names=NULL, stringsAsFactors=FALSE)

targets=unique(actTabNew$assId)

actInfo=actTabNew$actInfo
for(myTarget in targets) {
  cat(which(myTarget==targets), "/", length(targets), "\n")
  
  selRows=which(actTabNew$assId==myTarget)
  targetTab=actTabNew[selRows,]
  
  targetComps=targetTab[, "tpSampleId"]
  weakActComps=unique(targetTab[targetTab$actInfo==13L, "tpSampleId"])
  weakInactComps=unique(targetTab[targetTab$actInfo==11L, "tpSampleId"])
  actComps=unique(targetTab[targetTab$actInfo==3L, "tpSampleId"])
  inactComps=unique(targetTab[targetTab$actInfo==1L, "tpSampleId"])
  garbageComps=unique(targetTab[targetTab$actInfo==0L, "tpSampleId"])
  
  intersectActInact=intersect(actComps, inactComps)
  garbageComps=unique(union(intersectActInact, garbageComps))
  actComps=unique(setdiff(actComps, garbageComps))
  inactComps=unique(setdiff(inactComps, garbageComps))
  
  intersectWeakActGarbage=intersect(weakActComps, garbageComps)
  intersectWeakActInact=intersect(weakActComps, inactComps)
  intersectWeakActAct=intersect(weakActComps, actComps)
  intersectWeakInactGarbage=intersect(weakInactComps, garbageComps)
  intersectWeakInactAct=intersect(weakInactComps, actComps)
  intersectWeakInactInact=intersect(weakInactComps, inactComps)
  weakActComps=unique(setdiff(weakActComps, unique(union(union(intersectWeakActGarbage, intersectWeakActInact), intersectWeakActAct))))
  weakInactComps=unique(setdiff(weakInactComps, unique(union(union(intersectWeakInactGarbage, intersectWeakInactAct), intersectWeakInactInact))))
  intersectWeakActWeakInact=intersect(weakActComps, weakInactComps)
  weakActComps=unique(setdiff(weakActComps, intersectWeakActWeakInact))
  weakInactComps=unique(setdiff(weakInactComps, intersectWeakActWeakInact))
  weakGarbageComps=unique(intersectWeakActWeakInact)
  
  actInfo[selRows[targetComps%in%weakActComps]]=13L
  actInfo[selRows[targetComps%in%weakInactComps]]=11L  
  actInfo[selRows[targetComps%in%weakGarbageComps]]=10L    
  actInfo[selRows[targetComps%in%actComps]]=3L
  actInfo[selRows[targetComps%in%inactComps]]=1L
  actInfo[selRows[targetComps%in%garbageComps]]=0L
}
actTabNew$actInfo=actInfo

write.table(actTabNew[,c("actInfo", "tpSampleId", "assId")], file=trainFileName, row.names=FALSE, col.names=FALSE, quote=FALSE)
write.table(targets, file=targetsFilename, row.names=FALSE, col.names=FALSE, quote=FALSE)

save.image(trainWSName, compress=FALSE)
