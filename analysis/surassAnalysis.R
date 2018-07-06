#Copyright (C) 2018 Guenter Klambauer, Andreas Mayr
#Licensed under GNU General Public License v3.0 (see http://www.bioinf.jku.at/research/lsc/LICENSE and https://github.com/ml-jku/lsc/blob/master/LICENSE)

acc <- function(T){
  if (dim(T)[1]==2 & dim(T)[2]==2){
    acc <- (T[2,2]+T[1,1])/(sum(T))
  } else{
    acc <- 0
  }
  return(acc)
}

getMaxThreshold <- function(x, y){
  xs <- sort(x)
  return(xs[which.max(sapply(xs, function(xx){ return(acc(table(x>=xx, y))) } ))])
}



library(Matrix)
library(rhdf5)

projectName="chembl20";
rawDataFoldername="chembl20";

basePathname=paste0(Sys.getenv("HOME"), "/mydata/trgpred");
rawDataPathname=paste0(Sys.getenv("HOME"), "/mydata/raw");
rawDataPathname=file.path(rawDataPathname, rawDataFoldername);

projectPathname=file.path(basePathname, projectName);
chemPathname=file.path(projectPathname, "chemFeatures");
clusterPathname=file.path(chemPathname, "cl");
dchemPathname=file.path(chemPathname, "d");
schemPathname=file.path(chemPathname, "s");
trainPathname=file.path(projectPathname, "train");
runPathname=file.path(projectPathname, "run");
pythonPathname=Pathname=file.path(projectPathname, "dataPython");
pythonReducedPathname=Pathname=file.path(projectPathname, "dataPythonReduced");

sampleIdFilename=file.path(chemPathname, "SampleIdTable.txt")
targetSampleFilename=file.path(trainPathname, "t1.info")
predictTargetFilename=file.path(trainPathname, "trg1.info")
clusterSampleFilename=file.path(clusterPathname, "cl1.info")
predictSampleFilename=file.path(trainPathname, "p1.info")



actFileName=paste0(rawDataPathname, "/", "chemblActivities.txt")
adFileName=paste0(rawDataPathname, "/", "assayDescription.txt")
trainFileName=paste0(rawDataPathname, "/", "train.info")
targetsFilename=paste0(rawDataPathname, "/", "computeTargets.in")
trainWSName=paste0(rawDataPathname, "/", "trainTable.RData")



actTab=read.table(actFileName, header=FALSE, sep="\t", na.strings="\\N", stringsAsFactors=FALSE, quote="", comment.char="")
adTab=read.table(adFileName, header=FALSE, sep="\t", na.strings="\\N", stringsAsFactors=FALSE, quote="", comment.char="")
alabels=readMM(paste0(pythonReducedPathname, "/", "labelsWeakHard.mtx"))
rownames(alabels)=readLines(paste0(pythonReducedPathname, "/", "labelsWeakHard.cmpNames"))
colnames(alabels)=readLines(paste0(pythonReducedPathname, "/", "labelsWeakHard.targetNames"))
alabels=as.matrix(alabels)

mydata=h5dump(paste0(Sys.getenv("HOME"), "/mydata/trgpred/chembl20/results/allRes/ecfpTox.h5"))
pred=mydata[["mymatrix"]]
rownames(pred)=mydata$targets
colnames(pred)=mydata$samples
pred=t(pred)

actTabSel=actTab[(actTab[,2]%in%colnames(pred)),]

minimalCommonCompounds=10

assayToTarget=unique(actTabSel[,c(2L, 9L)])
assayDF=data.frame(assay=assayToTarget[,1], target=assayToTarget[,2], description=adTab[match(assayToTarget[,1], adTab[,1]), 3], stringsAsFactors=FALSE)
assayDF$surass=grepl("qHTS|HTS", assayDF$description)
#assayDF=assayDF[order(assayDF$target), ]
assayDF=assayDF[match(colnames(alabels), assayDF$assay),]

targets2=table(assayDF[,2], assayDF[,4])
consideredTargets=rownames(targets2)[apply(targets2, 1, function(x) all(x>0))]
consideredTargets=consideredTargets[!(consideredTargets=="22226")]



set.seed(1234567)

resultDF=list()
count=1
for(i in 1:length(consideredTargets)){
  currentTarget=consideredTargets[i]
  gtAssays=assayDF$assay[(assayDF$target==currentTarget)&(assayDF$surass==FALSE)]
  surAssays=assayDF$assay[(assayDF$target==currentTarget)&(assayDF$surass==TRUE)]
  
  for(j in 1:length(gtAssays)){
    for(k in 1:length(surAssays)){
      x=alabels[,gtAssays[j]]
      xm=rep(NA,length(x))
      xm[x==1|x==11]=0
      xm[x==3|x==13]=1
      
      z=alabels[,surAssays[k]]
      zm=rep(NA,length(z))
      zm[z==1|z==11]=0
      zm[z==3|z==13]=1
      
      conf1=table(factor(xm, levels=c(0,1)), factor(zm, levels=c(0,1)))
      
      if (sum(conf1)>=minimalCommonCompounds&(all(c(0,1)%in%zm))&(all(c(0,1)%in%xm))){
        x=alabels[,gtAssays[j]]
        xm=rep(NA,length(x))
        xm[x==1|x==11]=0
        xm[x==3|x==13]=1
        
        y=pred[, gtAssays[j]]
        
        idxEstimateThreshold=as.logical(rbinom(size=1, prob=0.5, n=length(xm)))
        myThresh=getMaxThreshold(y[!is.na(xm)&idxEstimateThreshold], xm[!is.na(xm)&idxEstimateThreshold])
        ym=as.numeric(y>myThresh)
        conf2=table(factor(xm[!idxEstimateThreshold], levels=c(0,1)), factor(ym[!idxEstimateThreshold], levels=c(0,1)))
        
        
        
        PT1=prop.test(conf1[1,1]+conf1[2,2], sum(conf1))
        PT2=prop.test(conf2[1,1]+conf2[2,2], sum(conf2))
        FT=fisher.test(matrix(c(conf1[1,2]+conf1[2,1], conf1[1,1]+conf1[2,2], conf2[1,2]+conf2[2,1], conf2[1,1]+conf2[2,2]), nrow=2))
        
        cat("i: ", i, "\n")
        cat("Assay:", gtAssays[j], "\n")
        cat("surAssay:", surAssays[k], "\n")
        cat("FisherPropTest:", FT$p.value, "\n")
        
        
        resultDF$assay[count]=gtAssays[j]
        resultDF$surass[count]=surAssays[k]
        resultDF$accSurrogate[count]=PT1$estimate
        resultDF$accSurrogateCIL[count]=PT1$conf1.int[1]
        resultDF$accSurrogateCIR[count]=PT1$conf1.int[2]
        resultDF$TN1[count]=conf1[1,1]
        resultDF$FN1[count]=conf1[2,1]
        resultDF$FP1[count]=conf1[1,2]
        resultDF$TP1[count]=conf1[2,2]
        resultDF$accCOMP[count]=PT2$estimate
        resultDF$accCOMPCIL[count]=PT2$conf2.int[1]
        resultDF$accCOMPCIR[count]=PT2$conf2.int[2]
        resultDF$TN2[count]=conf2[1,1]
        resultDF$FN2[count]=conf2[2,1]
        resultDF$FP2[count]=conf2[1,2]
        resultDF$TP2[count]=conf2[2,2]
        resultDF$pval[count]=FT$p.value
        resultDF$desc1[count]=adTab[match(gtAssays[j], adTab[, 1]), 3]
        resultDF$desc2[count]=adTab[match(surAssays[k], adTab[, 1]), 3]
        
        count=count+1
      }
    }
  }
}
resultDF=as.data.frame(resultDF, stringsAsFactors=FALSE)

selAssays=data.frame(
  assay=c("CHEMBL1614310", "CHEMBL1737868", "CHEMBL1614355", "CHEMBL1614479", "CHEMBL1614539", "CHEMBL1614016", "CHEMBL1613806", "CHEMBL1614105", "CHEMBL1614197", "CHEMBL1737863", "CHEMBL1738575", "CHEMBL1741321", "CHEMBL1741323", "CHEMBL1741324", "CHEMBL1741325", "CHEMBL1794393", "CHEMBL1909134", "CHEMBL1909135", "CHEMBL1909136", "CHEMBL1909138", "CHEMBL1909200", "CHEMBL1963940"),
  surass=c("CHEMBL1614544", "CHEMBL1738317", "CHEMBL1614052", "CHEMBL1614052", "CHEMBL1614052", "CHEMBL1794352", "CHEMBL1613949", "CHEMBL1614290", "CHEMBL1614087", "CHEMBL1614255", "CHEMBL1614247", "CHEMBL1614110", "CHEMBL1613777", "CHEMBL1613886", "CHEMBL1614027", "CHEMBL1614512", "CHEMBL1613777", "CHEMBL1614027", "CHEMBL1614110", "CHEMBL1614108", "CHEMBL1614521", "CHEMBL1794352")
)

resultDFSel=resultDF[match(paste0(selAssays$assay, selAssays$surass), paste0(resultDF$assay, resultDF$surass)),]


