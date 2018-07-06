/*
Copyright (C) 2018 Andreas Mayr
Licensed under GNU General Public License v3.0 (see http://www.bioinf.jku.at/research/lsc/LICENSE and https://github.com/ml-jku/lsc/blob/master/LICENSE)
*/

#include <omp.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <list>
#include <map>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <algorithm>

using namespace std;



signed char tanimotoSC(long* featuresPred, long beginPredFeat, long endPredFeat, long* featuresTrain, long beginTrainFeat, long endTrainFeat) {
  long countPred=beginPredFeat;
  long countTrain=beginTrainFeat;
  long out=0;

  if((countPred==endPredFeat)&&(countTrain==endTrainFeat))
    return(0.0);

  while((countPred<endPredFeat)&&(countTrain<endTrainFeat)) {
    if(featuresTrain[countTrain]==featuresPred[countPred]) {
      out++;
      countTrain++;
      countPred++;
    }
    else if(featuresTrain[countTrain]<featuresPred[countPred]) {
      countTrain++;
    }
    else if(featuresTrain[countTrain]>featuresPred[countPred]) {
      countPred++;
    }
  }
  signed char retval=(signed char) ((out*100L)/(endTrainFeat+endPredFeat-beginTrainFeat-beginPredFeat-out));
  return retval;
}



int main(int argc, char** argv) {
  if(sizeof(long)<8) {
    printf("This program is optimized for machines with at least 8 byte longs!\n");
    return -1;     
  }
  string basePathname(getenv("HOME")); basePathname=basePathname+"/mydata/trgpred/";
  
  
  
  struct stat sb;
  int statRet;
  string dataTypeInfoFilename;
  ifstream dataTypeInfoFile;
  string dataType;
  
  
  
  if(argc!=4) {
    printf("Usage: clusterMinFull projectName fpVersionName maxProc\n");
    return -1; 
  }
  
  
  string progName("clusterMinFull");
  
  string projectName(argv[1]);
  string projectPathname=basePathname+projectName+"/";
  string chemPathname=projectPathname+"chemFeatures/";
  string clusterPathname=chemPathname+"cl/";
  string dchemPathname=chemPathname+"d/";
  string schemPathname=chemPathname+"s/";
  string trainPathname=projectPathname+"train/";
  string runPathname=projectPathname+"run/";
  string sampleIdFilename=chemPathname+"SampleIdTable.txt";
  statRet=(stat(sampleIdFilename.c_str(), &sb)==-1);
  if(statRet==-1) {
    fprintf(stderr, "The database '%s' does either not exist or is not complete!\n", projectPathname.c_str());
    exit(-1);
  }
  
  string fpVersionName(argv[2]);
  string sfeatureDBPathname(schemPathname); sfeatureDBPathname=sfeatureDBPathname+fpVersionName+"/";
  string intToExtMappingFilename=sfeatureDBPathname+"IntToExtMappingTable.bin";
  string sampleFilename=sfeatureDBPathname+"fpSampleTable.bin";
  string featureIntFilename=sfeatureDBPathname+"fpFeatureTableInt.bin";
  string featureExtFilename=sfeatureDBPathname+"fpFeatureTableExt.bin";
  string featureCountFilename=sfeatureDBPathname+"fpFeatureCountTable.bin";
  dataTypeInfoFilename=sfeatureDBPathname+"dt.txt";
  statRet=(stat(intToExtMappingFilename.c_str(), &sb)==-1);
  statRet=statRet||(stat(sampleFilename.c_str(), &sb)==-1);
  statRet=statRet||(stat(featureIntFilename.c_str(), &sb)==-1);
  statRet=statRet||(stat(featureExtFilename.c_str(), &sb)==-1);
  statRet=statRet||(stat(featureCountFilename.c_str(), &sb)==-1);
  if(statRet==-1) {
    fprintf(stderr, "The database '%s' does either not exist or is not complete!\n", sfeatureDBPathname.c_str());
    exit(-1);
  }
  dataTypeInfoFile.open(dataTypeInfoFilename.c_str(), ifstream::in);
  getline(dataTypeInfoFile, dataType);
  dataTypeInfoFile.close();
  if(sizeof(long)!=atoi(dataType.c_str())) {
    fprintf(stderr, "The system, which was used to create the database is not compatible with the current system!");
    exit(-1);
  }
  
  string clusterResPathname=clusterPathname+progName;
  clusterResPathname=clusterResPathname+"/";
  if(mkdir(clusterResPathname.c_str(), S_IRWXU)==-1) {
    fprintf(stderr, "The directory '%s' either already exists or it is an invalid directory!\n", clusterResPathname.c_str());
    exit(-1);
  }
  
  
  int maxProc;
  if(sscanf(argv[3], "%d", &maxProc)== EOF) {
    fprintf(stderr, "maxProc must be a postive integer!\n");
    exit(-1);
  }
 
  printf("Running cluster with the following parameters:\n");
  printf("projectName             %30s\n", projectName.c_str());
  printf("fpVersionName           %30s\n", fpVersionName.c_str());
  printf("maxProc:                %30d\n", maxProc);
  
  
  
  FILE* featureFile=fopen(featureIntFilename.c_str(), "rb");
  fseek(featureFile, 0, SEEK_END);
  long featureSize=ftell(featureFile);
  rewind(featureFile);
  long* features=(long*)malloc(featureSize);
  if(features==NULL) {
    fprintf(stderr, "Too few main memory. Program will terminate!\n");
    exit(-1);
  }
  fread(features, 1, featureSize, featureFile);
  fclose(featureFile);

  FILE* sampleFile=fopen(sampleFilename.c_str(), "rb");
  fseek(sampleFile, 0, SEEK_END);
  long sampleSize=ftell(sampleFile);
  rewind(sampleFile);
  long* samples=(long*)malloc(sampleSize+(long)(sizeof(long)));
  if(samples==NULL) {
    fprintf(stderr, "Too few main memory. Program will terminate!\n");
    exit(-1);
  }
  fread(samples, 1, sampleSize, sampleFile);
  fclose(sampleFile);

  FILE* intToExtMappingFile=fopen(intToExtMappingFilename.c_str(), "rb");
  fseek(intToExtMappingFile, 0, SEEK_END);
  long intToExtMappingSize=ftell(intToExtMappingFile);
  rewind(intToExtMappingFile);
  /*long* internalFeatNrToExternalFeatNr=(long*)malloc(intToExtMappingSize);
  if(internalFeatNrToExternalFeatNr==NULL) {
    fprintf(stderr, "Too few main memory. Program will terminate!\n");
    exit(-1);
  }
  fread(internalFeatNrToExternalFeatNr, 1, intToExtMappingSize, intToExtMappingFile);*/
  fclose(intToExtMappingFile);

  long featureNr=featureSize/sizeof(long);
  long sampleNr=sampleSize/sizeof(long);
  samples[sampleNr]=featureNr;
  long nrDistinctFeatures=intToExtMappingSize/sizeof(long);
  
  
  
  //sampleNr=20000L;
  long batchSize=10000L;
  long nrClusterings=100L;
  double stepSize=(double)(100.0/((double)nrClusterings));
  long** cluster=(long**)malloc(sizeof(long*)*nrClusterings);
  long* clusterStore=(long*)malloc(sizeof(long)*sampleNr*nrClusterings);
  long** clusterBegin=(long**)malloc(sizeof(long*)*nrClusterings);
  long* clusterBeginStore=(long*)malloc(sizeof(long)*sampleNr*nrClusterings);
  long** clusterEnd=(long**)malloc(sizeof(long*)*nrClusterings);
  long* clusterEndStore=(long*)malloc(sizeof(long)*sampleNr*nrClusterings);
  long** sampleNext=(long**)malloc(sizeof(long*)*nrClusterings);
  long* sampleNextStore=(long*)malloc(sizeof(long)*sampleNr*nrClusterings);
  
  for(long i=0; i<nrClusterings; i++) {
    //cluster[i]=(long*)malloc(sizeof(long)*sampleNr);
    /*cluster[i]=&(clusterStore[i*sampleNr]);
    clusterBegin[i]=&(clusterBeginStore[i*sampleNr]);
    clusterEnd[i]=&(clusterEndStore[i*sampleNr]);
    sampleNext[i]=&(sampleNextStore[i*sampleNr]);*/
    
    cluster[i]=(long*)malloc(sampleNr*sizeof(long));
    clusterBegin[i]=(long*)malloc(sampleNr*sizeof(long));
    clusterEnd[i]=(long*)malloc(sampleNr*sizeof(long));
    sampleNext[i]=(long*)malloc(sampleNr*sizeof(long));
    
    for(long j=0; j<sampleNr; j++) {
      cluster[i][j]=j;
      sampleNext[i][j]=-1L;
      clusterBegin[i][j]=j;
      clusterEnd[i][j]=j;
    }
  }
  
  long nrBatches=(sampleNr/batchSize);
  long resBatches=sampleNr-nrBatches*batchSize;
  nrBatches++;
  fprintf(stderr, "nrBat/resBat: %ld, %ld\n", nrBatches, resBatches);
  long *batches=(long*) malloc(sizeof(long)*(nrBatches+1L));
  batches[0L]=0L;
  for(long i=1L; i<nrBatches; i++) batches[i]=batchSize;
  batches[nrBatches]=resBatches;
  for(long i=0L; i<nrBatches; i++) batches[i+1L]=batches[i+1L]+batches[i];
  
  signed char **simVec=(signed char**)malloc(sizeof(signed char*)*batchSize);
  signed char *simVecStore=(signed char*)malloc(sizeof(signed char)*sampleNr*batchSize);
  for(long i=0L; i<batchSize; i++) simVec[i]=&(simVecStore[i*sampleNr]);

  #ifdef multiproc
  omp_set_num_threads(maxProc);
  #endif
  
  for(long i=0L; i<nrBatches; i++) {
    long start=batches[i];
    long end=batches[i+1L];
    long length=end-start;
    
    #ifdef multiproc
    #pragma omp parallel for schedule(runtime)
    #endif
    for(long j=0L; j<length; j++) {
      long elem=start+j+1L;
      if(elem%100L==0L)
        fprintf(stderr, "%ld/%ld\n", elem, sampleNr);
      long beginFeat1=samples[start+j];
      long endFeat1=samples[start+j+1L];
      for(long k=0L; k<sampleNr; k++) {
        long beginFeat2=samples[k];
        long endFeat2=samples[k+1L];
        signed char tm=tanimotoSC(features, beginFeat1, endFeat1, features, beginFeat2, endFeat2);
        simVec[j][k]=tm;

      }
      //fprintf(stderr, "j: %ld\n", j);
    }
    fprintf(stderr, "here\n");
    
    
    
    #ifdef multiproc
    #pragma omp parallel for schedule(runtime)
    #endif
    for(long l=0; l<nrClusterings; l++) {
      signed char thresh=(signed char)(((double)l)*stepSize);
      
      for(long j=0L; j<length; j++) {
        if(j%1000==0L)
          fprintf(stderr, "%ld: %ld/%ld\n", l, j, length);
        long oldClust=cluster[l][start+j];
        long renameClust=oldClust;
        for(long k=0L; k<sampleNr; k++) {
          if((simVec[j][k]>thresh)&&(cluster[l][k]<renameClust)) {
            renameClust=cluster[l][k];
          }
        }
        for(long k=0L; k<sampleNr; k++) {
          if(simVec[j][k]>thresh) {
            long simClust=cluster[l][k];
            if((simClust!=renameClust)&&(clusterBegin[l][simClust]!=-1L)) {
              //cluster[l][k]=renameClust;
              long runInd=clusterBegin[l][simClust];
              while(runInd!=-1L) {
                cluster[l][runInd]=renameClust;
                runInd=sampleNext[l][runInd];
              }
              sampleNext[l][clusterEnd[l][renameClust]]=clusterBegin[l][simClust];
              clusterEnd[l][renameClust]=clusterEnd[l][simClust];
              clusterBegin[l][simClust]=-1L;
              clusterEnd[l][simClust]=-1L;
            }
          }
        }
        if((oldClust!=renameClust)&&(clusterBegin[l][oldClust]!=-1L)) {
          long simClust=oldClust;
          long runInd=clusterBegin[l][simClust];
          while(runInd!=-1L) {
            cluster[l][runInd]=renameClust;
            runInd=sampleNext[l][runInd];
          }
          sampleNext[l][clusterEnd[l][renameClust]]=clusterBegin[l][simClust];
          clusterEnd[l][renameClust]=clusterEnd[l][simClust];
          clusterBegin[l][simClust]=-1L;
          clusterEnd[l][simClust]=-1L;
        }
      }
    }
  }
  
  free(simVecStore);
  free(simVec);
  free(batches);
  
  char* outFileClusterBuffer=(char*)malloc(sizeof(char)*1024*1024);
  
  for(long l=0; l<nrClusterings; l++) {
    std::stringstream ss;
    ss << l;
    string outFilenameCluster=clusterResPathname+string("clustering_")+ss.str()+".txt";
    
    FILE* outFileCluster=fopen(outFilenameCluster.c_str(), "w");
    setbuffer(outFileCluster, outFileClusterBuffer, 1024*1024);
    for(long k=0L; k<sampleNr; k++) {
      fprintf(outFileCluster, "%ld, %ld\n", cluster[l][k], k);
    }
    fclose(outFileCluster);
  }
  
  free(outFileClusterBuffer);
  free(clusterStore);
  free(cluster);


  
  //free(internalFeatNrToExternalFeatNr);
  free(samples);
  free(features);
  
  
  
  printf("clusterMinFull terminated successfully!\n");
  
  return 0;
}
