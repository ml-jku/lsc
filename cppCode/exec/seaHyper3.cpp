/*
Copyright (C) 2018 Andreas Mayr
Licensed under GNU General Public License v3.0 (see http://www.bioinf.jku.at/research/lsc/LICENSE and https://github.com/ml-jku/lsc/blob/master/LICENSE)
*/



#ifdef multiproc
#include <omp.h>
#endif

#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <gsl/gsl_randist.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <list>
#include <vector>
#include <map>
#include <algorithm>
#include <string>
#include <sstream>
#include <limits>
#include <fstream>
#include <set>
#include <gsl/gsl_sort.h>
#include <gsl/gsl_statistics.h>
#include <iostream>
#include <libconfig.h++>
#include "compoundData.h"
#include "targetPipeline.h"
#include "sums.h"

using namespace std;
using namespace libconfig;



int main(int argc, char** argv) {
  
  if(sizeof(long)<8) {
    printf("This program is optimized for machines with at least 8 byte longs!  The program will terminate!\n");
    return -1;
  }
  string basePathname(getenv("HOME")); basePathname=basePathname+"/mydata/trgpred/";
  
  
  
  struct stat sb;
  int statRet;
  
  if(argc!=3) {
    printf("Usage: seaHyper config maxProc\n");
    return -1;
  }
  
  
  
  string configFilename(argv[1]);
  Config cfg;

  try {
    cfg.readFile(configFilename.c_str());
  }
  catch(const FileIOException &fioex) {
    std::cout << "I/O error while reading file." << std::endl;
    return(EXIT_FAILURE);
  }
  catch(const ParseException &pex) {
    std::cout << "Parse error at " << pex.getFile() << ":" << pex.getLine() << " - " << pex.getError() << std::endl;
    return(EXIT_FAILURE);
  }
  
  
  
  string projectName = cfg.lookup("project");
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
    fprintf(stderr, "The database '%s' does either not exist or is not complete! - error %d\n", projectPathname.c_str(), __LINE__);
    exit(-1);
  }
  
  
  
  string trainInfo = cfg.lookup("trainInfo");
  string targetSampleFilename=trainPathname+trainInfo+".info";
  statRet=stat(targetSampleFilename.c_str(), &sb);
  if((statRet==-1)) {
    fprintf(stderr, "The file '%s' does not exist.! - error %d\n", targetSampleFilename.c_str(), __LINE__);
    exit(-1);
  }
  
  
    
  string clusterInfo = cfg.lookup("clusterInfo");
  string clusterSampleFilename=clusterPathname+clusterInfo+".info";
  statRet=stat(clusterSampleFilename.c_str(), &sb);
  if((statRet==-1)) {
    fprintf(stderr, "The file '%s' does not exist.! - error %d\n", clusterSampleFilename.c_str(), __LINE__);
    exit(-1);
  }
  
  
  
  string targetInfo = cfg.lookup("targetInfo");
  string predictTargetFilename=trainPathname+targetInfo+".info";
  statRet=stat(predictTargetFilename.c_str(), &sb);
  if((statRet==-1)) {
    fprintf(stderr, "The file '%s' does not exist.! - error %d\n", predictTargetFilename.c_str(), __LINE__);
    exit(-1);
  }
  
  
  
  int mode = cfg.lookup("mode");
  
  
  
  string predSavename = cfg.lookup("predSavename");
  string predSavePathname=runPathname+predSavename+"/";
  /*if(mkdir(predSavePathname.c_str(), S_IRWXU)==-1) {
    fprintf(stderr, "The directory '%s' either already exists or it is an invalid directory!\n", predSavePathname.c_str());
    exit(-1);
  }*/
  
  
  
  SampleData sample(0L, sampleIdFilename);
  long sampleNr=sample.sampleNr;
  string* sampleIndexToSampleId=sample.sampleIndexToSampleId;
  map<string, long>& sampleIdToSampleIndex=sample.sampleIdToSampleIndex;  
  
  
  
  std::map<string, SparseFeatureData* > sparseFeatureMap;
  for(int i=0; i<cfg.lookup("sfeatures").getLength(); i++)
    sparseFeatureMap[cfg.lookup("sfeatures")[i]]=new SparseFeatureData(i, schemPathname, cfg.lookup("sfeatures")[i]);
  
  std::map<string, DenseFeatureData* > denseFeatureMap;
  for(int i=0L; i<cfg.lookup("dfeatures").getLength(); i++)
    denseFeatureMap[cfg.lookup("dfeatures")[i]]=new DenseFeatureData(i, dchemPathname, cfg.lookup("dfeatures")[i]);
  
  /*std::map<string, SVMSFeature*>::iterator it1;
  for (it1=rawSFeaturesMap.begin(); it1!=rawSFeaturesMap.end(); ++it1) {
    string mystr=it1->first;
    fprintf(stderr, "%s\n", mystr.c_str());
  }
  std::map<string, SVMDFeature*>::iterator it2;
  for (it2=rawDFeaturesMap.begin(); it2!=rawDFeaturesMap.end(); ++it2) {
    string mystr=it2->first;
    fprintf(stderr, "%s\n", mystr.c_str());
  }*/
  
  
  
  std::vector<SparseFeatureKernel*> sparseFeatureKernelVec;
  std::map<string, SparseFeatureKernel*> sparseFeatureKernelMap;
  for(int i=0L; i<cfg.lookup("sfeatureKernels").getLength(); i++) {
    string ktstr=cfg.lookup("sfeatureKernels")[i][2];
    KernelType kt;
    if(ktstr=="TAN")
      kt=TAN;
    else if(ktstr=="TAN2")
      kt=TAN2;
    else if(ktstr=="LIN")
      kt=LIN;
    else if(ktstr=="GAUSS")
      kt=GAUSS;
    else
      fprintf(stderr, "ERROR1\n");
    
    SparseFeatureKernel* svmsfk=new SparseFeatureKernel(i, sparseFeatureMap[cfg.lookup("sfeatureKernels")[i][1]], kt);
    
    sparseFeatureKernelMap[cfg.lookup("sfeatureKernels")[i][0]]=svmsfk;
    sparseFeatureKernelVec.push_back(svmsfk);
  }
  
  std::vector<DenseFeatureKernel*> denseFeatureKernelVec;
  std::map<string, DenseFeatureKernel*> denseFeatureKernelMap;
  for(int i=0L; i<cfg.lookup("dfeatureKernels").getLength(); i++) {
    string ktstr=cfg.lookup("dfeatureKernels")[i][2];
    KernelType kt;
    if(ktstr=="TAN")
      kt=TAN;
    else if(ktstr=="TANS")
      kt=TANS;
    else if(ktstr=="TAN2")
      kt=TAN2;
    else if(ktstr=="LIN")
      kt=LIN;
    else if(ktstr=="GAUSS")
      kt=GAUSS;
    else
      fprintf(stderr, "ERROR2\n");

    
    DenseFeatureKernel* svmdfk=new DenseFeatureKernel(i, denseFeatureMap[cfg.lookup("dfeatureKernels")[i][1]], kt);
    
    denseFeatureKernelMap[cfg.lookup("dfeatureKernels")[i][0]]=svmdfk;
    denseFeatureKernelVec.push_back(svmdfk);
  }
  
  /*std::map<string, SVMSFeatureKernel*>::iterator it3;
  for (it3=sFeatureTypesMap.begin(); it3!=sFeatureTypesMap.end(); ++it3) {
    string mystr=it3->first;
    fprintf(stderr, "%s\n", mystr.c_str());
  }
  
  std::map<string, SVMDFeatureKernel*>::iterator it4;
  for (it4=dFeatureTypesMap.begin(); it4!=dFeatureTypesMap.end(); ++it4) {
    string mystr=it4->first;
    fprintf(stderr, "%s\n", mystr.c_str());
  }*/
  
  
  
  FeatureKernelCollection defaultFeature;
  std::vector<FeatureKernelCollection> paramFeatureTypes;
  for(int i=0L; i<cfg.lookup("paramFeatures").getLength(); i++) {
    int isDefault=cfg.lookup("paramFeatures")[i][0];
    
    std::vector<DenseFeatureKernel*> dvec;
    for(int j=0L; j<cfg.lookup("paramFeatures")[i][1].getLength(); j++)
      dvec.push_back(denseFeatureKernelMap[cfg.lookup("paramFeatures")[i][1][j]]);

    std::vector<SparseFeatureKernel*> svec;
    for(int j=0L; j<cfg.lookup("paramFeatures")[i][2].getLength(); j++)
      svec.push_back(sparseFeatureKernelMap[cfg.lookup("paramFeatures")[i][2][j]]);
    
    string kcstr=cfg.lookup("paramFeatures")[i][3];
    KernelComb kc;
    if(kcstr=="TANCOMB")
      kc=TANCOMB;
    else if(kcstr=="TANCOMB2")
      kc=TANCOMB2;
    else if(kcstr=="TANSCOMB")
      kc=TANSCOMB;
    else if(kcstr=="LINCOMB")
      kc=LINCOMB;
    else if(kcstr=="GAUSSCOMB")
      kc=GAUSSCOMB;
    else
      fprintf(stderr, "ERROR3\n");
    
    long norm=cfg.lookup("paramFeatures")[i][4];
    
    long sim=cfg.lookup("paramFeatures")[i][5];
    
    paramFeatureTypes.push_back(FeatureKernelCollection(i, dvec, svec, kc, norm, sim));
    
    if(isDefault==1)
      defaultFeature=FeatureKernelCollection(i, dvec, svec, kc, norm, sim);
  }
  
  
  
  int maxProc;
  if(sscanf(argv[2], "%d", &maxProc)== EOF) {
    fprintf(stderr, "maxProc must be a postive integer!\n");
    exit(-1);
  }
  
  
  
  #ifdef multiproc
  omp_set_num_threads(maxProc);
  #endif
  
  
  
  srand(15123423);
  long statSamples=20;
  long* myrand=(long*) malloc(sizeof(long)*statSamples);
  
  double* simMatrixStatStore=(double*) malloc(sizeof(double)*statSamples*statSamples);
  double** simMatrixStat=(double**) malloc(sizeof(double*)*statSamples);
  double* simMatrixStatStoreH=(double*) malloc(sizeof(double)*statSamples*statSamples);
  double** simMatrixStatH=(double**) malloc(sizeof(double*)*statSamples);
  
  for(long i=0L; i<statSamples; i++) {
    myrand[i]=rand()%sampleNr;
    simMatrixStat[i]=&(simMatrixStatStore[i*statSamples]);
    simMatrixStatH[i]=&(simMatrixStatStoreH[i*statSamples]);
  }
  
  std::vector<SEAHyperParam> paramComb;
  for(long i=0L; i<paramFeatureTypes.size(); i++) {
    paramComb.push_back(SEAHyperParam(paramFeatureTypes[i]));
  }
  std::sort(paramComb.begin(), paramComb.end(), paramSortSEA);
  
  std::vector<long> featureParamInd;
  long curId=paramComb[0].fkc.id;
  featureParamInd.push_back(0L);
  for(long i=0L; i<paramComb.size(); i++) {
    if(curId!=paramComb[i].fkc.id) {
      featureParamInd.push_back(i);
      curId=paramComb[i].fkc.id;
    }
  }
  featureParamInd.push_back(paramComb.size());
  
  free(simMatrixStatH);
  free(simMatrixStatStoreH);
  free(simMatrixStat);
  free(simMatrixStatStore);
  free(myrand);
  
  
  
  printf("Running SEA with the following parameters:\n");
  printf("project:              %30s\n", projectName.c_str());
  printf("trainInfo:            %30s\n", trainInfo.c_str());
  printf("clusterInfo:          %30s\n", clusterInfo.c_str());
  printf("targetInfo:           %30s\n", targetInfo.c_str());
  printf("mode:                 %30d\n", mode);
  printf("predSavename:         %30s\n", predSavename.c_str());
  printf("maxProc:              %30d\n", maxProc);
  
  string dbInformationFilename=predSavePathname+"settings.txt";
  ofstream dbInformationFile;
  dbInformationFile.open(dbInformationFilename.c_str());  
  dbInformationFile << "SEA" << endl;
  dbInformationFile << "project" << endl;
  dbInformationFile << projectName << endl;
  dbInformationFile << "trainInfo" << endl;
  dbInformationFile << trainInfo << endl;
  dbInformationFile << "clusterInfo" << endl;
  dbInformationFile << clusterInfo << endl;
  dbInformationFile << "targetInfo" << endl;
  dbInformationFile << targetInfo << endl;
  dbInformationFile << "mode" << endl;
  dbInformationFile << mode << endl;
  dbInformationFile << "predSavename" << endl;
  dbInformationFile << predSavename << endl;
  dbInformationFile.close();
  
  
  
  stat(targetSampleFilename.c_str(), &sb);
  char* targetSampleFileBuffer=(char*)malloc(sb.st_size);
  if(targetSampleFileBuffer==NULL) {
    fprintf(stderr, "Too few main memory. Program will terminate! - error %d\n", __LINE__);
    exit(-1);
  }
  FILE* targetSampleFile=fopen(targetSampleFilename.c_str(), "r");
  setbuffer(targetSampleFile, targetSampleFileBuffer, sb.st_size);
  long targetSampleDBSize=0L;
  long keySize=0L;
  long maxKeySize=0L;
  char ch;
  while((ch=fgetc(targetSampleFile))!=EOF) {
    keySize++;
    if(ch=='\n') {
      targetSampleDBSize++;
      if(keySize>maxKeySize)
        maxKeySize=keySize;
      keySize=0L;
    }
  }
  rewind(targetSampleFile);
  
  char* key=(char*)malloc(sizeof(char)*(maxKeySize+1));
  
  TrainRec* targetSampleDB=new TrainRec[targetSampleDBSize];
  if(targetSampleDB==NULL) {
    fprintf(stderr, "Too few main memory. Program will terminate! - error %d\n", __LINE__);
    remove(dbInformationFilename.c_str());
    exit(-1);
  }
  for(long i=0L; i<targetSampleDBSize; i++) {
    fscanf(targetSampleFile, "%ld", &targetSampleDB[i].label);
    fscanf(targetSampleFile, "%ld", &targetSampleDB[i].sampleNr);
    fscanf(targetSampleFile, "%s", key);
    targetSampleDB[i].target.assign(key);
  }
  fclose(targetSampleFile);
  free(targetSampleFileBuffer);
  free(key);
  qsort(targetSampleDB, targetSampleDBSize, sizeof(TrainRec), compareTrainRec);

  list<string> targetNameList;
  list<long> targetIndList;
  list<long> targetSizeList;

  {
    long targetSampleDBSizeNew=0L;
    long i=0L;
    string currentTargetName=targetSampleDB[i].target;
    targetNameList.push_back(currentTargetName);
    targetIndList.push_back(targetSampleDBSizeNew);
    targetSizeList.push_back(0L);
    while(!(targetSampleDB[i].label==1L||targetSampleDB[i].label==3L)) {
      if(currentTargetName!=targetSampleDB[i].target) {
        currentTargetName=targetSampleDB[i].target;
        targetNameList.push_back(currentTargetName);
        targetIndList.push_back(targetSampleDBSizeNew);
        targetSizeList.push_back(0L);
      }
      i++;
    }
    if(i<targetSampleDBSize) {
      targetSampleDB[targetSampleDBSizeNew]=targetSampleDB[i];
      targetSampleDBSizeNew++;
      targetSizeList.back()++;
      for(i=i+1; i<targetSampleDBSize; i++) {
        if(currentTargetName!=targetSampleDB[i].target) {
          currentTargetName=targetSampleDB[i].target;
          targetNameList.push_back(currentTargetName);
          targetIndList.push_back(targetSampleDBSizeNew);
          targetSizeList.push_back(0L);
        }
        if(targetSampleDB[i].label==1L||targetSampleDB[i].label==3L) {
          if(!((targetSampleDB[targetSampleDBSizeNew-1].sampleNr==targetSampleDB[i].sampleNr)&&(targetSampleDB[targetSampleDBSizeNew-1].target==targetSampleDB[i].target))) {
            targetSampleDB[targetSampleDBSizeNew]=targetSampleDB[i];
            targetSampleDBSizeNew++;
            targetSizeList.back()++;
          }
        }
      }
    }
    targetSampleDBSize=targetSampleDBSizeNew;
  }

  std::vector<string> targetNameVec(targetNameList.begin(), targetNameList.end());
  std::vector<long> targetIndVec(targetIndList.begin(), targetIndList.end());
  std::vector<long> targetSizeVec(targetSizeList.begin(), targetSizeList.end());
  long nrTargets=targetIndVec.size();
  long* indTargetSampleDB=getSampleIndices(targetSampleDB, targetSampleDBSize);
  long maxTargetSize=*std::max_element(targetSizeList.begin(), targetSizeList.end());
  
  
  
  stat(clusterSampleFilename.c_str(), &sb);
  char* clusterSampleFileBuffer=(char*)malloc(sb.st_size);
  if(clusterSampleFileBuffer==NULL) {
    fprintf(stderr, "Too few main memory. Program will terminate! - error %d\n", __LINE__);
    exit(-1);
  }
  FILE* clusterSampleFile=fopen(clusterSampleFilename.c_str(), "r");
  setbuffer(clusterSampleFile, clusterSampleFileBuffer, sb.st_size);
  long clusterSampleDBSize=0L;
  while((ch=fgetc(clusterSampleFile))!=EOF) {
    if(ch=='\n') {
      clusterSampleDBSize++;
    }
  }
  rewind(clusterSampleFile);

  ClusterRec* clusterSampleDB=new ClusterRec[clusterSampleDBSize];
  if(clusterSampleDB==NULL) {
    fprintf(stderr, "Too few main memory. Program will terminate! - error %d\n", __LINE__);
    remove(dbInformationFilename.c_str());
    exit(-1);
  }
  for(long i=0L; i<clusterSampleDBSize; i++) {
    fscanf(clusterSampleFile, "%ld", &clusterSampleDB[i].clusterNr);
    fscanf(clusterSampleFile, "%ld", &clusterSampleDB[i].sampleNr);
  }
  fclose(clusterSampleFile);
  free(clusterSampleFileBuffer);
  qsort(clusterSampleDB, clusterSampleDBSize, sizeof(ClusterRec), compareClusterRec);

  list<long> clusterNrList;
  list<long> clusterIndList;
  list<long> clusterSizeList;

  {
    long i=0L;
    long currentClusterNr=clusterSampleDB[i].clusterNr;
    clusterNrList.push_back(currentClusterNr);
    clusterIndList.push_back(i);
    clusterSizeList.push_back(0L);
    clusterSizeList.back()++;
    for(i=1; i<clusterSampleDBSize; i++) {
      if(currentClusterNr!=clusterSampleDB[i].clusterNr) {
        currentClusterNr=clusterSampleDB[i].clusterNr;
        clusterNrList.push_back(currentClusterNr);
        clusterIndList.push_back(i);
        clusterSizeList.push_back(0L);
      }
      clusterSizeList.back()++;
    }
  }

  std::vector<long> clusterNrVec(clusterNrList.begin(), clusterNrList.end());
  std::vector<long> clusterIndVec(clusterIndList.begin(), clusterIndList.end());
  std::vector<long> clusterSizeVec(clusterSizeList.begin(), clusterSizeList.end());
  long nrClusters=clusterIndVec.size();
  
  
  
  //set<long> predictTargetIndSet;
  std::vector<long> predictTargetIndSet;
  string line;
  ifstream predictTargetFile (predictTargetFilename.c_str());
  if (predictTargetFile.is_open()) {
    while(predictTargetFile.good()) {
      getline (predictTargetFile, line);
      if(line.size()>0) {
        long ind=std::lower_bound (targetNameVec.begin(), targetNameVec.end(), line)-targetNameVec.begin();
        //predictTargetIndSet.insert(ind);
        predictTargetIndSet.push_back(ind);
      }
    }
    predictTargetFile.close();
  }
  std::vector<long> predictTargetIndVec(predictTargetIndSet.begin(), predictTargetIndSet.end());

  for(long predictTargetIndInd=0; predictTargetIndInd<predictTargetIndVec.size(); predictTargetIndInd++) {
    long targetInd=predictTargetIndVec[predictTargetIndInd];
    string currentPredSavePathname=predSavePathname+targetNameVec[targetInd]+"/";
    if(mkdir(currentPredSavePathname.c_str(), S_IRWXU)==-1) {
      fprintf(stderr, "The directory '%s' either already exists or it is an invalid directory! - error %d\n", currentPredSavePathname.c_str(), __LINE__);
      exit(-1);
    }
  }
  
  
  
  for(long predictTargetIndInd=0; predictTargetIndInd<predictTargetIndVec.size(); predictTargetIndInd++)
  {
    {
      long predictTargetIndVecSize=predictTargetIndVec.size();
      printf("t(%ld)/t(%ld)\n",  predictTargetIndInd, predictTargetIndVecSize);

      long targetInd=predictTargetIndVec[predictTargetIndInd];
      
      long targetIndStart=targetIndVec[targetInd];
      long targetSize=targetSizeVec[targetInd];
      long targetIndEnd=targetIndStart+targetSize;
      
      printf("targetSize: %ld\n",  targetSize);
      //if(targetSize<100)
      //  goto endLoopTarget;
      
      
      
      float* sSimMatrixStore2=(float*) malloc(sizeof(float)*targetSize*targetSize*sparseFeatureKernelVec.size());
      float** sSimMatrixStore1=(float**) malloc(sizeof(float*)*targetSize*sparseFeatureKernelVec.size());
      float*** sSimMatrix=(float***) malloc(sizeof(float**)*sparseFeatureKernelVec.size());
      float* dSimMatrixStore2=(float*) malloc(sizeof(float)*targetSize*targetSize*denseFeatureKernelVec.size());
      float** dSimMatrixStore1=(float**) malloc(sizeof(float*)*targetSize*denseFeatureKernelVec.size());
      float*** dSimMatrix=(float***) malloc(sizeof(float**)*denseFeatureKernelVec.size());
      
      
      
      long simMatrixStoreSize2;
      long simMatrixStoreSize1;
      
      simMatrixStoreSize2=0L;
      simMatrixStoreSize1=0L;
      for(long i=0L; i<sparseFeatureKernelVec.size(); i++) {
        sSimMatrix[i]=&(sSimMatrixStore1[simMatrixStoreSize1]);
        for(long j=0L; j<targetSize; j++) {
          sSimMatrix[i][j]=&(sSimMatrixStore2[simMatrixStoreSize2]);
          simMatrixStoreSize2=simMatrixStoreSize2+targetSize;
        }
        simMatrixStoreSize1=simMatrixStoreSize1+targetSize;
      }
      
      simMatrixStoreSize2=0L;
      simMatrixStoreSize1=0L;
      for(long i=0L; i<denseFeatureKernelVec.size(); i++) {
        dSimMatrix[i]=&(dSimMatrixStore1[simMatrixStoreSize1]);
        for(long j=0L; j<targetSize; j++) {
          dSimMatrix[i][j]=&(dSimMatrixStore2[simMatrixStoreSize2]);
          simMatrixStoreSize2=simMatrixStoreSize2+targetSize;
        }
        simMatrixStoreSize1=simMatrixStoreSize1+targetSize;
      }
      
      for(long i=0L; i<sparseFeatureKernelVec.size(); i++) {
        long* samples=((sparseFeatureKernelVec[i])->feature)->samples;
        long* features=((sparseFeatureKernelVec[i])->feature)->features;
        double* featureCounts=((sparseFeatureKernelVec[i])->feature)->featureCounts;
        KernelType ktype=(sparseFeatureKernelVec[i])->kt;
        
        for(long k=0L; k<targetSizeVec[targetInd]; k++)
        {
          long sampleInd1=targetSampleDB[targetIndStart+k].sampleNr;
          long beginFeat1=samples[sampleInd1];
          long endFeat1=samples[sampleInd1+1L];
          
          #ifdef multiproc
          #pragma omp parallel for
          #endif
          for(long l=0L; l<targetSizeVec[targetInd]; l++) {
            long sampleInd2=targetSampleDB[targetIndStart+l].sampleNr;
            long beginFeat2=samples[sampleInd2];
            long endFeat2=samples[sampleInd2+1L];
            
            if(k<l) {
              if(ktype==LIN) {
                sSimMatrix[i][k][l]=linearSumSparseCount(features, featureCounts, beginFeat1, endFeat1, features, featureCounts, beginFeat2, endFeat2);
                sSimMatrix[i][l][k]=sSimMatrix[i][k][l];
              }
              else if(ktype==GAUSS) {
                sSimMatrix[i][k][l]=gaussianSumSparseCount(features, featureCounts, beginFeat1, endFeat1, features, featureCounts, beginFeat2, endFeat2);
                sSimMatrix[i][l][k]=sSimMatrix[i][k][l];
              }
              else if(ktype==TAN) {
                sumRet_t res=tanimotoSumSparseCount(features, featureCounts, beginFeat1, endFeat1, features, featureCounts, beginFeat2, endFeat2);
                sSimMatrix[i][k][l]=res.sum1;
                sSimMatrix[i][l][k]=res.sum2;
              }
              else if(ktype==TAN2) {
                sumRet_t res=tanimotoSumSparseCount2(features, featureCounts, beginFeat1, endFeat1, features, featureCounts, beginFeat2, endFeat2);
                sSimMatrix[i][k][l]=res.sum1;
                sSimMatrix[i][l][k]=res.sum2;
              }
            }
            else if(k==l) {
              if(ktype==LIN) {
                sSimMatrix[i][k][l]=linearSumSparseCount(features, featureCounts, beginFeat1, endFeat1, features, featureCounts, beginFeat2, endFeat2);
              }
              else if(ktype==GAUSS) {
                sSimMatrix[i][k][l]=gaussianSumSparseCount(features, featureCounts, beginFeat1, endFeat1, features, featureCounts, beginFeat2, endFeat2);
              }
              else if(ktype==TAN) {
                sSimMatrix[i][k][l]=1.0;
              }
              else if(ktype==TAN2) {
                sSimMatrix[i][k][l]=1.0;
              }
            }
          }
        }
      }
      
      for(long i=0L; i<denseFeatureKernelVec.size(); i++) {
        double** properties=((denseFeatureKernelVec[i])->feature)->properties;
        long propertyNr=((denseFeatureKernelVec[i])->feature)->propertyNr;
        long sampleNr=((denseFeatureKernelVec[i])->feature)->sampleNr;
        KernelType ktype=(denseFeatureKernelVec[i])->kt;
        
        for(long k=0L; k<targetSizeVec[targetInd]; k++)
        {
          long sampleInd1=targetSampleDB[targetIndStart+k].sampleNr;
          
          #ifdef multiproc
          #pragma omp parallel for
          #endif
          for(long l=0L; l<targetSizeVec[targetInd]; l++) {
            long sampleInd2=targetSampleDB[targetIndStart+l].sampleNr;
            if(k<l) {
              if(ktype==LIN) {
                dSimMatrix[i][k][l]=linearSumDense(properties[sampleInd1], properties[sampleInd2], propertyNr);
                dSimMatrix[i][l][k]=dSimMatrix[i][k][l];
              }
              else if(ktype==GAUSS) {
                dSimMatrix[i][k][l]=gaussianSumDense(properties[sampleInd1], properties[sampleInd2], propertyNr);
                dSimMatrix[i][l][k]=dSimMatrix[i][k][l];
              }
              else if(ktype==TAN) {
                sumRet_t res=tanimotoSumDenseOrig(properties[sampleInd1], properties[sampleInd2], propertyNr);
                dSimMatrix[i][k][l]=res.sum1;
                dSimMatrix[i][l][k]=res.sum2;
              }
              else if(ktype==TANS) {
                sumRet_t res=tanimotoSumDenseOrigSplit(properties[sampleInd1], properties[sampleInd2], propertyNr);
                dSimMatrix[i][k][l]=res.sum1;
                dSimMatrix[i][l][k]=res.sum2;
              }
              else if(ktype==TAN2) {
                sumRet_t res=tanimotoSumDenseOrig2(properties[sampleInd1], properties[sampleInd2], propertyNr);
                dSimMatrix[i][k][l]=res.sum1;
                dSimMatrix[i][l][k]=res.sum2;
              }
            }
            else if(k==l) {
              if(ktype==LIN) {
                dSimMatrix[i][k][l]=linearSumDense(properties[sampleInd1], properties[sampleInd2], propertyNr);
              }
              else if(ktype==GAUSS) {
                dSimMatrix[i][k][l]=gaussianSumDense(properties[sampleInd1], properties[sampleInd2], propertyNr);
              }
              else if(ktype==TAN) {
                dSimMatrix[i][k][l]=1.0;
              }
              else if(ktype==TANS) {
                dSimMatrix[i][k][l]=1.0;
              }
              else if(ktype==TAN2) {
                dSimMatrix[i][k][l]=1.0;
              }
            }
          }
        }
      }
      
      
      
      for(long fpi=0L; fpi<featureParamInd.size()-1L; fpi++) {
        std::stringstream ss;
        ss << paramComb[featureParamInd[fpi]].fkc.id;
        string currentPredSavePathname=predSavePathname+targetNameVec[targetInd]+"/"+ss.str()+"/";
        mkdir(currentPredSavePathname.c_str(), S_IRWXU);
        
        
        
        double* aucEvalTrain=(double*)malloc(sizeof(double)*nrClusters);
        double* aucEval=(double*)malloc(sizeof(double)*nrClusters);
        
        for(long clusterIndEval=0L; clusterIndEval<nrClusters; clusterIndEval++)
        {
          {
            printf("cE(%ld)/cE(%ld)\n", clusterIndEval, nrClusters);
            long clusterIndStartEval=clusterIndVec[clusterIndEval];
            long clusterSizeEval=clusterSizeVec[clusterIndEval];
            long clusterIndEndEval=clusterIndStartEval+clusterSizeEval;
            
            bool* validEvalSamples=new bool[sampleNr];
            for(long i=0; i<sampleNr; i++)
              validEvalSamples[i]=true;
            
            {
              long removeInd=clusterIndStartEval;
              while(removeInd<clusterIndEndEval) {
                validEvalSamples[clusterSampleDB[removeInd].sampleNr]=false;
                removeInd++;
              }
            }
            
            bool* validEval=new bool[targetSampleDBSize];
            for(long i=0; i<targetSampleDBSize; i++)
              validEval[i]=true;
              
            {
              long targetSampleDBInd=0L;
              long removeInd=clusterIndStartEval;

              while((targetSampleDBInd<targetSampleDBSize)&&(removeInd<clusterIndEndEval)) {
                if(targetSampleDB[indTargetSampleDB[targetSampleDBInd]].sampleNr<clusterSampleDB[removeInd].sampleNr) {
                  targetSampleDBInd++;
                }
                else if(targetSampleDB[indTargetSampleDB[targetSampleDBInd]].sampleNr>clusterSampleDB[removeInd].sampleNr) {
                  removeInd++;
                }
                else {
                  validEval[indTargetSampleDB[targetSampleDBInd]]=false;
                  targetSampleDBInd++;
                }
              }
            }
            
            
            
            long nrTrain=0L;
            long nrPositiveTrain=0L;
            long nrNegativeTrain=0L;
            long* indTrain=(long*)malloc(sizeof(long)*targetSize);
            long* sampleNrTrain=(long*)malloc(sizeof(long)*targetSize);
            long* labelTrain=(long*)malloc(sizeof(long)*targetSize);
            
            long nrPred=0L;
            long nrPositivePred=0L;
            long nrNegativePred=0L;                        
            long* indPred=(long*)malloc(sizeof(long)*targetSize);
            long* sampleNrPred=(long*)malloc(sizeof(long)*targetSize);
            long* labelPred=(long*)malloc(sizeof(long)*targetSize);
            
            if(indTrain==NULL||labelTrain==NULL||indPred==NULL||labelPred==NULL) {
              fprintf(stderr, "Too few main memory. Program will terminate! - error %d\n", __LINE__);
              exit(-1);
            }
            
            long targetSampleDBInd=targetIndStart;
            while(targetSampleDBInd<targetIndEnd) {
              if(validEval[targetSampleDBInd]) {
                if(targetSampleDB[targetSampleDBInd].label==1L) {
                  indTrain[nrTrain]=targetSampleDBInd-targetIndStart;
                  sampleNrTrain[nrTrain]=targetSampleDB[targetSampleDBInd].sampleNr;
                  labelTrain[nrTrain]=0L;
                  nrNegativeTrain++;
                  nrTrain++;
                }
                else if(targetSampleDB[targetSampleDBInd].label==3L) {
                  indTrain[nrTrain]=targetSampleDBInd-targetIndStart;
                  sampleNrTrain[nrTrain]=targetSampleDB[targetSampleDBInd].sampleNr;
                  labelTrain[nrTrain]=1L;
                  nrPositiveTrain++;
                  nrTrain++;
                }
              }
              targetSampleDBInd++;
            }
            
            targetSampleDBInd=targetIndStart;
            while(targetSampleDBInd<targetIndEnd) {
              if(!validEval[targetSampleDBInd]) {
                if(targetSampleDB[targetSampleDBInd].label==1L) {
                  indPred[nrPred]=targetSampleDBInd-targetIndStart;
                  sampleNrPred[nrPred]=targetSampleDB[targetSampleDBInd].sampleNr;
                  labelPred[nrPred]=0L;
                  nrNegativePred++;
                  nrPred++;
                }
                else if(targetSampleDB[targetSampleDBInd].label==3L) {
                  indPred[nrPred]=targetSampleDBInd-targetIndStart;
                  sampleNrPred[nrPred]=targetSampleDB[targetSampleDBInd].sampleNr;
                  labelPred[nrPred]=1L;
                  nrPositivePred++;
                  nrPred++;
                }
              }
              targetSampleDBInd++;
            }
            
            
            
            std::stringstream ss;
            ss << clusterNrVec[clusterIndEval];
            string outFilenameSampleTrain=currentPredSavePathname+"clustSampleTrain"+ss.str()+".out";
            string outFilenameProbTrain=currentPredSavePathname+"clustProbTrain"+ss.str()+".out";
            string outFilenameClassTrain=currentPredSavePathname+"clustClassTrain"+ss.str()+".out";
            string outFilenameTrueTrain=currentPredSavePathname+"clustTrueTrain"+ss.str()+".out";
            
            string outFilenameSample=currentPredSavePathname+"clustSample"+ss.str()+".out";
            string outFilenameProb=currentPredSavePathname+"clustProb"+ss.str()+".out";
            string outFilenameClass=currentPredSavePathname+"clustClass"+ss.str()+".out";
            string outFilenameTrue=currentPredSavePathname+"clustTrue"+ss.str()+".out";
            
            char* outFileSampleTrainBuffer;
            FILE* outFileSampleTrain;
            char* outFileProbTrainBuffer;
            FILE* outFileProbTrain;
            char* outFileClassTrainBuffer;
            FILE* outFileClassTrain;
            char* outFileTrueTrainBuffer;
            FILE* outFileTrueTrain;
            
            char* outFileSampleBuffer;
            FILE* outFileSample;
            char* outFileProbBuffer;
            FILE* outFileProb;
            char* outFileClassBuffer;
            FILE* outFileClass;
            char* outFileTrueBuffer;
            FILE* outFileTrue;
            
            outFileSampleTrainBuffer=(char*)malloc(1024*1024);
            outFileSampleTrain=fopen(outFilenameSampleTrain.c_str(), "w");
            setbuffer(outFileSampleTrain, outFileSampleTrainBuffer, 1024*1024);
            outFileProbTrainBuffer=(char*)malloc(1024*1024);
            outFileProbTrain=fopen(outFilenameProbTrain.c_str(), "w");
            setbuffer(outFileProbTrain, outFileProbTrainBuffer, 1024*1024);
            outFileClassTrainBuffer=(char*)malloc(1024*1024);
            outFileClassTrain=fopen(outFilenameClassTrain.c_str(), "w");
            setbuffer(outFileClassTrain, outFileClassTrainBuffer, 1024*1024);
            outFileTrueTrainBuffer=(char*)malloc(1024*1024);
            outFileTrueTrain=fopen(outFilenameTrueTrain.c_str(), "w");
            setbuffer(outFileTrueTrain, outFileTrueTrainBuffer, 1024*1024);
            
            outFileSampleBuffer=(char*)malloc(1024*1024);
            outFileSample=fopen(outFilenameSample.c_str(), "w");
            setbuffer(outFileSample, outFileSampleBuffer, 1024*1024);
            outFileProbBuffer=(char*)malloc(1024*1024);
            outFileProb=fopen(outFilenameProb.c_str(), "w");
            setbuffer(outFileProb, outFileProbBuffer, 1024*1024);
            outFileClassBuffer=(char*)malloc(1024*1024);
            outFileClass=fopen(outFilenameClass.c_str(), "w");
            setbuffer(outFileClass, outFileClassBuffer, 1024*1024);
            outFileTrueBuffer=(char*)malloc(1024*1024);
            outFileTrue=fopen(outFilenameTrue.c_str(), "w");
            setbuffer(outFileTrue, outFileTrueBuffer, 1024*1024);
            
            
            
            if((nrPositiveTrain<1)||(nrNegativeTrain<1)||(nrPositivePred<1)||(nrNegativePred<1)) {
              for(long i=0L; i<nrTrain; i++) {
                fprintf(outFileSampleTrain, "%ld\n", sampleNrTrain[i]);
                fprintf(outFileTrueTrain, "%ld\n", labelTrain[i]);
                fprintf(outFileProbTrain, "%.10lf\n", 0.0);
                fprintf(outFileClassTrain, "%.10lf\n", 0.0);     
              }
              
              for(long i=0L; i<nrPred; i++) {
                fprintf(outFileSample, "%ld\n", sampleNrPred[i]);
                fprintf(outFileTrue, "%ld\n", labelPred[i]);
                fprintf(outFileProb, "%.10lf\n", 0.0);
                fprintf(outFileClass, "%.10lf\n", 0.0);            
              }
              
              free(labelPred);
              free(sampleNrPred);
              free(indPred);
              free(labelTrain);
              free(sampleNrTrain);
              free(indTrain);
              
              delete[] validEval;
              delete[] validEvalSamples;
              
              
              
              fclose(outFileTrue);
              free(outFileTrueBuffer);
              fclose(outFileClass);
              free(outFileClassBuffer);
              fclose(outFileProb);
              free(outFileProbBuffer);
              fclose(outFileSample);
              free(outFileSampleBuffer);
              
              fclose(outFileTrueTrain);
              free(outFileTrueTrainBuffer);
              fclose(outFileClassTrain);
              free(outFileClassTrainBuffer);
              fclose(outFileProbTrain);
              free(outFileProbTrainBuffer);
              fclose(outFileSampleTrain);
              free(outFileSampleTrainBuffer);
              
              
              
              aucEvalTrain[clusterIndEval]=-1.0;
              aucEval[clusterIndEval]=-1.0;
              
              goto endLoopClusterEval;
            }
            
            
            
            delete[] validEval;
            delete[] validEvalSamples;
            
            
            
            long aucSelIndex=featureParamInd[fpi];
            
            
            
            
            float* simMatrixPredStore=(float*) malloc(sizeof(float)*nrTrain*nrPred);
            float** simMatrixPred=(float**) malloc(sizeof(float*)*nrTrain);
            float* simMatrixPredStoreH=(float*) malloc(sizeof(float)*nrTrain*nrPred);
            float** simMatrixPredH=(float**) malloc(sizeof(float*)*nrTrain);
            float* normMatrixTrain=(float*) malloc(sizeof(float)*nrTrain);
            float* normMatrixPred=(float*) malloc(sizeof(float)*nrPred);
            
            for(long i=0L; i<nrTrain; i++) {
              simMatrixPred[i]=&(simMatrixPredStore[i*nrPred]);
              simMatrixPredH[i]=&(simMatrixPredStoreH[i*nrPred]);
            }
            
            
            
            FeatureKernelCollection featureParam;
            long featureParamId;
            KernelComb kc;
            long kn;
            
            if(aucSelIndex==-1) {
              featureParam=defaultFeature;
              featureParamId=featureParam.id;
              kc=defaultFeature.kc;
              kn=defaultFeature.kn;
            }
            else {
              featureParam=paramComb[aucSelIndex].fkc;
              featureParamId=featureParam.id;
              kc=paramComb[aucSelIndex].fkc.kc;
              kn=paramComb[aucSelIndex].fkc.kn;
            }
            
            
            
            double bestThreshold;
            double bestmnlMean;
            double bestnnlMean;
            double bestpnlMean;
            double bestmnl;
            double bestnnl;
            double bestpnl;
            
            std::stringstream ssPar;
            ssPar << clusterIndEval << "_" << aucSelIndex;
            
            string parFilename=predSavePathname+"par_"+ssPar.str()+".bin";
            FILE* parFile=fopen(parFilename.c_str(), "rb");
            fread(&bestThreshold, sizeof(double), 1L, parFile);
            fread(&bestmnlMean, sizeof(double), 1L, parFile);
            fread(&bestnnlMean, sizeof(double), 1L, parFile);
            fread(&bestpnlMean, sizeof(double), 1L, parFile);
            fread(&bestmnl, sizeof(double), 1L, parFile);
            fread(&bestnnl, sizeof(double), 1L, parFile);
            fread(&bestpnl, sizeof(double), 1L, parFile);
            fclose(parFile);
            
            
            
            fprintf(stderr, "bestThreshold: %.10lf\n", bestThreshold);
            fprintf(stderr, "bestmnlMean: %.10lf\n", bestmnlMean);
            fprintf(stderr, "bestnnlMean: %.10lf\n", bestnnlMean);
            fprintf(stderr, "bestpnlMean: %.10lf\n", bestpnlMean);
            fprintf(stderr, "bestmnl: %.10lf\n", bestmnl);
            fprintf(stderr, "bestnnl: %.10lf\n", bestnnl);
            fprintf(stderr, "bestpnl: %.10lf\n", bestpnl);
            
            
            
            std::fill(simMatrixPredStore, simMatrixPredStore+nrTrain*nrPred, 0.0f);
            std::fill(simMatrixPredStoreH, simMatrixPredStoreH+nrTrain*nrPred, 0.0f);
            std::fill(normMatrixTrain, normMatrixTrain+nrTrain, 0.0f);
            std::fill(normMatrixPred, normMatrixPred+nrPred, 0.0f);
            
            for(long i=0L; i<featureParam.sFeatureKernels.size()+featureParam.dFeatureKernels.size(); i++) {
              long matInd;
              float** sim;
              if(i<featureParam.sFeatureKernels.size()) {
                matInd=featureParam.sFeatureKernels[i]->id;
                sim=sSimMatrix[matInd];
              }
              else {
                matInd=featureParam.dFeatureKernels[i-featureParam.sFeatureKernels.size()]->id;
                sim=dSimMatrix[matInd];
              }
              
              #ifdef multiproc
              #pragma omp parallel for
              #endif
              for(long j=0L; j<nrTrain; j++)
              {
                for(long k=0L; k<nrPred; k++) {
                  if(indTrain[j]<indPred[k]) {
                    simMatrixPred[j][k]=simMatrixPred[j][k]+sim[indTrain[j]][indPred[k]];
                    simMatrixPredH[j][k]=simMatrixPredH[j][k]+sim[indPred[k]][indTrain[j]];
                  }
                  else {
                    simMatrixPred[j][k]=simMatrixPred[j][k]+sim[indPred[k]][indTrain[j]];
                    simMatrixPredH[j][k]=simMatrixPredH[j][k]+sim[indTrain[j]][indPred[k]];
                  }
                }
              }
              
              #ifdef multiproc
              #pragma omp parallel for
              #endif
              for(long j=0L; j<nrTrain; j++) {
                normMatrixTrain[j]=normMatrixTrain[j]+sim[indTrain[j]][indTrain[j]];
              }
              
              #ifdef multiproc
              #pragma omp parallel for
              #endif
              for(long j=0L; j<nrPred; j++) {
                normMatrixPred[j]=normMatrixPred[j]+sim[indPred[j]][indPred[j]];
              }
            }
            
            #ifdef multiproc
            #pragma omp parallel for
            #endif
            for(long j=0L; j<nrTrain; j++) {
              normMatrixTrain[j]=sqrt(normMatrixTrain[j]);
            } 
            
            #ifdef multiproc
            #pragma omp parallel for
            #endif
            for(long j=0L; j<nrPred; j++) {
              normMatrixPred[j]=sqrt(normMatrixPred[j]);
            }          
            
            
            
            if(kc==TANCOMB||kc==TANCOMB2||kc==TANSCOMB) {
              #ifdef multiproc
              #pragma omp parallel for
              #endif     
              for(long i=0L; i<nrTrain; i++)
              {
                for(long j=0L; j<nrPred; j++) {
                  if(simMatrixPredH[i][j]>=0.1)
                    simMatrixPred[i][j]=simMatrixPred[i][j]/simMatrixPredH[i][j];
                  else
                    simMatrixPred[i][j]=0.0;
                }
              }
            }
            
            if(kn==1) {
              #ifdef multiproc
              #pragma omp parallel for
              #endif                
              for(long i=0L; i<nrTrain; i++)
              {
                for(long j=0L; j<nrPred; j++) {
                  simMatrixPred[i][j]=simMatrixPred[i][j]/(normMatrixTrain[i]*normMatrixPred[j]);
                }
              }
            }
            
            
            
#ifdef dbgmat
            std::stringstream ssTrainMat;
            ssTrainMat << currentPredSavePathname << "dbgMatTrain_" << clusterIndEval << "_" << featureParam.id;
            string ssTrainMatStr=ssTrainMat.str();

            std::stringstream ssPredMat;
            ssPredMat << currentPredSavePathname << "dbgMatPred_" << clusterIndEval << "_" << featureParam.id;
            string ssPredMatStr=ssPredMat.str();
            
            std::stringstream ssTrainInd;
            ssTrainInd << currentPredSavePathname << "dbgIndTrain_" << clusterIndEval << "_" << featureParam.id;
            string ssTrainIndStr=ssTrainInd.str();

            std::stringstream ssPredInd;
            ssPredInd << currentPredSavePathname << "dbgIndPred_" << clusterIndEval << "_" << featureParam.id;
            string ssPredIndStr=ssPredInd.str();
            
            std::stringstream ssTrainLabel;
            ssTrainLabel << currentPredSavePathname << "dbgLabelTrain_" << clusterIndEval << "_" << featureParam.id;
            string ssTrainLabelStr=ssTrainLabel.str();

            std::stringstream ssPredLabel;
            ssPredLabel << currentPredSavePathname << "dbgLabelPred_" << clusterIndEval << "_" << featureParam.id;
            string ssPredLabelStr=ssPredLabel.str();
            
            int statRet=stat(ssTrainMatStr.c_str(), &sb);
            if((statRet==-1)) {
              FILE* matFileTrain=fopen(ssTrainMatStr.c_str(), "w");
              FILE* matFilePred=fopen(ssPredMatStr.c_str(), "w");
              FILE* indFileTrain=fopen(ssTrainIndStr.c_str(), "w");
              FILE* indFilePred=fopen(ssPredIndStr.c_str(), "w");
              FILE* labelFileTrain=fopen(ssTrainLabelStr.c_str(), "w");
              FILE* labelFilePred=fopen(ssPredLabelStr.c_str(), "w");
              
              for(long i=0L;  i<nrTrain; i++) {
                fprintf(indFileTrain, "%ld\n", sampleNrTrain[i]);
                fprintf(labelFileTrain, "%ld\n", labelTrain[i]);
              }
              
              for(long i=0L;  i<nrPred; i++) {
                fprintf(indFilePred, "%ld\n", sampleNrPred[i]);
                fprintf(labelFilePred, "%ld\n", labelPred[i]);
              }
              
              for(long i=0L; i<nrTrain; i++) {
                for(long j=0L; j<nrPred; j++) {
                  fprintf(matFilePred, "%lf", simMatrixPred[i][j]);
                  if(j==nrPred-1)
                    fprintf(matFilePred, "\n");
                  else
                    fprintf(matFilePred, " ");
                }
              }
              
              fclose(labelFilePred);
              fclose(labelFileTrain);
              fclose(indFilePred);
              fclose(indFileTrain);
              fclose(matFilePred);
              fclose(matFileTrain);
            }
#endif
            
            
            
            double* predVec=(double*)malloc(sizeof(double)*nrPred);                  
            for(long i=0L; i<nrPred; i++) {
              fprintf(outFileSample, "%ld\n", sampleNrPred[i]);
              fprintf(outFileTrue, "%ld\n", labelPred[i]);
              
              double Score=0.0;
              for(long j=0L; j<nrTrain; j++) {
                if(labelTrain[j]==1L) {
                  if(simMatrixPred[j][i]>bestThreshold)
                    Score=Score+simMatrixPred[j][i];
                }
              }
              
              double Eraw=bestpnlMean+bestmnlMean*pow(nrPositiveTrain, bestnnlMean); 
              double Estd=bestpnl+bestmnl*pow(nrPositiveTrain, bestnnl);
              double z=(Score-Eraw)/(Estd+0.00000000001);
              
              predVec[i]=z;
              fprintf(outFileProb, "%.10lf\n", z);
              double cls=(z>0.5);
              fprintf(outFileClass, "%.10lf\n", cls);
            }
            aucEval[clusterIndEval]=compAUC(labelPred, predVec, nrPred);
            free(predVec);
            
            
            
            fclose(outFileTrue);
            free(outFileTrueBuffer);
            fclose(outFileClass);
            free(outFileClassBuffer);
            fclose(outFileProb);
            free(outFileProbBuffer);
            fclose(outFileSample);
            free(outFileSampleBuffer);
            
            fclose(outFileTrueTrain);
            free(outFileTrueTrainBuffer);
            fclose(outFileClassTrain);
            free(outFileClassTrainBuffer);
            fclose(outFileProbTrain);
            free(outFileProbTrainBuffer);
            fclose(outFileSampleTrain);
            free(outFileSampleTrainBuffer);
            
            
            
            free(normMatrixPred);
            free(normMatrixTrain);
            free(simMatrixPredH);
            free(simMatrixPredStoreH);
            free(simMatrixPred);
            free(simMatrixPredStore);
            
            free(labelPred);
            free(sampleNrPred);
            free(indPred);
            free(labelTrain);
            free(sampleNrTrain);
            free(indTrain);
          }
          endLoopClusterEval: ;
        }
        
        string outFilenameEvalTrainAUC=currentPredSavePathname+"clustEvalTrainAUC.out";
        string outFilenameEvalPredAUC=currentPredSavePathname+"clustEvalPredAUC.out";
        
        FILE* outFileEvalTrainAUC=fopen(outFilenameEvalTrainAUC.c_str(), "w");
        for(long i=0L; i<(nrClusters-1L); i++) {
          //fprintf(outFileEvalTrainAUC, "%5.10lf,", aucEvalTrain[i]);
          fprintf(outFileEvalTrainAUC, ",", aucEvalTrain[i]);
        }
        //fprintf(outFileEvalTrainAUC, "%5.10lf\n", aucEvalTrain[nrClusters-1L]);
        fclose(outFileEvalTrainAUC);
        
        FILE* outFileEvalPredAUC=fopen(outFilenameEvalPredAUC.c_str(), "w");
        for(long i=0L; i<(nrClusters-1L); i++) {
          fprintf(outFileEvalPredAUC, "%5.10lf,", aucEval[i]);
        }
        fprintf(outFileEvalPredAUC, "%5.10lf\n", aucEval[nrClusters-1L]);
        fclose(outFileEvalPredAUC);
        
        free(aucEval);
        free(aucEvalTrain);      
      
      
      
      }

      
      
      
      free(dSimMatrix);
      free(dSimMatrixStore1);
      free(dSimMatrixStore2);
      free(sSimMatrix);
      free(sSimMatrixStore1);
      free(sSimMatrixStore2);
    }
    endLoopTarget: ;
  }
  
  
  
  delete[] clusterSampleDB;
  free(indTargetSampleDB);
  delete[] targetSampleDB;
  
  
  
  std::map<string, DenseFeatureKernel*>::iterator it4;
  for (it4=denseFeatureKernelMap.begin(); it4!=denseFeatureKernelMap.end(); ++it4)
    delete it4->second;
  
  std::map<string, SparseFeatureKernel*>::iterator it3;
  for (it3=sparseFeatureKernelMap.begin(); it3!=sparseFeatureKernelMap.end(); ++it3)
    delete it3->second;
  
  std::map<string, DenseFeatureData*>::iterator it2;
  for (it2=denseFeatureMap.begin(); it2!=denseFeatureMap.end(); ++it2)
    delete it2->second;
  
  std::map<string, SparseFeatureData*>::iterator it1;
  for (it1=sparseFeatureMap.begin(); it1!=sparseFeatureMap.end(); ++it1)
    delete it1->second;
  
  
  
  printf("seaHyper3 terminated successfully!\n");

  return 0;
}