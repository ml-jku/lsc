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

typedef struct ScoreT {
  long size;
  double* score;
} ScoreT;

long* getIntegerFactors(long number, long min, long max, long& nrFactors) {
  list<long> lFactors;
  if(number<=max)
    max=number;
  for(long i=min; i<=max; i++) {
    if(number%i==0L)
      lFactors.push_back(i);
  }
  if(lFactors.size()==0L) {
    nrFactors=0L;
    return NULL;
  }
  long* arr=(long*)malloc(sizeof(long)*lFactors.size());
  long i=0L;
  for (std::list<long>::iterator it=lFactors.begin(); it != lFactors.end(); ++it) {
    arr[i]=*it;
    i++;
  }
  nrFactors=lFactors.size();
  return arr;
}

void randomSample(gsl_rng* r, long** selCompounds, long *nrSelCompounds, long N, long bound) {
  *nrSelCompounds=N;
  *selCompounds=(long*)malloc(sizeof(long)*N);
  long i=0;
  while(i<N) {
    long newNr=gsl_rng_uniform_int(r, bound);
    (*selCompounds)[i]=newNr;
    for(long j=0; j<i; j++) {
      if((*selCompounds)[j]==newNr) {
        i--;
        break;
      }
    }
    i++;
  }
}



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
  if(mkdir(predSavePathname.c_str(), S_IRWXU)==-1) {
    fprintf(stderr, "The directory '%s' either already exists or it is an invalid directory!\n", predSavePathname.c_str());
    exit(-1);
  }
  
  
  
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
  
  vector<long> featureParamInd;
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
  
  
  
  long smin=10;
  long smax=300;
  
  //smin=3;
  //smax=10;
  
  double stepsize=0.01;
  double minThreshold=0.00;
  double maxThreshold=1.00;
  long nrTests=(maxThreshold-minThreshold+0.00000000001)/stepsize;
  
  long nrIntegers=1000;
  long nrIntegerFactors=30;
  //nrIntegers=10;
  //nrIntegerFactors=3;
  long nrSetComparisons=nrIntegers*nrIntegerFactors;
  
  const gsl_rng_type * T=gsl_rng_mt19937;
  gsl_rng* rInit=gsl_rng_alloc(T);
  gsl_rng_set(rInit, 1234567);
  
  gsl_rng** r=new gsl_rng*[nrIntegers];
  for(long i=0; i<nrIntegers; i++) {
    r[i]=gsl_rng_alloc(T);
    gsl_rng_set(r[i], gsl_rng_uniform_int(rInit, 10000)*(i+1L));
  }
  gsl_rng_free(rInit);
  
  
  
  

  
  
  
  for(long clusterIndEval=0L; clusterIndEval<nrClusters; clusterIndEval++) {
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
    
    std::vector<long> sampleClusterEval;
    for(long i=0; i<sampleNr; i++) {
      if(validEvalSamples[i]) {
        sampleClusterEval.push_back(i);
      }
    }
    
    long maxSize=ceil((double)(smax*smax)/((double)smin)+1.0);
    
    fprintf(stderr, "allocating");
    
    //long nrReserve=maxProc;
    long nrReserve=1L;
    
    float* sSimMatrixStore1=(float*) malloc(sizeof(float)*maxSize*maxSize*sparseFeatureKernelVec.size()*nrReserve);
    float** sSimMatrixStore2=(float**) malloc(sizeof(float*)*maxSize*sparseFeatureKernelVec.size()*nrReserve);
    float*** sSimMatrixStore3=(float***) malloc(sizeof(float**)*sparseFeatureKernelVec.size()*nrReserve);
    float**** sSimMatrix=(float****) malloc(sizeof(float***)*nrReserve);
    float* sSimMatrixHStore1=(float*) malloc(sizeof(float)*maxSize*maxSize*sparseFeatureKernelVec.size()*nrReserve);
    float** sSimMatrixHStore2=(float**) malloc(sizeof(float*)*maxSize*sparseFeatureKernelVec.size()*nrReserve);
    float*** sSimMatrixHStore3=(float***) malloc(sizeof(float**)*sparseFeatureKernelVec.size()*nrReserve);
    float**** sSimMatrixH=(float****) malloc(sizeof(float***)*nrReserve);    
    
    float* dSimMatrixStore1=(float*) malloc(sizeof(float)*maxSize*maxSize*denseFeatureKernelVec.size()*nrReserve);
    float** dSimMatrixStore2=(float**) malloc(sizeof(float*)*maxSize*denseFeatureKernelVec.size()*nrReserve);
    float*** dSimMatrixStore3=(float***) malloc(sizeof(float**)*denseFeatureKernelVec.size()*nrReserve);
    float**** dSimMatrix=(float****) malloc(sizeof(float***)*nrReserve);
    float* dSimMatrixHStore1=(float*) malloc(sizeof(float)*maxSize*maxSize*denseFeatureKernelVec.size()*nrReserve);
    float** dSimMatrixHStore2=(float**) malloc(sizeof(float*)*maxSize*denseFeatureKernelVec.size()*nrReserve);
    float*** dSimMatrixHStore3=(float***) malloc(sizeof(float**)*denseFeatureKernelVec.size()*nrReserve);
    float**** dSimMatrixH=(float****) malloc(sizeof(float***)*nrReserve);
    
    float* sumSimMatrixStore1=(float*) malloc(sizeof(float)*maxSize*maxSize*nrReserve);
    float** sumSimMatrixStore2=(float**) malloc(sizeof(float*)*maxSize*nrReserve);
    float*** sumSimMatrix=(float***) malloc(sizeof(float**)*nrReserve);
    float* sumSimMatrixHStore1=(float*) malloc(sizeof(float)*maxSize*maxSize*nrReserve);
    float** sumSimMatrixHStore2=(float**) malloc(sizeof(float*)*maxSize*nrReserve);
    float*** sumSimMatrixH=(float***) malloc(sizeof(float**)*nrReserve);

    fprintf(stderr, "allocated");
    
    long simMatrixStoreSize1;
    long simMatrixStoreSize2;
    long simMatrixStoreSize3;
    
    long ia;
    long ja;
    long ka;    
    
    simMatrixStoreSize1=0L;
    simMatrixStoreSize2=0L;
    simMatrixStoreSize3=0L;
    ia=0L;
    ja=0L;
    ka=0L;
    
    for(long k=0L; k<nrReserve; k++) {
      for(long j=0L; j<sparseFeatureKernelVec.size(); j++) {
        for(long i=0L; i<maxSize; i++) {
          sSimMatrixStore2[ia]=&(sSimMatrixStore1[simMatrixStoreSize1]);
          sSimMatrixHStore2[ia]=&(sSimMatrixHStore1[simMatrixStoreSize1]);
          simMatrixStoreSize1=simMatrixStoreSize1+maxSize;
          ia++;
        }
        sSimMatrixStore3[ja]=&(sSimMatrixStore2[simMatrixStoreSize2]);
        sSimMatrixHStore3[ja]=&(sSimMatrixHStore2[simMatrixStoreSize2]);
        simMatrixStoreSize2=simMatrixStoreSize2+maxSize;
        ja++;
      }
      sSimMatrix[ka]=&(sSimMatrixStore3[simMatrixStoreSize3]);
      sSimMatrixH[ka]=&(sSimMatrixHStore3[simMatrixStoreSize3]);
      simMatrixStoreSize3=simMatrixStoreSize3+sparseFeatureKernelVec.size();
      ka++;
    }
    
    simMatrixStoreSize1=0L;
    simMatrixStoreSize2=0L;
    simMatrixStoreSize3=0L;
    ia=0L;
    ja=0L;
    ka=0L;
    
    for(long k=0L; k<nrReserve; k++) {
      for(long j=0L; j<denseFeatureKernelVec.size(); j++) {
        for(long i=0L; i<maxSize; i++) {
          dSimMatrixStore2[ia]=&(dSimMatrixStore1[simMatrixStoreSize1]);
          dSimMatrixHStore2[ia]=&(dSimMatrixHStore1[simMatrixStoreSize1]);
          simMatrixStoreSize1=simMatrixStoreSize1+maxSize;
          ia++;
        }
        dSimMatrixStore3[ja]=&(dSimMatrixStore2[simMatrixStoreSize2]);
        dSimMatrixHStore3[ja]=&(dSimMatrixHStore2[simMatrixStoreSize2]);
        simMatrixStoreSize2=simMatrixStoreSize2+maxSize;
        ja++;
      }
      dSimMatrix[ka]=&(dSimMatrixStore3[simMatrixStoreSize3]);
      dSimMatrixH[ka]=&(dSimMatrixHStore3[simMatrixStoreSize3]);
      simMatrixStoreSize3=simMatrixStoreSize3+denseFeatureKernelVec.size();
      ka++;
    }
    
    simMatrixStoreSize1=0L;
    simMatrixStoreSize2=0L;
    ia=0L;
    ja=0L;
    
    for(long j=0L; j<nrReserve; j++) {
      for(long i=0L; i<maxSize; i++) {
        sumSimMatrixStore2[ia]=&(sumSimMatrixStore1[simMatrixStoreSize1]);
        sumSimMatrixHStore2[ia]=&(sumSimMatrixHStore1[simMatrixStoreSize1]);
        simMatrixStoreSize1=simMatrixStoreSize1+maxSize;
        ia++;
      }
      sumSimMatrix[ja]=&(sumSimMatrixStore2[simMatrixStoreSize2]);
      sumSimMatrixH[ja]=&(sumSimMatrixHStore2[simMatrixStoreSize2]);
      simMatrixStoreSize2=simMatrixStoreSize2+maxSize;
      ja++;
    }
    
    
    
    long scoreStoreSize1;
    
    scoreStoreSize1=0;
    
    ScoreT* scoreStore1=(ScoreT*)malloc(sizeof(ScoreT)*nrSetComparisons*featureParamInd.size());
    ScoreT** score=(ScoreT**)malloc(sizeof(ScoreT*)*featureParamInd.size());
    for(long i=0L; i<featureParamInd.size(); i++) {
      score[i]=&(scoreStore1[scoreStoreSize1]);
      scoreStoreSize1=scoreStoreSize1+nrSetComparisons;
    }
    
    
    
    /*#ifdef multiproc
    #pragma omp parallel for
    #endif*/
    for(long i=0L; i<nrIntegers; i++) {
      long curProc=0L;
      //#ifdef multiproc
      //curProc=omp_get_thread_num();
      //curProc=i
      //#endif
      
      long randNr=gsl_rng_uniform_int(r[i], smax*smax-smin*smin+1L)+smin*smin;
      long* factors=NULL;
      
      long nrRandNrFactors=0L;
      long* factors0=NULL;
      while(nrRandNrFactors==0L) {
        if(factors0!=NULL) {
          free(factors0);
        }
        randNr=gsl_rng_uniform_int(r[i], smax*smax-smin*smin+1L)+smin*smin;
        factors0=getIntegerFactors(randNr, smin, smax, nrRandNrFactors);
      }
      
      if(nrRandNrFactors<nrIntegerFactors) {
        factors=(long*)malloc(sizeof(long)*nrIntegerFactors);
        long j=0L;
        for(j=0; j<nrRandNrFactors; j++)
          factors[j]=factors0[j];
        while(j<nrIntegerFactors) {
          factors[j]=factors0[gsl_rng_uniform_int(r[i], nrRandNrFactors)];
          j++;
        }
        free(factors0);
      }
      else if(nrRandNrFactors>nrIntegerFactors) {
        long j=0L;
        while(j<nrRandNrFactors-nrIntegerFactors) {
          long rem=gsl_rng_uniform_int(r[i], nrRandNrFactors-j);
          for(long k=rem; k<nrRandNrFactors-j-1L; k++)
            factors0[k]=factors0[k+1L];
          j++;
        }
        factors=factors0;
      }
      else {
        factors=factors0;
      }
      
      for(long j=0L; j<nrIntegerFactors; j++) {
        long* selCompounds1;
        long nrSelCompounds1;
        long* selCompounds2;
        long nrSelCompounds2;
        
        randomSample(r[i], &selCompounds1, &nrSelCompounds1, std::min((long)factors[j], (long)sampleClusterEval.size()/2), sampleClusterEval.size());
        randomSample(r[i], &selCompounds2, &nrSelCompounds2, std::min((long)randNr/factors[j], (long)sampleClusterEval.size()/2), sampleClusterEval.size());
        
        long finalInd=i*nrIntegerFactors+j;
        fprintf(stderr, "finalInd: %ld\n", finalInd);
        
        
        string selCmpFilename;
        FILE* selCmp;
        
        std::stringstream ss;
        ss << clusterIndEval << "_" << finalInd;
        
        selCmpFilename=predSavePathname+"checkSel1_"+ss.str()+".txt";
        selCmp=fopen(selCmpFilename.c_str(), "w");
        for(long cmpNr=0L; cmpNr<nrSelCompounds1; cmpNr++)
          fprintf(selCmp, "%ld\n", sampleClusterEval[selCompounds1[cmpNr]]);
        fclose(selCmp);

        selCmpFilename=predSavePathname+"checkSel2_"+ss.str()+".txt";
        selCmp=fopen(selCmpFilename.c_str(), "w");
        for(long cmpNr=0L; cmpNr<nrSelCompounds2; cmpNr++)
          fprintf(selCmp, "%ld\n", sampleClusterEval[selCompounds2[cmpNr]]);
        fclose(selCmp);
        
        
        
        std::fill(sSimMatrix[curProc][0][0], sSimMatrix[curProc][0][0]+maxSize*maxSize*sparseFeatureKernelVec.size(), 0.0f);
        std::fill(sSimMatrixH[curProc][0][0], sSimMatrixH[curProc][0][0]+maxSize*maxSize*sparseFeatureKernelVec.size(), 0.0f);
        std::fill(dSimMatrix[curProc][0][0], dSimMatrix[curProc][0][0]+maxSize*maxSize*denseFeatureKernelVec.size(), 0.0f);
        std::fill(dSimMatrixH[curProc][0][0], dSimMatrixH[curProc][0][0]+maxSize*maxSize*denseFeatureKernelVec.size(), 0.0f);
        
        
        
        for(long k=0L; k<sparseFeatureKernelVec.size(); k++) {
          long* samples=((sparseFeatureKernelVec[k])->feature)->samples;
          long* features=((sparseFeatureKernelVec[k])->feature)->features;
          double* featureCounts=((sparseFeatureKernelVec[k])->feature)->featureCounts;
          KernelType ktype=(sparseFeatureKernelVec[k])->kt;
          
          #ifdef multiproc
          #pragma omp parallel for
          #endif
          for(long m=0L; m<nrSelCompounds2; m++) {
            long sampleInd2=sampleClusterEval[selCompounds2[m]];
            long beginFeat2=samples[sampleInd2];
            long endFeat2=samples[sampleInd2+1L];
            
            for(long l=0L; l<nrSelCompounds1; l++) {
              long sampleInd1=sampleClusterEval[selCompounds1[l]];
              long beginFeat1=samples[sampleInd1];
              long endFeat1=samples[sampleInd1+1L];

              
              if(ktype==LIN) {
                double res=linearSumSparseCount(features, featureCounts, beginFeat1, endFeat1, features, featureCounts, beginFeat2, endFeat2);
                sSimMatrix[curProc][k][l][m]=res;
                sSimMatrixH[curProc][k][l][m]=res;
              }
              else if(ktype==GAUSS) {
                double res=gaussianSumSparseCount(features, featureCounts, beginFeat1, endFeat1, features, featureCounts, beginFeat2, endFeat2);
                sSimMatrix[curProc][k][l][m]=res;
                sSimMatrixH[curProc][k][l][m]=res;
              }
              else if(ktype==TAN) {
                sumRet_t res=tanimotoSumSparseCount(features, featureCounts, beginFeat1, endFeat1, features, featureCounts, beginFeat2, endFeat2);
                sSimMatrix[curProc][k][l][m]=res.sum1;
                sSimMatrixH[curProc][k][l][m]=res.sum2;
              }
              else if(ktype==TAN2) {
                sumRet_t res=tanimotoSumSparseCount2(features, featureCounts, beginFeat1, endFeat1, features, featureCounts, beginFeat2, endFeat2);
                sSimMatrix[curProc][k][l][m]=res.sum1;
                sSimMatrixH[curProc][k][l][m]=res.sum2;
              }
            }
          }
        }
        
        for(long k=0L; k<denseFeatureKernelVec.size(); k++) {
          double** properties=((denseFeatureKernelVec[k])->feature)->properties;
          long propertyNr=((denseFeatureKernelVec[k])->feature)->propertyNr;
          long sampleNr=((denseFeatureKernelVec[k])->feature)->sampleNr;
          KernelType ktype=(denseFeatureKernelVec[k])->kt;
          
          #ifdef multiproc
          #pragma omp parallel for
          #endif
          for(long m=0L; m<nrSelCompounds2; m++) {
            long sampleInd2=sampleClusterEval[selCompounds2[m]];

            for(long l=0L; l<nrSelCompounds1; l++) {
              long sampleInd1=sampleClusterEval[selCompounds1[l]];

              
              if(ktype==LIN) {
                double res=linearSumDense(properties[sampleInd1], properties[sampleInd2], propertyNr);
                dSimMatrix[curProc][k][l][m]=res;
                dSimMatrixH[curProc][k][l][m]=res;
              }
              else if(ktype==GAUSS) {
                double res=gaussianSumDense(properties[sampleInd1], properties[sampleInd2], propertyNr);
                dSimMatrix[curProc][k][l][m]=res;
                dSimMatrixH[curProc][k][l][m]=res;
              }
              else if(ktype==TAN) {
                sumRet_t res=tanimotoSumDenseOrig(properties[sampleInd1], properties[sampleInd2], propertyNr);
                dSimMatrix[curProc][k][l][m]=res.sum1;
                dSimMatrixH[curProc][k][l][m]=res.sum2;
              }
              else if(ktype==TANS) {
                sumRet_t res=tanimotoSumDenseOrigSplit(properties[sampleInd1], properties[sampleInd2], propertyNr);
                dSimMatrix[curProc][k][l][m]=res.sum1;
                dSimMatrixH[curProc][k][l][m]=res.sum2;
              }
              else if(ktype==TAN2) {
                sumRet_t res=tanimotoSumDenseOrig2(properties[sampleInd1], properties[sampleInd2], propertyNr);
                dSimMatrix[curProc][k][l][m]=res.sum1;
                dSimMatrixH[curProc][k][l][m]=res.sum2;
              }
            }
          }
        }
        
        
        
        
        
        
        for(long paramInd=0L; paramInd<featureParamInd.size()-1L; paramInd++) {
          //fprintf(stderr, " paramInd:%ld, ", paramInd);
          
          std::fill(sumSimMatrix[curProc][0], sumSimMatrix[curProc][0]+maxSize*maxSize, 0.0f);
          std::fill(sumSimMatrixH[curProc][0], sumSimMatrixH[curProc][0]+maxSize*maxSize, 0.0f);

          
          FeatureKernelCollection featureParam=paramComb[paramInd].fkc;
          long featureParamId=featureParam.id;
          KernelComb kc=featureParam.kc;
          long kn=featureParam.kn;
          
          
          
          for(long k=0L; k<featureParam.sFeatureKernels.size()+featureParam.dFeatureKernels.size(); k++) {
            long matInd;
            float** sim;
            float** simH;
            if(k<featureParam.sFeatureKernels.size()) {
              matInd=featureParam.sFeatureKernels[k]->id;
              sim=sSimMatrix[curProc][matInd];
              simH=sSimMatrixH[curProc][matInd];
            }
            else {
              matInd=featureParam.dFeatureKernels[k-featureParam.sFeatureKernels.size()]->id;
              sim=dSimMatrix[curProc][matInd];
              simH=dSimMatrixH[curProc][matInd];
            }
            
            #ifdef multiproc
            #pragma omp parallel for
            #endif
            for(long m=0; m<nrSelCompounds2; m++) {
              for(long l=0; l<nrSelCompounds1; l++) {
                sumSimMatrix[curProc][l][m]=sumSimMatrix[curProc][l][m]+sim[l][m];
                sumSimMatrixH[curProc][l][m]=sumSimMatrixH[curProc][l][m]+simH[l][m];
              }
            }
          }
          
          if(kc==TANCOMB||kc==TANCOMB2||kc==TANSCOMB) {
            #ifdef multiproc
            #pragma omp parallel for
            #endif
            for(long l=0; l<nrSelCompounds2; l++) {
              for(long k=0; k<nrSelCompounds1; k++) {
                if(sumSimMatrixH[curProc][k][l]>=0.1)
                  sumSimMatrix[curProc][k][l]=sumSimMatrix[curProc][k][l]/sumSimMatrixH[curProc][k][l];
                else
                  sumSimMatrix[curProc][k][l]=0.0;
              }
            }
          }
          
          
          
          double* scoreArr=(double*)malloc(sizeof(double)*nrTests);
          for(long k=0; k<nrTests; k++)
            scoreArr[k]=0.0;
          
          
          
          #ifdef multiproc
          #pragma omp parallel for
          #endif
          for(long l=0; l<nrSelCompounds2; l++) {
            for(long k=0; k<nrSelCompounds1; k++) {
              double Tm=sumSimMatrix[curProc][k][l];
              double threshold=minThreshold;
              for(long m=0; m<nrTests; m++) {
                if(threshold>=Tm)
                  break;
                scoreArr[m]=scoreArr[m]+Tm;
                threshold=threshold+stepsize;
              }
            }
          }
          
          score[paramInd][i*nrIntegerFactors+j].size=nrSelCompounds1*nrSelCompounds2;
          score[paramInd][i*nrIntegerFactors+j].score=scoreArr;
        }
      }
      
      free(factors);
    }
    
    
    /*for(long paramInd=0L; paramInd<featureParamInd.size()-1L; paramInd++) {
      std::stringstream ss;
      ss << clusterIndEval << "_" << paramInd;
      string scoreSizeFilename=predSavePathname+"scoreSize_"+ss.str()+".txt";
      FILE* scoreSizeFile=fopen(scoreSizeFilename.c_str(), "w");
      string scoreFilename=predSavePathname+"score_"+ss.str()+".txt";
      FILE* scoreFile=fopen(scoreFilename.c_str(), "w");
      for(long i=0; i<nrSetComparisons; i++) {
        fprintf(scoreSizeFile, "%ld\n", score[paramInd][i].size);
        for(long j=0; j<nrTests; j++)
          fprintf(scoreFile, "%10.10lf,", score[paramInd][i].score[j]);
        fprintf(scoreFile, "\n");
      }
      fclose(scoreSizeFile);
      fclose(scoreFile);
    }*/
    
    for(long paramInd=0L; paramInd<featureParamInd.size()-1L; paramInd++) {
      std::stringstream ss;
      ss << clusterIndEval << "_" << paramInd;
      string scoreSizeFilename=predSavePathname+"scoreSize_"+ss.str()+".bin";
      FILE* scoreSizeFile=fopen(scoreSizeFilename.c_str(), "wb");
      string scoreFilename=predSavePathname+"score_"+ss.str()+".bin";
      FILE* scoreFile=fopen(scoreFilename.c_str(), "wb");
      for(long i=0; i<nrSetComparisons; i++) {
        fwrite(&score[paramInd][i].size, sizeof(long), 1, scoreSizeFile);
        fwrite(score[paramInd][i].score, sizeof(double), nrTests, scoreFile);
        free(score[paramInd][i].score);
      }
      fclose(scoreSizeFile);
      fclose(scoreFile);
    }
    

    
    
    
    
    
    free(score);
    free(scoreStore1);
    
    free(sumSimMatrixH);
    free(sumSimMatrixHStore2);
    free(sumSimMatrixHStore1);
    free(sumSimMatrix);
    free(sumSimMatrixStore2);
    free(sumSimMatrixStore1);
    
    free(dSimMatrixH);
    free(dSimMatrixHStore3);
    free(dSimMatrixHStore2);
    free(dSimMatrixHStore1);
    free(dSimMatrix);
    free(dSimMatrixStore3);
    free(dSimMatrixStore2);
    free(dSimMatrixStore1);
    
    free(sSimMatrixH);
    free(sSimMatrixHStore3);
    free(sSimMatrixHStore2);
    free(sSimMatrixHStore1);
    free(sSimMatrix);
    free(sSimMatrixStore3);
    free(sSimMatrixStore2);
    free(sSimMatrixStore1);
    
    
    
    delete[] validEvalSamples;
  }
  
  for(long i=0; i<nrIntegers; i++)
    gsl_rng_free(r[i]);
  
  delete[] r;
  
  
  
  
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
  
  
  
  printf("seaHyper1 terminated successfully!\n");

  return 0;
}
