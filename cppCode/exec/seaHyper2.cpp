/*
Copyright (C) 2018 Andreas Mayr
Licensed under GNU General Public License v3.0 (see http://www.bioinf.jku.at/research/lsc/LICENSE and https://github.com/ml-jku/lsc/blob/master/LICENSE)
*/



#ifdef multiproc
#include <omp.h>
#endif

#include "dlib/optimization.h"
#include <boost/math/special_functions/erf.hpp>
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
//using namespace boost::math;
//using namespace dlib;

typedef struct ScoreT {
  long size;
  double* score;
} ScoreT;

int compareDouble(const void * a, const void * b) {
  if(((*(double*)a - *(double*)b))<0.0)
    return -1;
  if(((*(double*)a - *(double*)b))>0.0)
    return 1;
  return 0;
}

int compareScore (const void* a, const void* b) {
  if((((*(ScoreT*)a).size - (*(ScoreT*)b).size))<0L)
    return -1;
  if((((*(ScoreT*)a).size - (*(ScoreT*)b).size))>0L)
    return 1;
  return 0;
}

void linearLS(double* x, double* y, long length, double* a, double* b) {
  double meanX=0.0;
  double meanY=0.0;
  double X2=0.0;
  double XY=0.0;
  
  for(long i=0L; i<length; i++) {
    meanX=meanX+x[i];
    meanY=meanY+y[i];
    X2=X2+x[i]*x[i];
    XY=XY+x[i]*y[i];
  }
  meanX=meanX/length;
  meanY=meanY/length;
  if(sqrt(fabs(X2-length*meanX*meanX)/length)<0.00000000001) {
    *a=meanY;
    *b=0.0;
  }
  else {
    *a=(meanY*X2-meanX*XY)/(X2-length*meanX*meanX); 
    *b=(XY-length*meanX*meanY)/(X2-length*meanX*meanX);
  }
}

typedef dlib::matrix<double,1,1> input_vector;
typedef dlib::matrix<double,3,1> parameter_vector;

double model(const input_vector& input, const parameter_vector& params) {
  //params: m, n, p
  double m=params(0);
  double n=params(1);
  double p=params(2);
  double x=input(0);
  double myp=pow(x, n);
  return m*myp+p;
}

double residual(const std::pair<input_vector, double>& data, const parameter_vector& params) {
  return model(data.first, params) - data.second;
}

parameter_vector residual_derivative(const std::pair<input_vector, double>& data, const parameter_vector& params) {
  parameter_vector der;
  double m=params(0);
  double n=params(1);
  double p=params(2);
  double x=data.first(0);
  double y=data.second;
  double myp=pow(x, n);
  double mGrad=myp;
  double pGrad=1.0;
  double nGrad=myp*m*log(x);
  der(0)=mGrad;
  der(1)=nGrad;
  der(2)=pGrad;
  return der;
}

void nonlinearLS(double* x, double* y, long length, double& m, double& n, double& p) {
  std::vector<std::pair<input_vector, double> > data_samples;
  for(long i=0; i<length; i++) {
    input_vector input;
    input(0)=x[i];
    data_samples.push_back(make_pair(input, y[i]));
  }
  parameter_vector param;
  //param = 0, 0, 0;
  param = 1, 1, 0;
  dlib::solve_least_squares_lm(dlib::objective_delta_stop_strategy(1e-7, 100), residual, residual_derivative, data_samples, param);
  m=param(0);
  n=param(1);
  p=param(2);
}

void chiSQGumbel(double* values, long nrValues, long Nbins, double* chi, double* chiNorm) {
  double pi=M_PI;
  double EM1=0.577215665;
  double* sortedValues=(double*)malloc(sizeof(double)*nrValues);
  double* expected=(double*)malloc(sizeof(double)*Nbins);
  double* observed=(double*)malloc(sizeof(double)*Nbins);
  
  for(long i=0L; i<nrValues; i++)
    sortedValues[i]=values[i];
  qsort(sortedValues, nrValues, sizeof(double), compareDouble);
  
  *chi=0.0;
  *chiNorm=0.0;
  
  double expectedProb;
  double expectedValue;
  double expectedProbNorm;
  double expectedValueNorm;
  
  long sI=0L;
  long observedValue=0;
  double oldExp=0.0;
  double oldNorm=0.0;
  for(long i=1; i<Nbins; i++) {
    double border=sqrt(2.0)*boost::math::erf_inv(2.0*((1.0)/((double)Nbins))*i-1.000);
    observedValue=0;
    for(; sI<nrValues; sI++) {
      if(sortedValues[sI]<border)
        observedValue++;
      else {
        break;
      }
    }
    
    double evalExp=exp(-1.0*exp((-border*pi)/sqrt(6.0)-EM1));
    double evalNorm=0.5*(1+erf(border*M_SQRT1_2));

    expectedProb=evalExp-oldExp;
    expectedValue=expectedProb*((double)nrValues);
    expectedProbNorm=evalNorm-oldNorm;
    expectedValueNorm=expectedProbNorm*((double)nrValues);
    oldExp=evalExp;
    oldNorm=evalNorm;
    
    *chi=*chi+((double)((observedValue-expectedValue)*(observedValue-expectedValue)))/((double)(expectedValue+observedValue));
    *chiNorm=*chiNorm+((double)((observedValue-expectedValueNorm)*(observedValue-expectedValueNorm)))/((double)(expectedValueNorm+observedValue));
    
    expected[i]=(double)expectedValue;
    observed[i]=(double)observedValue;
  }
  
  observedValue=nrValues-sI;
  
  expectedProb=1.0-oldExp;
  expectedValue=expectedProb*((double)nrValues);
  expectedProbNorm=1.0-oldNorm;
  expectedValueNorm=expectedProbNorm*((double)nrValues);
  
  *chi=*chi+((double)((observedValue-expectedValue)*(observedValue-expectedValue)))/((double)(expectedValue+observedValue));
  *chiNorm=*chiNorm+((double)((observedValue-expectedValueNorm)*(observedValue-expectedValueNorm)))/((double)(expectedValueNorm+observedValue));

  expected[Nbins-1L]=(double)expectedValue;
  observed[Nbins-1L]=(double)observedValue;
  
  free(observed);
  free(expected);
  free(sortedValues);
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
  
  
  
  
  
  
  

  
  
  
  for(long clusterIndEval=0L; clusterIndEval<nrClusters; clusterIndEval++) {
    for(long paramInd=0L; paramInd<featureParamInd.size()-1L; paramInd++) {
      std::stringstream ss;
      ss << clusterIndEval << "_" << paramInd;
      string scoreSizeFilename=predSavePathname+"scoreSize_"+ss.str()+".bin";
      FILE* scoreSizeFile=fopen(scoreSizeFilename.c_str(), "r");
      string scoreFilename=predSavePathname+"score_"+ss.str()+".bin";
      FILE* scoreFile=fopen(scoreFilename.c_str(), "r");
      
      fseek(scoreSizeFile, 0, SEEK_END);
      long nrSetComparisons=(ftell(scoreSizeFile)/sizeof(long));
      rewind(scoreSizeFile);
      
      fseek(scoreFile, 0, SEEK_END);
      long nrTests=(ftell(scoreFile)/(nrSetComparisons*sizeof(double)));
      rewind(scoreFile);
      
      ScoreT* score=(ScoreT*)malloc(sizeof(ScoreT)*nrSetComparisons);
      
      for(long i=0; i<nrSetComparisons; i++) {
        double* scoreArr=(double*)malloc(sizeof(double)*nrTests);
        score[i].score=scoreArr;
        
        fread(&score[i].size, sizeof(long), 1, scoreSizeFile);
        fread(score[i].score, sizeof(double), nrTests, scoreFile);
      }
      fclose(scoreSizeFile);
      fclose(scoreFile);
      
      
      
      qsort(score, nrSetComparisons, sizeof(ScoreT), compareScore);
      double* scoreSize=(double*)malloc(sizeof(double)*nrSetComparisons);
      for(long i=0L; i<nrSetComparisons; i++)
        scoreSize[i]=score[i].size;
      double** tscore=(double**)malloc(sizeof(double*)*nrTests);
      for(long i=0L; i<nrTests; i++) {
        tscore[i]=(double*)malloc(sizeof(double)*nrSetComparisons);
        for(long j=0L; j<nrSetComparisons; j++) {
          tscore[i][j]=score[j].score[i];
        }
      }
      
      
      
      double *mnlMean=(double*)malloc(sizeof(double)*nrTests);
      double *nnlMean=(double*)malloc(sizeof(double)*nrTests);
      double *pnlMean=(double*)malloc(sizeof(double)*nrTests);
      
      for(long i=0L; i<nrTests; i++) {
        nonlinearLS(scoreSize, tscore[i], nrSetComparisons, mnlMean[i], nnlMean[i], pnlMean[i]);
        
        /*std::stringstream ss;
        ss << clusterIndEval << "_" << paramInd << "_" << i;
        string myFilename1=predSavePathname+"mean_"+ss.str()+".txt";
        FILE* myFile1=fopen(myFilename1.c_str(), "w");
        string myFilename2=predSavePathname+"meanRes_"+ss.str()+".txt";
        FILE* myFile2=fopen(myFilename2.c_str(), "w");
        for(long j=0; j<nrSetComparisons; j++)
          fprintf(myFile1, "%10.20lf,%10.20lf\n", scoreSize[j], tscore[i][j]);
        fprintf(myFile2, "%10.20lf\n%10.20lf\n%10.20lf\n", mnlMean[i], nnlMean[i], pnlMean[i]);
        fclose(myFile2);
        fclose(myFile1);*/
      }
      
      
      
      double* scoreSizeMean=(double*)malloc(sizeof(double*)*nrSetComparisons);
      double** scoreStd=(double**)malloc(sizeof(double**)*nrSetComparisons);
      long counter=0L;
      
      {
        double nrIntervals=floor(0.1*nrSetComparisons);
        long nrIntervalsL=(long)nrIntervals;
        long minSize=scoreSize[0];
        long maxSize=scoreSize[nrSetComparisons-1];
        double intLength=((double)(maxSize-minSize))/((double)(nrIntervals));
        double endInt=minSize+intLength;
        long j=0;

        scoreSizeMean[counter]=0.0;
        for(long i=0; i<nrIntervalsL; i++) {
          long nrElems=0;
          scoreSizeMean[counter]=0.0;
          
          while(((scoreSize[j]<=endInt)||(i==nrIntervalsL-1L))&&(j<nrSetComparisons)) {
            scoreSizeMean[counter]=scoreSizeMean[counter]+scoreSize[j];
            nrElems++;
            j++;
          }
          
          if(nrElems>=5) {
            scoreSizeMean[counter]=scoreSizeMean[counter]/((double)nrElems);
            scoreStd[counter]=(double*)malloc(sizeof(double*)*nrTests);
            for(long k=0L; k<nrTests; k++) {
              double sum=0.0;
              for(long l=j-nrElems; l<j; l++) {
                double ExpRaw=pnlMean[k]+mnlMean[k]*pow(scoreSize[l], nnlMean[k]);
                sum=sum+(tscore[k][l]-ExpRaw)*(tscore[k][l]-ExpRaw);
              }
              scoreStd[counter][k]=sqrt(sum/((double)nrElems+0.00000000001));
            }

            counter++;
          }
          endInt=endInt+intLength;
        }
      }
      
      
      
      double** tscoreStd=(double**)malloc(sizeof(double*)*nrTests);
      for(long i=0L; i<nrTests; i++) {
        tscoreStd[i]=(double*) malloc(sizeof(double)*counter);
        for(long j=0L; j<counter; j++) {
          tscoreStd[i][j]=scoreStd[j][i];
        }
      }
      
      
      
      double *mnl=(double*)malloc(sizeof(double)*nrTests);
      double *nnl=(double*)malloc(sizeof(double)*nrTests);
      double *pnl=(double*)malloc(sizeof(double)*nrTests);
      for(long i=0L; i<nrTests; i++) {
        nonlinearLS(scoreSizeMean, tscoreStd[i], counter, mnl[i], nnl[i], pnl[i]);
        
        /*std::stringstream ss;
        ss << clusterIndEval << "_" << paramInd << "_" << i;
        string myFilename1=predSavePathname+"std_"+ss.str()+".txt";
        FILE* myFile1=fopen(myFilename1.c_str(), "w");
        string myFilename2=predSavePathname+"stdRes_"+ss.str()+".txt";
        FILE* myFile2=fopen(myFilename2.c_str(), "w");
        for(long j=0; j<counter; j++)
          fprintf(myFile1, "%10.20lf,%10.20lf\n", scoreSizeMean[j], tscoreStd[i][j]);
        fprintf(myFile2, "%10.20lf\n%10.20lf\n%10.20lf\n", mnl[i], nnl[i], pnl[i]);
        fclose(myFile2);
        fclose(myFile1);*/
      }
      
      
      
      double** Zvalues=(double**)malloc(sizeof(double*)*nrTests);
      for(long i=0L; i<nrTests; i++) {
        /*std::stringstream ss;
        ss << clusterIndEval << "_" << paramInd << "_" << i;
        string myFilename1=predSavePathname+"zscore_"+ss.str()+".txt";
        FILE* myFile1=fopen(myFilename1.c_str(), "w");*/
        
        Zvalues[i]=(double*)malloc(sizeof(double)*nrSetComparisons);
        for(long j=0L; j<nrSetComparisons; j++) {
          double Eraw=pnlMean[i]+mnlMean[i]*pow(scoreSize[j], nnlMean[i]);
          double Estd=pnl[i]+mnl[i]*pow(scoreSize[j], nnl[i]);
          Zvalues[i][j]=(tscore[i][j]-Eraw)/(Estd+0.00000000001);
          //fprintf(myFile1, "%10.20lf\n", Zvalues[i][j]);
        }
        //fclose(myFile1);
      }
      
      
      
      string outChiFilename=predSavePathname+"chi_"+ss.str()+".txt";
      FILE* outChiFile;
      outChiFile=fopen(outChiFilename.c_str(), "w");

      double bestChi=numeric_limits<double>::max();
      long bestChiI=0L;
      double bestChiNorm=numeric_limits<double>::max();
      
      double bestDiff=0.0;
      long bestDiffI=0L;
      double bestDiffNorm=0.0;

      for(long i=0L; i<nrTests; i++) {
        double chi;
        double chiNorm;
        chiSQGumbel(Zvalues[i], nrSetComparisons, 100L, &chi, &chiNorm);
        fprintf(outChiFile, "%ld, %lf, %lf\n", i, chi, chiNorm);
        
        if(chi<bestChi) {
          bestChi=chi;
          bestChiI=i;
          bestChiNorm=chiNorm;
        }
        if(chi-chiNorm<bestDiff) {
          bestDiff=chi-chiNorm;
          bestDiffI=i;
          bestDiffNorm=chiNorm;
        }
      }
      
      fclose(outChiFile);
     
      
      double stepsize=0.01;
      double minThreshold=0.00;
      double maxThreshold=1.00;

      double bestThreshold=minThreshold+bestChiI*stepsize;
      
      double bestmnlMean=mnlMean[bestChiI];
      double bestnnlMean=nnlMean[bestChiI];
      double bestpnlMean=pnlMean[bestChiI];
      double bestmnl=mnl[bestChiI];
      double bestnnl=nnl[bestChiI];
      double bestpnl=pnl[bestChiI];
      
      fprintf(stderr, "bestChi: %.10lf\n", bestChi);
      fprintf(stderr, "bestChiNorm: %.10lf\n", bestChiNorm);
      fprintf(stderr, "bestChiI: %ld\n", bestChiI);
      fprintf(stderr, "bestThreshold: %.10lf\n", bestThreshold);

      fprintf(stderr, "bestmnlMean: %.10lf\n", bestmnlMean);
      fprintf(stderr, "bestnnlMean: %.10lf\n", bestnnlMean);
      fprintf(stderr, "bestpnlMean: %.10lf\n", bestpnlMean);
      fprintf(stderr, "bestmnl: %.10lf\n", bestmnl);
      fprintf(stderr, "bestnnl: %.10lf\n", bestnnl);
      fprintf(stderr, "bestpnl: %.10lf\n", bestpnl);
      
      /*bestThreshold=minThreshold+bestDiffI*stepsize;
      
      bestmnlMean=mnlMean[bestDiffI];
      bestnnlMean=nnlMean[bestDiffI];
      bestpnlMean=pnlMean[bestDiffI];
      bestmnl=mnl[bestDiffI];
      bestnnl=nnl[bestDiffI];
      bestpnl=pnl[bestDiffI];
      
      fprintf(stderr, "bestChi: %.10lf\n", bestDiff);
      fprintf(stderr, "bestChiNorm: %.10lf\n", bestDiffNorm);
      fprintf(stderr, "bestChiI: %ld\n", bestDiffI);
      fprintf(stderr, "bestThreshold: %.10lf\n", bestThreshold);

      fprintf(stderr, "bestmnlMean: %.10lf\n", bestmnlMean);
      fprintf(stderr, "bestnnlMean: %.10lf\n", bestnnlMean);
      fprintf(stderr, "bestpnlMean: %.10lf\n", bestpnlMean);
      fprintf(stderr, "bestmnl: %.10lf\n", bestmnl);
      fprintf(stderr, "bestnnl: %.10lf\n", bestnnl);
      fprintf(stderr, "bestpnl: %.10lf\n", bestpnl);*/
      
      
      
      string parFilename=predSavePathname+"par_"+ss.str()+".bin";
      FILE* parFile=fopen(parFilename.c_str(), "wb");
      fwrite(&bestThreshold, sizeof(double), 1L, parFile);
      fwrite(&bestmnlMean, sizeof(double), 1L, parFile);
      fwrite(&bestnnlMean, sizeof(double), 1L, parFile);
      fwrite(&bestpnlMean, sizeof(double), 1L, parFile);
      fwrite(&bestmnl, sizeof(double), 1L, parFile);
      fwrite(&bestnnl, sizeof(double), 1L, parFile);
      fwrite(&bestpnl, sizeof(double), 1L, parFile);
      fclose(parFile);
      
      
      
      for(long i=0L; i<nrTests; i++)
        free(Zvalues[i]);
      free(Zvalues);
      
      free(mnl);
      free(nnl);
      free(pnl);
      
      for(long i=0L; i<nrTests; i++)
        free(tscoreStd[i]);
      free(tscoreStd);
      for(long i=0L; i<counter; i++)
        free(scoreStd[i]);
      free(scoreStd);
      free(scoreSizeMean);
      
      
      
      free(pnlMean);
      free(nnlMean);
      free(mnlMean);
      
      
      
      for(long i=0L; i<nrTests; i++)
        free(tscore[i]);
      free(tscore);
      free(scoreSize);
      
      for(long i=0; i<nrSetComparisons; i++)
        free(score[i].score);
      free(score);
    }
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
  
  
  
  printf("seaHyper2 terminated successfully!\n");

  return 0;
}