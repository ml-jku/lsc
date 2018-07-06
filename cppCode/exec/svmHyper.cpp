/*
Copyright (C) 2018 Andreas Mayr
Licensed under GNU General Public License v3.0 (see http://www.bioinf.jku.at/research/lsc/LICENSE and https://github.com/ml-jku/lsc/blob/master/LICENSE)
*/



#define _DENSE_REP

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
#include "libsvm/svm.h"
#include <iostream>
#include <libconfig.h++>
#include "compoundData.h"
#include "targetPipeline.h"
#include "sums.h"
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

using namespace std;
using namespace libconfig;

void print_null(const char *s) {}

int main(int argc, char** argv) {
  
  if(sizeof(long)<8) {
    printf("This program is optimized for machines with at least 8 byte longs!  The program will terminate!\n");
    return -1;
  }
  string basePathname(getenv("HOME")); basePathname=basePathname+"/mydata/trgpred/";
  
  
  
  struct stat sb;
  int statRet;
  
  if(argc!=3) {
    printf("Usage: svmHyper config maxProc\n");
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
  
  
  
  double eps=cfg.lookup("eps");
  
  
  
  int shrinking=cfg.lookup("shrinking");
  if((shrinking!=0)&&(shrinking!=1)) {
    fprintf(stderr, "shrinking must be 0 or 1!\n");
    exit(-1);
  }
  
  
  
  double w1=cfg.lookup("w1");
  
  double w2=cfg.lookup("w2");
  
  
  
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
  
  
  
  std::vector<long> paramGTypes;
  for(int i=0; i<cfg.lookup("paramGTypes").getLength(); i++)
    paramGTypes.push_back(cfg.lookup("paramGTypes")[i]);
  long defaultG=cfg.lookup("defaultG");
  
  std::vector<double> paramRBFTypes;
  for(int i=0; i<cfg.lookup("paramRBFTypes").getLength(); i++)
    paramRBFTypes.push_back(cfg.lookup("paramRBFTypes")[i]);
  double defaultRBF=cfg.lookup("defaultRBF");  
  
  
  
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
  
  std::vector<SVMHyperParam> paramComb;
  for(long i=0L; i<paramFeatureTypes.size(); i++) {
    double quantVec[7];
    
    if(paramFeatureTypes[i].sim==1) {
      std::fill(simMatrixStatStore, simMatrixStatStore+statSamples*statSamples, 0.0f);
      std::fill(simMatrixStatStoreH, simMatrixStatStoreH+statSamples*statSamples, 0.0f);
      
      for(long j=0L; j<paramFeatureTypes[i].sFeatureKernels.size(); j++) {
        long matInd=paramFeatureTypes[i].sFeatureKernels[j]->id;
        KernelType ktype=sparseFeatureKernelVec[matInd]->kt;
        long* samples=((sparseFeatureKernelVec[matInd])->feature)->samples;
        long* features=((sparseFeatureKernelVec[matInd])->feature)->features;
        double* featureCounts=((sparseFeatureKernelVec[matInd])->feature)->featureCounts;
        
        #ifdef multiproc
        #pragma omp for
        #endif
        for(long k=0L; k<statSamples; k++) {
          long sampleInd1=myrand[k];
          long beginFeat1=samples[sampleInd1];
          long endFeat1=samples[sampleInd1+1L];
          
          for(long l=0L; l<statSamples; l++) {
            long sampleInd2=myrand[l];
            long beginFeat2=samples[sampleInd2];
            long endFeat2=samples[sampleInd2+1L];
            
            float part1=0.0;
            float part2=0.0;
            if(ktype==LIN) {
              part1=linearSumSparseCount(features, featureCounts, beginFeat1, endFeat1, features, featureCounts, beginFeat2, endFeat2);
              part2=part1;
            }
            else if(ktype==GAUSS) {
              part1=gaussianSumSparseCount(features, featureCounts, beginFeat1, endFeat1, features, featureCounts, beginFeat2, endFeat2);
              part2=part1;
            }
            else if(ktype==TAN) {
              sumRet_t res=tanimotoSumSparseCount(features, featureCounts, beginFeat1, endFeat1, features, featureCounts, beginFeat2, endFeat2);
              part1=res.sum1;
              part2=res.sum2;
            }
            else if(ktype==TAN2) {
              sumRet_t res=tanimotoSumSparseCount2(features, featureCounts, beginFeat1, endFeat1, features, featureCounts, beginFeat2, endFeat2);
              part1=res.sum1;
              part2=res.sum2;
            }
            
            simMatrixStat[l][k]=simMatrixStat[l][k]+part1;
            simMatrixStatH[l][k]=simMatrixStatH[l][k]+part2;
          }
        }
      }
      
      for(long j=0L; j<paramFeatureTypes[i].dFeatureKernels.size(); j++) {
        long matInd=paramFeatureTypes[i].dFeatureKernels[j]->id;
        KernelType ktype=denseFeatureKernelVec[matInd]->kt;
        double** properties=((denseFeatureKernelVec[matInd])->feature)->properties;
        long propertyNr=((denseFeatureKernelVec[matInd])->feature)->propertyNr;
        long sampleNr=((denseFeatureKernelVec[matInd])->feature)->sampleNr;
        
        #ifdef multiproc
        #pragma omp for
        #endif
        for(long k=0L; k<statSamples; k++) {
          long sampleInd1=myrand[k];

          for(long l=0L; l<statSamples; l++) {
            long sampleInd2=myrand[l];
            
            float part1=0.0;
            float part2=0.0;
            if(ktype==LIN) {
              part1=linearSumDense(properties[sampleInd1], properties[sampleInd2], propertyNr);
              part2=part1;
            }
            else if(ktype==GAUSS) {
              part1=gaussianSumDense(properties[sampleInd1], properties[sampleInd2], propertyNr);
              part2=part1;
            }
            else if(ktype==TAN) {
              sumRet_t res=tanimotoSumDenseOrig(properties[sampleInd1], properties[sampleInd2], propertyNr);
              part1=res.sum1;
              part2=res.sum2;
            }
            else if(ktype==TANS) {
              sumRet_t res=tanimotoSumDenseOrigSplit(properties[sampleInd1], properties[sampleInd2], propertyNr);
              part1=res.sum1;
              part2=res.sum2;
            }
            else if(ktype==TAN2) {
              sumRet_t res=tanimotoSumDenseOrig2(properties[sampleInd1], properties[sampleInd2], propertyNr);
              part1=res.sum1;
              part2=res.sum2;
            }
            
            simMatrixStat[l][k]=simMatrixStat[l][k]+part1;
            simMatrixStatH[l][k]=simMatrixStatH[l][k]+part2;
          }
        }
      }
      
      if(paramFeatureTypes[i].kc==TANCOMB||paramFeatureTypes[i].kc==TANCOMB2||paramFeatureTypes[i].kc==TANSCOMB) {
        #ifdef multiproc
        #pragma omp for
        #endif
        for(long j=0L; j<statSamples; j++) {
          for(long k=0L; k<statSamples; k++) {
            if(simMatrixStatH[k][j]>=0.1)
              simMatrixStat[k][j]=simMatrixStat[k][j]/simMatrixStatH[k][j];
            else
              simMatrixStat[k][j]=0.0;
          }
        }
      }
      
      #ifdef multiproc
      #pragma omp for
      #endif
      for(long j=0L; j<statSamples; j++) {
        for(long k=0L; k<statSamples; k++) {
          simMatrixStat[k][j]=abs(simMatrixStat[k][j]);
        }
      }
      
      
      
      gsl_sort(simMatrixStatStore, 1, statSamples*statSamples);
      quantVec[0]=gsl_stats_quantile_from_sorted_data (simMatrixStatStore, 1, statSamples*statSamples, 0.05)*2.0;
      quantVec[1]=gsl_stats_quantile_from_sorted_data (simMatrixStatStore, 1, statSamples*statSamples, 0.10)*2.0;
      quantVec[2]=gsl_stats_quantile_from_sorted_data (simMatrixStatStore, 1, statSamples*statSamples, 0.25)*2.0;
      quantVec[3]=gsl_stats_quantile_from_sorted_data (simMatrixStatStore, 1, statSamples*statSamples, 0.50)*2.0;
      quantVec[4]=gsl_stats_quantile_from_sorted_data (simMatrixStatStore, 1, statSamples*statSamples, 0.75)*2.0;
      quantVec[5]=gsl_stats_quantile_from_sorted_data (simMatrixStatStore, 1, statSamples*statSamples, 0.90)*2.0;
      quantVec[6]=gsl_stats_quantile_from_sorted_data (simMatrixStatStore, 1, statSamples*statSamples, 0.95)*2.0;
    }
    
    
    
    if(paramFeatureTypes[i].kc==TANCOMB||paramFeatureTypes[i].kc==TANCOMB2||paramFeatureTypes[i].kc==TANSCOMB) {
      for(long j=0L; j<paramGTypes.size(); j++) {
        if(paramFeatureTypes[i].sim==0L) {
          for(long k=0L; k<paramRBFTypes.size(); k++) {
            paramComb.push_back(SVMHyperParam(paramFeatureTypes[i], paramGTypes[j], paramRBFTypes[k]));
          }
        }
        else if(paramFeatureTypes[i].sim==1L) {
          for(long k=0L; k<7L; k++) {
            paramComb.push_back(SVMHyperParam(paramFeatureTypes[i], paramGTypes[j], quantVec[k]));
          }
        }
      }
    }
    else if(paramFeatureTypes[i].kc==LINCOMB) {
      for(long j=0L; j<paramGTypes.size(); j++) {
        if(paramFeatureTypes[i].sim==0L) {
          paramComb.push_back(SVMHyperParam(paramFeatureTypes[i], paramGTypes[j], paramRBFTypes[0]));
        }
        else if(paramFeatureTypes[i].sim==1L) {
          for(long k=0L; k<7L; k++) {
            paramComb.push_back(SVMHyperParam(paramFeatureTypes[i], paramGTypes[j], quantVec[k]));
          }
        }
      }
    }
    else if(paramFeatureTypes[i].kc==GAUSSCOMB) {
      for(long j=0L; j<paramGTypes.size(); j++) {
        if(paramFeatureTypes[i].sim==1L) {
          for(long k=0L; k<7L; k++) {
            paramComb.push_back(SVMHyperParam(paramFeatureTypes[i], paramGTypes[j], quantVec[k]));
          }
        }
      }
    }
  }
  std::sort(paramComb.begin(), paramComb.end(), paramSort);
  
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
  
  
  
  printf("Running svm with the following parameters:\n");
  printf("project:              %30s\n", projectName.c_str());
  printf("trainInfo:            %30s\n", trainInfo.c_str());
  printf("clusterInfo:          %30s\n", clusterInfo.c_str());
  printf("targetInfo:           %30s\n", targetInfo.c_str());
  printf("mode:                 %30d\n", mode);
  printf("predSavename:         %30s\n", predSavename.c_str());
  printf("epsilon:              %30.4lf\n", eps);
  printf("shrinking:            %30d\n", shrinking);
  printf("w1:                   %30.4lf\n", w1);
  printf("w2:                   %30.4lf\n", w2);
  printf("maxProc:              %30d\n", maxProc);
  
  string dbInformationFilename=predSavePathname+"settings.txt";
  ofstream dbInformationFile;
  dbInformationFile.open(dbInformationFilename.c_str());  
  dbInformationFile << "svm" << endl;
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
  dbInformationFile << "epsilon" << endl;
  dbInformationFile << eps << endl;
  dbInformationFile << "shrinking" << endl;
  dbInformationFile << shrinking << endl;
  dbInformationFile << "w1" << endl;
  dbInformationFile << w1 << endl;
  dbInformationFile << "w2" << endl;
  dbInformationFile << w2 << endl;
  dbInformationFile.close();
  
  
  
  struct svm_parameter param;
  //param.svm_type = svm_type;
  param.svm_type = 0;  //C
  param.kernel_type = PRECOMPUTED;
  param.degree = 3;
  param.gamma = 0;
  param.coef0 = 0;
  //param.nu = 0.5;
  param.cache_size = 100;
  //param.C = C;
  param.eps = eps;
  param.p = 0.1;
  param.shrinking = shrinking;
  param.probability = 0;
  param.nr_weight = 2;
  param.weight_label = NULL;
  param.weight_label = (int *)realloc(param.weight_label,sizeof(int)*param.nr_weight);
  param.weight = NULL;
  param.weight = (double *)realloc(param.weight,sizeof(double)*param.nr_weight);
  param.weight_label[0]=0;
  param.weight[0]=w1;
  param.weight_label[1]=1;
  param.weight[1]=w2;
  void (*print_func)(const char*) = &print_null;
  svm_set_print_string_function(print_func);  
  
  
  
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
            
            
            
            double* paramG=new double[paramGTypes.size()];
            paramG[0]=0.001;
            paramG[1]=0.01;
            paramG[2]=0.1;
            paramG[3]=1.0;
            paramG[4]=10.0;
            paramG[5]=100.0;
            paramG[6]=1000.0;
            
            
            
            double* aucStoreTrainP1=(double*)malloc(sizeof(double)*nrClusters*paramComb.size());
            double** aucStoreTrainP2=(double**)malloc(sizeof(double*)*nrClusters);
            for(long i=0L; i<nrClusters; i++) {
              aucStoreTrainP2[i]=&(aucStoreTrainP1[i*paramComb.size()]);
            }
            double** aucTrain=aucStoreTrainP2;
            
            double* aucStoreP1=(double*)malloc(sizeof(double)*nrClusters*paramComb.size());
            double** aucStoreP2=(double**)malloc(sizeof(double*)*nrClusters);
            for(long i=0L; i<nrClusters; i++) {
              aucStoreP2[i]=&(aucStoreP1[i*paramComb.size()]);
            }
            double** auc=aucStoreP2;
            
            
            
            for(long clusterIndHyper=0; clusterIndHyper<nrClusters; clusterIndHyper++)
            {
              {
                if(clusterIndEval==clusterIndHyper)
                  goto endLoopClusterHyper;
                printf("cE(%ld)/cE(%ld);cH(%ld)/cH(%ld)\n", clusterIndEval, nrClusters, clusterIndHyper, nrClusters);

                long clusterIndStartHyper=clusterIndVec[clusterIndHyper];
                long clusterSizeHyper=clusterSizeVec[clusterIndHyper];
                long clusterIndEndHyper=clusterIndStartHyper+clusterSizeHyper;
                
                bool* validHyperSamples=new bool[sampleNr];
                for(long i=0; i<sampleNr; i++)
                  validHyperSamples[i]=validEvalSamples[i];
                
                {
                  long removeInd=clusterIndStartHyper;
                  while(removeInd<clusterIndEndHyper) {
                    validHyperSamples[clusterSampleDB[removeInd].sampleNr]=false;
                    removeInd++;
                  }
                }
                
                bool* validHyper=new bool[targetSampleDBSize];
                for(long i=0; i<targetSampleDBSize; i++)
                  validHyper[i]=validEval[i];

                {
                  long targetSampleDBInd=0L;
                  long removeInd=clusterIndStartHyper;

                  while((targetSampleDBInd<targetSampleDBSize)&&(removeInd<clusterIndEndHyper)) {
                    if(targetSampleDB[indTargetSampleDB[targetSampleDBInd]].sampleNr<clusterSampleDB[removeInd].sampleNr) {
                      targetSampleDBInd++;
                    }
                    else if(targetSampleDB[indTargetSampleDB[targetSampleDBInd]].sampleNr>clusterSampleDB[removeInd].sampleNr) {
                      removeInd++;
                    }
                    else {
                      validHyper[indTargetSampleDB[targetSampleDBInd]]=false;
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
                  if(validHyper[targetSampleDBInd]&&validEval[targetSampleDBInd]) {
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
                  if(!validHyper[targetSampleDBInd]&&validEval[targetSampleDBInd]) {
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
                
                
                
                delete[] validHyper;
                delete[] validHyperSamples;
                
                
                
                if((nrPositiveTrain<1)||(nrNegativeTrain<1)||(nrPositivePred<1)||(nrNegativePred<1)) {
                  free(labelPred);
                  free(sampleNrPred);
                  free(indPred);
                  free(labelTrain);
                  free(sampleNrTrain);
                  free(indTrain);
                  
                  for(long paramInd=0L; paramInd<paramComb.size(); paramInd++)
                    aucTrain[clusterIndHyper][paramInd]=-1.0;
                  for(long paramInd=0L; paramInd<paramComb.size(); paramInd++)
                    auc[clusterIndHyper][paramInd]=-1.0;
                  
                  goto endLoopClusterHyper;
                }
                
                
                
                float* simMatrixTrainStore=(float*) malloc(sizeof(float)*nrTrain*nrTrain);
                float** simMatrixTrain=(float**) malloc(sizeof(float*)*nrTrain);
                float* simMatrixPredStore=(float*) malloc(sizeof(float)*nrTrain*nrPred);
                float** simMatrixPred=(float**) malloc(sizeof(float*)*nrTrain);
                float* simMatrixPredStoreH=(float*) malloc(sizeof(float)*nrTrain*nrPred);
                float** simMatrixPredH=(float**) malloc(sizeof(float*)*nrTrain);
                float* normMatrixPred=(float*) malloc(sizeof(float)*nrPred);
                
                for(long i=0L; i<nrTrain; i++) {
                  simMatrixTrain[i]=&(simMatrixTrainStore[i*nrTrain]);
                  simMatrixPred[i]=&(simMatrixPredStore[i*nrPred]);
                  simMatrixPredH[i]=&(simMatrixPredStoreH[i*nrPred]);
                }
                
                std::vector<struct svm_problem*> svmProblems;
                std::vector<struct svm_node*> svmNodes;
                std::vector<struct svm_problem*> svmProblemsUse;
                std::vector<struct svm_node*> svmNodesUse;
                struct svm_problem* problem=NULL;
                struct svm_node* predData=NULL;
                
                
                
                long oldFeatureParamId=-1;
                double oldRBF=-10.0;
                for(long paramInd=featureParamInd[fpi]; paramInd<featureParamInd[fpi+1L]; paramInd++) {

                  //fprintf(stderr, " paramInd:%ld, ", paramInd);
                  
                  FeatureKernelCollection featureParam=paramComb[paramInd].fkc;
                  long featureParamId=featureParam.id;
                  double rbf=paramComb[paramInd].rbf;
                  long svmGen=paramComb[paramInd].svmGen;
                  KernelComb kc=featureParam.kc;
                  long kn=featureParam.kn;
                  
                  if(oldFeatureParamId!=featureParamId) {
                    oldFeatureParamId=featureParamId;
                    oldRBF=-10.0;
                    std::fill(simMatrixTrainStore, simMatrixTrainStore+nrTrain*nrTrain, 0.0f);
                    std::fill(simMatrixPredStore, simMatrixPredStore+nrTrain*nrPred, 0.0f);
                    std::fill(simMatrixPredStoreH, simMatrixPredStoreH+nrTrain*nrPred, 0.0f);
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
                        for(long k=0L; k<nrTrain; k++) {
                          simMatrixTrain[j][k]=simMatrixTrain[j][k]+sim[indTrain[j]][indTrain[k]];
                        }
                        
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
                      for(long j=0L; j<nrPred; j++) {
                        normMatrixPred[j]=normMatrixPred[j]+sim[indPred[j]][indPred[j]];
                      }
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
                        for(long j=0L; j<=i; j++) {
                          if(indTrain[i]<indTrain[j]) {
                            if(simMatrixTrain[j][i]>=0.1) {
                              simMatrixTrain[i][j]=simMatrixTrain[i][j]/simMatrixTrain[j][i];
                              simMatrixTrain[j][i]=simMatrixTrain[i][j];
                            }
                            else {
                              simMatrixTrain[i][j]=0.0;
                              simMatrixTrain[j][i]=0.0;
                            }
                          }
                          else if(indTrain[i]>indTrain[j]) {
                            if(simMatrixTrain[i][j]>=0.1) {
                              simMatrixTrain[j][i]=simMatrixTrain[j][i]/simMatrixTrain[i][j];
                              simMatrixTrain[i][j]=simMatrixTrain[j][i];
                            }
                            else {
                              simMatrixTrain[j][i]=0.0;
                              simMatrixTrain[i][j]=0.0;
                            }
                          }
                          else if(indTrain[i]==indTrain[j]) {
                            simMatrixTrain[i][j]=1.0;
                            simMatrixTrain[j][i]=1.0;
                          }
                        }
                        
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
                        for(long j=0L; j<nrTrain; j++) {
                          if(i!=j) {
                            simMatrixTrain[i][j]=simMatrixTrain[i][j]/sqrt(simMatrixTrain[i][i]*simMatrixTrain[j][j]);
                          }
                        }
                        
                        for(long j=0L; j<nrPred; j++) {
                            simMatrixPred[i][j]=simMatrixPred[i][j]/(sqrt(simMatrixTrain[i][i])*normMatrixPred[j]);
                        }
                      }
                      
                      #ifdef multiproc
                      #pragma omp parallel for
                      #endif
                      for(long i=0L; i<nrTrain; i++) {
                        simMatrixTrain[i][i]=1.0;
                      }
                    }
                  }
                  
                  
                  
#ifdef dbgmat
                  std::stringstream ssTrainMat;
                  ssTrainMat << currentPredSavePathname << "dbgMatTrain_" << clusterIndEval << "_" << clusterIndHyper << "_" << featureParam.id;
                  string ssTrainMatStr=ssTrainMat.str();

                  std::stringstream ssPredMat;
                  ssPredMat << currentPredSavePathname << "dbgMatPred_" << clusterIndEval << "_" << clusterIndHyper << "_" << featureParam.id;
                  string ssPredMatStr=ssPredMat.str();
                  
                  std::stringstream ssTrainInd;
                  ssTrainInd << currentPredSavePathname << "dbgIndTrain_" << clusterIndEval << "_" << clusterIndHyper << "_" << featureParam.id;
                  string ssTrainIndStr=ssTrainInd.str();

                  std::stringstream ssPredInd;
                  ssPredInd << currentPredSavePathname << "dbgIndPred_" << clusterIndEval << "_" << clusterIndHyper << "_" << featureParam.id;
                  string ssPredIndStr=ssPredInd.str();
                  
                  std::stringstream ssTrainLabel;
                  ssTrainLabel << currentPredSavePathname << "dbgLabelTrain_" << clusterIndEval << "_" << clusterIndHyper << "_" << featureParam.id;
                  string ssTrainLabelStr=ssTrainLabel.str();

                  std::stringstream ssPredLabel;
                  ssPredLabel << currentPredSavePathname << "dbgLabelPred_" << clusterIndEval << "_" << clusterIndHyper << "_" << featureParam.id;
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
                      for(long j=0L; j<nrTrain; j++) {
                        fprintf(matFileTrain, "%lf", simMatrixTrain[i][j]);
                        if(j==nrTrain-1)
                          fprintf(matFileTrain, "\n");
                        else
                          fprintf(matFileTrain, " ");
                      }
                      
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
                  
                  
                  
                  if(fabs(oldRBF-rbf)/fabs(oldRBF)>0.0001) {
                    problem=(struct svm_problem*)malloc(sizeof(struct svm_problem));
                    problem->l = nrTrain;
                    problem->y = Malloc(double, problem->l);
                    problem->x = Malloc(struct svm_node, problem->l);
                    double* problemStorage=(double*)malloc(sizeof(double)*(nrTrain+1L)*nrTrain);
                    for(long i=0L; i<nrTrain; i++) {
                      (problem->y)[i]=labelTrain[i];
                      ((problem->x)+i)->dim=nrTrain+1L;
                      ((problem->x)+i)->values=&(problemStorage[(nrTrain+1L)*i]);
                      ((problem->x)+i)->values[0]=(double)(i+1L);
                    }
                    
                    predData=(struct svm_node*)malloc(sizeof(struct svm_node)*nrPred);
                    double* predStorage=(double*)malloc(sizeof(double)*(nrTrain+1L)*nrPred);
                    for(long i=0L; i<nrPred; i++) {
                      (predData+i)->dim=(nrTrain+1L);
                      (predData+i)->values=&(predStorage[(nrTrain+1L)*i]);
                      (predData+i)->values[0]=0;
                    }
                    
                    svmProblems.push_back(problem);
                    svmNodes.push_back(predData);
                    
                    
                    
                    #ifdef multiproc
                    #pragma omp parallel for
                    #endif                 
                    for(long i=0L; i<nrTrain; i++)
                    {
                      if(rbf<0) {
                        for(long j=0L; j<nrTrain; j++) ((problem->x)+j)->values[i+1L]=simMatrixTrain[i][j];
                        for(long j=0L; j<nrPred; j++) (predData+j)->values[i+1L]=simMatrixPred[i][j];
                      }
                      else {
                        for(long j=0L; j<nrTrain; j++) ((problem->x)+j)->values[i+1L]=exp(simMatrixTrain[i][j]/rbf);
                        for(long j=0L; j<nrPred; j++) (predData+j)->values[i+1L]=exp(simMatrixPred[i][j]/rbf);
                      }
                    }
                    oldRBF=rbf;
                  }
                  
                  svmProblemsUse.push_back(problem);
                  svmNodesUse.push_back(predData);
                }
                
                
                
                #ifdef multiproc
                omp_set_nested(1);
                int nrThreadsL1=ceil(((double)maxProc)/((double)(featureParamInd[fpi+1L]-featureParamInd[fpi])));
                int nrThreadsL2=maxProc/nrThreadsL1;
                omp_set_num_threads(nrThreadsL1);
                #endif
                
                #ifdef multiproc
                #pragma omp parallel for
                #endif
                for(long paramInd=featureParamInd[fpi]; paramInd<featureParamInd[fpi+1L]; paramInd++) {
                  struct svm_problem* problem=svmProblemsUse[paramInd-featureParamInd[fpi]];
                  struct svm_node* predData=svmNodesUse[paramInd-featureParamInd[fpi]];
                  
                  FeatureKernelCollection featureParam=paramComb[paramInd].fkc;
                  long featureParamId=featureParam.id;
                  double rbf=paramComb[paramInd].rbf;
                  long svmGen=paramComb[paramInd].svmGen;
                  KernelComb kc=featureParam.kc;
                  long kn=featureParam.kn;

                  struct svm_parameter myparam=param;
                  myparam.C=paramG[svmGen];
                  struct svm_model *model;

                  std::stringstream mys;
                  mys << featureParam.id << ":" << svmGen << ":" << rbf << "\n\n";
                  string errstr=mys.str();            
                  char* myerr=(char*)malloc(sizeof(char)*errstr.size());
                  strncpy(myerr, errstr.c_str(), errstr.size());
                  myerr[errstr.size()-1]=0;
                  
                  #ifdef multiproc
                  omp_set_num_threads(nrThreadsL2);
                  #endif
                  
                  model=svm_train(problem, &myparam, myerr);
                  free(myerr);
                  
                  
                  
                  double* predTrainVec=(double*)malloc(sizeof(double)*nrTrain);
                  for(long i=0L; i<nrTrain; i++) {
                    double prob=0.0;
                    double predict_label = svm_predict_values(model, ((problem->x)+i), &prob);
                    if((predict_label==0L)&&(prob>0))
                      prob=prob*(-1);
                    else if((predict_label==1L)&&(prob<0))
                      prob=prob*(-1);                
                    predTrainVec[i]=prob;
                  }
                  aucTrain[clusterIndHyper][paramInd]=compAUC(labelTrain, predTrainVec, nrTrain);
                  free(predTrainVec);
                  
                  
                  
                  double* predVec=(double*)malloc(sizeof(double)*nrPred);                  
                  for(long i=0L; i<nrPred; i++) {
                    double prob=0.0;
                    double predict_label = svm_predict_values(model, (predData+i), &prob);
                    if((predict_label==0L)&&(prob>0))
                      prob=prob*(-1);
                    else if((predict_label==1L)&&(prob<0))
                      prob=prob*(-1);
                    predVec[i]=prob;
                  }
                  auc[clusterIndHyper][paramInd]=compAUC(labelPred, predVec, nrPred);
                  free(predVec);
                  
                  svm_free_and_destroy_model(&model);
                }
                
                #ifdef multiproc
                omp_set_nested(0);
                #endif
                
                
                
                for(long i=0; i<svmProblems.size(); i++) {
                  free((svmNodes[i])->values);
                  free(svmNodes[i]);
                  free(((svmProblems[i])->x)->values);
                  free((svmProblems[i])->x);
                  free((svmProblems[i])->y);
                  free(svmProblems[i]);
                }
                
                
                
                free(normMatrixPred);
                free(simMatrixPredH);
                free(simMatrixPredStoreH);
                free(simMatrixPred);
                free(simMatrixPredStore);
                free(simMatrixTrain);
                free(simMatrixTrainStore);          
            
                free(labelPred);
                free(sampleNrPred);
                free(indPred);
                free(labelTrain);
                free(sampleNrTrain);
                free(indTrain);
              }
              endLoopClusterHyper: ;
            }
            
            
            
            delete[] validEval;
            delete[] validEvalSamples;
            
            
            
            long aucSelIndex;
            double maxValue=-0.5;
            long maxIndex=-1L;
            for(long i=featureParamInd[fpi]; i<featureParamInd[fpi+1L]; i++) {
              long count=0L;
              double sum=0.0;
              for(long j=0L; j<nrClusters; j++) {
                if(j!=clusterIndEval) {
                  if(auc[j][i]>=0.0) {
                    sum=sum+auc[j][i];
                    count++;
                  }
                  else {
                    sum=sum+0.5;
                    count++;
                  }
                }
              }
              double mean;
              if(count>0) 
                mean=sum/((double)count);
              else
                mean=-1;
              if(mean>maxValue) {
                maxValue=mean;
                maxIndex=i;
              }
            }
            aucSelIndex=maxIndex;
            //fprintf(stderr, "%ld, ", maxIndex);
            
            
            
            string outFilenameHyperTrainAUC=currentPredSavePathname+"clustHyperTrainAUC"+ss.str()+".out";
            char* outFilenameHyperTrainAUCBuffer=(char*)malloc(1024*1024);
            FILE* outFileHyperTrainAUC=fopen(outFilenameHyperTrainAUC.c_str(), "w");
            setbuffer(outFileHyperTrainAUC, outFilenameHyperTrainAUCBuffer, 1024*1024);
            for(long k=featureParamInd[fpi]; k<featureParamInd[fpi+1L]; k++) {
              fprintf(outFileHyperTrainAUC, "featureParam=%ld/svmGen=%5.10lf/rbf=%5.10lf", paramComb[k].fkc.id, paramG[paramComb[k].svmGen], paramComb[k].rbf);
              for(long i=0L; i<nrClusters; i++) {
                if(i!=clusterIndEval) fprintf(outFileHyperTrainAUC, ",%5.10lf", aucTrain[i][k]);
                else fprintf(outFileHyperTrainAUC, ",NA");
              }
              fprintf(outFileHyperTrainAUC, "\n");
            }
            fclose(outFileHyperTrainAUC);
            free(outFilenameHyperTrainAUCBuffer);
            
            string outFilenameHyperPredAUC=currentPredSavePathname+"clustHyperPredAUC"+ss.str()+".out";
            char* outFilenameHyperPredAUCBuffer=(char*)malloc(1024*1024);
            FILE* outFileHyperPredAUC=fopen(outFilenameHyperPredAUC.c_str(), "w");
            setbuffer(outFileHyperPredAUC, outFilenameHyperPredAUCBuffer, 1024*1024);
            for(long k=featureParamInd[fpi]; k<featureParamInd[fpi+1L]; k++) {
              fprintf(outFileHyperPredAUC, "featureParam=%ld/svmGen=%5.10lf/rbf=%5.10lf", paramComb[k].fkc.id, paramG[paramComb[k].svmGen], paramComb[k].rbf);
              for(long i=0L; i<nrClusters; i++) {
                if(i!=clusterIndEval) fprintf(outFileHyperPredAUC, ",%5.10lf", auc[i][k]);
                else fprintf(outFileHyperPredAUC, ",NA");
              }
              fprintf(outFileHyperPredAUC, "\n");
            }
            fprintf(outFileHyperPredAUC, "selected: %ld\n", aucSelIndex);
            fprintf(outFileHyperPredAUC, "featureParam=%ld/svmGen=%5.10lf/rbf=%5.10lf", paramComb[aucSelIndex].fkc.id, paramG[paramComb[aucSelIndex].svmGen], paramComb[aucSelIndex].rbf);
            for(long i=0L; i<nrClusters; i++) {
              if(i!=clusterIndEval) fprintf(outFileHyperPredAUC, ",%5.10lf", auc[i][aucSelIndex]);
              else fprintf(outFileHyperPredAUC, ",NA");
            }
            fprintf(outFileHyperPredAUC, "\n");
            fclose(outFileHyperPredAUC);
            free(outFilenameHyperPredAUCBuffer);
            
            
            
            free(aucStoreP2);
            free(aucStoreP1);
            free(aucStoreTrainP2);
            free(aucStoreTrainP1);
            
            
            
            float* simMatrixTrainStore=(float*) malloc(sizeof(float)*nrTrain*nrTrain);
            float** simMatrixTrain=(float**) malloc(sizeof(float*)*nrTrain);
            float* simMatrixPredStore=(float*) malloc(sizeof(float)*nrTrain*nrPred);
            float** simMatrixPred=(float**) malloc(sizeof(float*)*nrTrain);
            float* simMatrixPredStoreH=(float*) malloc(sizeof(float)*nrTrain*nrPred);
            float** simMatrixPredH=(float**) malloc(sizeof(float*)*nrTrain);
            float* normMatrixPred=(float*) malloc(sizeof(float)*nrPred);
            
            for(long i=0L; i<nrTrain; i++) {
              simMatrixTrain[i]=&(simMatrixTrainStore[i*nrTrain]);
              simMatrixPred[i]=&(simMatrixPredStore[i*nrPred]);
              simMatrixPredH[i]=&(simMatrixPredStoreH[i*nrPred]);
            }
            
            
            
            FeatureKernelCollection featureParam;
            long featureParamId;
            double rbf;
            long svmGen;
            KernelComb kc;
            long kn=featureParam.kn;
            
            if(aucSelIndex==-1) {
              featureParam=defaultFeature;
              featureParamId=featureParam.id;
              rbf=defaultRBF;
              svmGen=defaultG;
              kc=defaultFeature.kc;
              kn=defaultFeature.kn;
            }
            else {
              featureParam=paramComb[aucSelIndex].fkc;
              featureParamId=featureParam.id;
              rbf=paramComb[aucSelIndex].rbf;
              svmGen=paramComb[aucSelIndex].svmGen;
              kc=paramComb[aucSelIndex].fkc.kc;
              kn=paramComb[aucSelIndex].fkc.kn;
            }
            
            
            
            std::fill(simMatrixTrainStore, simMatrixTrainStore+nrTrain*nrTrain, 0.0f);
            std::fill(simMatrixPredStore, simMatrixPredStore+nrTrain*nrPred, 0.0f);
            std::fill(simMatrixPredStoreH, simMatrixPredStoreH+nrTrain*nrPred, 0.0f);
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
                for(long k=0L; k<nrTrain; k++) {
                  simMatrixTrain[j][k]=simMatrixTrain[j][k]+sim[indTrain[j]][indTrain[k]];
                }
                
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
              for(long j=0L; j<nrPred; j++) {
                normMatrixPred[j]=normMatrixPred[j]+sim[indPred[j]][indPred[j]];
              }
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
                for(long j=0L; j<=i; j++) {
                  if(indTrain[i]<indTrain[j]) {
                    if(simMatrixTrain[j][i]>=0.1) {
                      simMatrixTrain[i][j]=simMatrixTrain[i][j]/simMatrixTrain[j][i];
                      simMatrixTrain[j][i]=simMatrixTrain[i][j];
                    }
                    else {
                      simMatrixTrain[i][j]=0.0;
                      simMatrixTrain[j][i]=0.0;
                    }
                  }
                  else if(indTrain[i]>indTrain[j]) {
                    if(simMatrixTrain[i][j]>=0.1) {
                      simMatrixTrain[j][i]=simMatrixTrain[j][i]/simMatrixTrain[i][j];
                      simMatrixTrain[i][j]=simMatrixTrain[j][i];
                    }
                    else {
                      simMatrixTrain[j][i]=0.0;
                      simMatrixTrain[i][j]=0.0;
                    }
                  }
                  else if(indTrain[i]==indTrain[j]) {
                    simMatrixTrain[i][j]=1.0;
                    simMatrixTrain[j][i]=1.0;
                  }
                }
                
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
                for(long j=0L; j<nrTrain; j++) {
                  if(i!=j) {
                    simMatrixTrain[i][j]=simMatrixTrain[i][j]/sqrt(simMatrixTrain[i][i]*simMatrixTrain[j][j]);
                  }
                }
                
                for(long j=0L; j<nrPred; j++) {
                    simMatrixPred[i][j]=simMatrixPred[i][j]/(sqrt(simMatrixTrain[i][i])*normMatrixPred[j]);
                }
              }
              
              #ifdef multiproc
              #pragma omp parallel for
              #endif
              for(long i=0L; i<nrTrain; i++) {
                simMatrixTrain[i][i]=1.0;
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
                for(long j=0L; j<nrTrain; j++) {
                  fprintf(matFileTrain, "%lf", simMatrixTrain[i][j]);
                  if(j==nrTrain-1)
                    fprintf(matFileTrain, "\n");
                  else
                    fprintf(matFileTrain, " ");
                }
                
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
            
            
            
            struct svm_problem problem;
            problem.l = nrTrain;
            problem.y = Malloc(double, problem.l);
            problem.x = Malloc(struct svm_node, problem.l);
            double* problemStorage=(double*)malloc(sizeof(double)*(nrTrain+1L)*nrTrain);
            for(long i=0L; i<nrTrain; i++) {
              problem.y[i]=labelTrain[i];
              (problem.x+i)->dim=nrTrain+1L;
              (problem.x+i)->values=&(problemStorage[(nrTrain+1L)*i]);
              (problem.x+i)->values[0]=(double)(i+1L);
            }
            
            struct svm_node *predData=(struct svm_node*)malloc(sizeof(struct svm_node)*nrPred);
            double* predStorage=(double*)malloc(sizeof(double)*(nrTrain+1L)*nrPred);
            for(long i=0L; i<nrPred; i++) {
              (predData+i)->dim=(nrTrain+1L);
              (predData+i)->values=&(predStorage[(nrTrain+1L)*i]);
              (predData+i)->values[0]=0;
            }
            
            #ifdef multiproc
            #pragma omp parallel for
            #endif
            for(long i=0L; i<nrTrain; i++)
            {
              if(rbf<0) {
                for(long j=0L; j<nrTrain; j++) (problem.x+j)->values[i+1L]=simMatrixTrain[i][j];
                for(long j=0L; j<nrPred; j++) (predData+j)->values[i+1L]=simMatrixPred[i][j];
              }
              else {
                for(long j=0L; j<nrTrain; j++) (problem.x+j)->values[i+1L]=exp(simMatrixTrain[i][j]/rbf);
                for(long j=0L; j<nrPred; j++) (predData+j)->values[i+1L]=exp(simMatrixPred[i][j]/rbf);
              }
            }
            
            struct svm_parameter myparam=param;
            myparam.C=paramG[svmGen];
            struct svm_model *model;
            
            std::stringstream mys;
            mys << featureParam.id << ":" << svmGen << ":" << rbf << "\n\n";
            string errstr=mys.str();            
            char* myerr=(char*)malloc(sizeof(char)*errstr.size());
            strncpy(myerr, errstr.c_str(), errstr.size());
            myerr[errstr.size()-1]=0;
            
            model=svm_train(&problem, &myparam, myerr);
            free(myerr);
            
            
            
            string outFilenameModel=currentPredSavePathname+"model"+ss.str()+".out";
            char* outFilenameModelBuffer=(char*)malloc(1024*1024);
            FILE* outFileModel=fopen(outFilenameModel.c_str(), "wb");
            setbuffer(outFileModel, outFilenameModelBuffer, 1024*1024);
            fwrite(&(aucSelIndex), sizeof(long), 1, outFileModel);
            fwrite(&(model->l), sizeof(int), 1, outFileModel);
            fwrite(&(model->rho[0]), sizeof(double), 1, outFileModel);
            fwrite(model->label, sizeof(int), 2, outFileModel);
            fwrite(model->nSV, sizeof(int), 2, outFileModel);
            fwrite(model->sv_indices, sizeof(int), model->l, outFileModel);
            fwrite(model->sv_coef[0], sizeof(double), model->l, outFileModel);
            fclose(outFileModel);
            free(outFilenameModelBuffer);
            
            
            
            double* predTrainVec=(double*)malloc(sizeof(double)*nrTrain);
            for(long i=0L; i<nrTrain; i++) {
              fprintf(outFileSampleTrain, "%ld\n", sampleNrTrain[i]);
              fprintf(outFileTrueTrain, "%ld\n", labelTrain[i]);

              double prob=0.0;
              double predict_label = svm_predict_values(model, ((problem.x)+i), &prob);
              if((predict_label==0L)&&(prob>0))
                prob=prob*(-1);
              else if((predict_label==1L)&&(prob<0))
                prob=prob*(-1);                
              predTrainVec[i]=prob;
              fprintf(outFileProbTrain, "%.10lf\n", prob);
              double cls=predict_label;
              fprintf(outFileClassTrain, "%.10lf\n", cls);
            }
            aucEvalTrain[clusterIndEval]=compAUC(labelTrain, predTrainVec, nrTrain);
            free(predTrainVec);
            
            double* predVec=(double*)malloc(sizeof(double)*nrPred);
            for(long i=0L; i<nrPred; i++) {
              fprintf(outFileSample, "%ld\n", sampleNrPred[i]);
              fprintf(outFileTrue, "%ld\n", labelPred[i]);
              
              double prob=0.0;
              double predict_label = svm_predict_values(model, (predData+i), &prob);
              if((predict_label==0L)&&(prob>0))
                prob=prob*(-1);
              else if((predict_label==1L)&&(prob<0))
                prob=prob*(-1);
              predVec[i]=prob;
              fprintf(outFileProb, "%.10lf\n", prob);
              double cls=predict_label;
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
            
            
            
            svm_free_and_destroy_model(&model);
            
            
            
            free(predStorage);
            free(predData);
            free(problemStorage);
            free(problem.x);
            free(problem.y);
            
            
            
            free(normMatrixPred);
            free(simMatrixPredH);
            free(simMatrixPredStoreH);
            free(simMatrixPred);
            free(simMatrixPredStore);
            free(simMatrixTrain);
            free(simMatrixTrainStore);
            
            free(labelPred);
            free(sampleNrPred);
            free(indPred);
            free(labelTrain);
            free(sampleNrTrain);
            free(indTrain);
            
            
            
            delete[] paramG;
          }
          endLoopClusterEval: ;
        }
        
        string outFilenameEvalTrainAUC=currentPredSavePathname+"clustEvalTrainAUC.out";
        string outFilenameEvalPredAUC=currentPredSavePathname+"clustEvalPredAUC.out";
        
        FILE* outFileEvalTrainAUC=fopen(outFilenameEvalTrainAUC.c_str(), "w");
        for(long i=0L; i<(nrClusters-1L); i++) {
          fprintf(outFileEvalTrainAUC, "%5.10lf,", aucEvalTrain[i]);
        }
        fprintf(outFileEvalTrainAUC, "%5.10lf\n", aucEvalTrain[nrClusters-1L]);
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
  
  
  
  free(param.weight);
  free(param.weight_label);

  printf("svmHyper terminated successfully!\n");

  return 0;
}
