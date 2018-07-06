/*
Copyright (C) 2018 Andreas Mayr
Licensed under GNU General Public License v3.0 (see http://www.bioinf.jku.at/research/lsc/LICENSE and https://github.com/ml-jku/lsc/blob/master/LICENSE)
*/



#include <sys/stat.h>
#include <sys/types.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <list>
#include <vector>
#include <algorithm>
#include "compoundData.h"

using namespace std;



int main(int argc, char** argv) {
  if(sizeof(long)<8) {
    printf("This program is optimized for machines with at least 8 byte longs!  The program will terminate!\n");
    return -1;     
  }
  string progName(argv[0]);
  string basePathname(getenv("HOME")); basePathname=basePathname+"/mydata/trgpred/";
  
  
  
  struct stat sb;
  int statRet;
  
  if(argc!=4) {
    fprintf(stderr, "Usage: convertBinary project rawDataFolder fpVersion\n");
    exit(-1); 
  }
  
  string projectName(argv[1]);
  string projectPathname=basePathname+projectName+"/";
  string chemPathname=projectPathname+"chemFeatures/";
  string clusterPathname=chemPathname+"cl/";
  string dchemPathname=chemPathname+"d/";
  string schemPathname=chemPathname+"s/";
  string trainPathname=projectPathname+"train/";
  string runPathname=projectPathname+"run/";
  string sampleIdFilename=chemPathname+"SampleIdTable.txt";
  
  string rawDataFoldername(argv[2]);
  string rawDataPathname(getenv("HOME"));
  rawDataPathname=rawDataPathname+"/mydata/raw/"+rawDataFoldername+"/";
  
  string fpVersionName(argv[3]);
  
  string jCMapperFilename=rawDataPathname+fpVersionName+".fpf";
  statRet=(stat(jCMapperFilename.c_str(), &sb)==-1);
  if(statRet==-1) {
    fprintf(stderr, "The files '%s' does either not exist or is not complete!\n", jCMapperFilename.c_str());
    exit(-1);
  }
  
  string sfeatureDBPathname(schemPathname); sfeatureDBPathname=sfeatureDBPathname+fpVersionName+"/";
  if(mkdir(sfeatureDBPathname.c_str(), S_IRWXU)==-1) {
    fprintf(stderr, "The directory '%s' either already exists or it is an invalid directory!\n", sfeatureDBPathname.c_str());
    exit(-1);
  }
  string intToExtMappingFilename=sfeatureDBPathname+"IntToExtMappingTable.bin";
  string sampleFilename=sfeatureDBPathname+"fpSampleTable.bin";
  string featureIntFilename=sfeatureDBPathname+"fpFeatureTableInt.bin";
  string featureExtFilename=sfeatureDBPathname+"fpFeatureTableExt.bin";
  string featureCountFilename=sfeatureDBPathname+"fpFeatureCountTableD.bin";
  string dataTypeInfoFilename=sfeatureDBPathname+"dt.txt";
  
  printf("Creating binary database with the following parameters:\n");
  printf("projectName:   %30s\n", projectName.c_str());
  printf("rawDataFolder: %30s\n", rawDataFoldername.c_str());
  printf("fpVersion:     %30s\n", fpVersionName.c_str());  
  
  
  
  ofstream dataTypeInfoFile;
  dataTypeInfoFile.open(dataTypeInfoFilename.c_str());
  dataTypeInfoFile << sizeof(long) << endl;
  dataTypeInfoFile.close();
  
  long sampleNr=0L;
  map<string, long> sampleIdToSampleIndex;
  std::list<string> validIdsList;
  ifstream sampleIdFile(sampleIdFilename.c_str(), ifstream::in);
  string sampleId;
  while(getline(sampleIdFile, sampleId)) {
    sampleIdToSampleIndex.insert(pair<string, long>(sampleId, sampleNr));
    validIdsList.push_back(sampleId);
    sampleNr++;
  }
  sampleIdFile.close();
  validIdsList.sort();
  std::vector<string> validIdsVec(validIdsList.begin(), validIdsList.end());
  fprintf(stderr, "sampleNr: %ld\n", sampleNr);
  
  long* sampleMap=(long*)malloc(sizeof(long)*sampleNr);
  for(long i=0L; i<sampleNr; i++)
    sampleMap[i]=-1L;
  
  long i;
  i=0L;

  std::list<long> jCMapperSamples;
  std::list<long> jCMapperFeatures;
  std::list<double> jCMapperFeatureCounts;
  std::map<long, long> featureList;
  string inputLine;

  std::ifstream jCMapperFile(jCMapperFilename.c_str());
  long index=0L;
  while(std::getline(jCMapperFile, inputLine)) {
    std::stringstream ss;
    ss << inputLine;
    string compoundId;
    ss >> compoundId;
    
    bool valid=false;
    if(std::binary_search(validIdsVec.begin(), validIdsVec.end(), compoundId))
      valid=true;
    else
      valid=false;
    
    if(valid) {
      jCMapperSamples.push_back(index);
      sampleMap[sampleIdToSampleIndex[compoundId]]=i;
      i++;
    }
    long nr1;
    double nr2;
    while(!ss.eof()) {
      ss >> nr1;
      ss.ignore(1, ':');
      ss >> nr2;
      if(valid) {
        jCMapperFeatures.push_back(nr1);
        jCMapperFeatureCounts.push_back(nr2);
        featureList.insert(pair<long, long>(nr1, 1L));
        index=index+1L;
      }
    }
  }
  jCMapperFile.close();

  long *samples=(long*)malloc(sizeof(long)*(((long)jCMapperSamples.size())+1));
  if(samples==NULL) {
    fprintf(stderr, "Too few main memory. Program will terminate!\n");
    exit(-1);
  }
  i=0;
  for(std::list<long>::iterator it=jCMapperSamples.begin(); it != jCMapperSamples.end(); ++it) {
    samples[i]=*it;  
    i++;
  }
  samples[((long)jCMapperSamples.size())]=((long)jCMapperFeatures.size());
  
  long *features=(long*)malloc(sizeof(long)*((long)jCMapperFeatures.size()));
  if(features==NULL) {
    fprintf(stderr, "Too few main memory. Program will terminate!\n");
    exit(-1);
  }
  i=0;
  for(std::list<long>::iterator it=jCMapperFeatures.begin(); it != jCMapperFeatures.end(); ++it) {
    features[i]=*it;  
    i++;
  }
  
  double *featureCounts=(double*)malloc(sizeof(double)*((long)jCMapperFeatures.size()));
  if(featureCounts==NULL) {
    fprintf(stderr, "Too few main memory. Program will terminate!\n");
    exit(-1);
  }
  i=0;
  for(std::list<double>::iterator it=jCMapperFeatureCounts.begin(); it != jCMapperFeatureCounts.end(); ++it) {
    featureCounts[i]=*it;  
    i++;
  }
  
  long featureNr=0L;
  long *samplesCopy=(long*)malloc(sizeof(long)*(sampleNr+1));
  long *featuresCopy=(long*)malloc(sizeof(long)*((long)jCMapperFeatures.size()));
  double *featureCountsCopy=(double*)malloc(sizeof(double)*((long)jCMapperFeatures.size()));
  for(i=0L; i<sampleNr; i++) {
    samplesCopy[i]=featureNr;
    if(sampleMap[i]!=(-1L)) {
      long samplesBegin=samples[sampleMap[i]];
      long samplesEnd=samples[sampleMap[i]+1L];
      for(long j=samplesBegin; j<samplesEnd; j++) {
        featuresCopy[featureNr]=features[j];
        featureCountsCopy[featureNr]=featureCounts[j];
        featureNr++;
      }
    }
  }
  samplesCopy[sampleNr]=featureNr;
  free(sampleMap);
  free(samples);
  free(features);
  free(featureCounts);
  samples=samplesCopy;
  features=featuresCopy;
  featureCounts=featureCountsCopy;
  
  long* internalFeatNrToOrigFeatNr=(long*)malloc(sizeof(long)*featureList.size());
  if(internalFeatNrToOrigFeatNr==NULL) {
    fprintf(stderr, "Too few main memory. Program will terminate!\n");
    exit(-1);    
  }
  long counter=0L;
  for(map<long, long>::iterator it=featureList.begin(); it!=featureList.end(); it++) {
    (*it).second=counter;
    internalFeatNrToOrigFeatNr[counter]=(*it).first;
    counter++;
  }
  
  FILE* intToExtMappingFile=fopen(intToExtMappingFilename.c_str(), "wb");
  fwrite(internalFeatNrToOrigFeatNr, sizeof(long), counter, intToExtMappingFile);
  fclose(intToExtMappingFile);
  free(internalFeatNrToOrigFeatNr);
  
  FeatExtIntT* featuresExtInt=(FeatExtIntT*)malloc(sizeof(FeatExtIntT)*featureNr);
  if(featuresExtInt==NULL) {
    fprintf(stderr, "Too few main memory. Program will terminate!\n");
    exit(-1);    
  }
  for(long i=0L; i<featureNr; i++) {
    featuresExtInt[i].extId=features[i];
    featuresExtInt[i].intId=featureList[features[i]];
    featuresExtInt[i].count=featureCounts[i];
  }
  free(features);
  for(long i=0L; i<sampleNr; i++) {
    qsort(&featuresExtInt[samples[i]], samples[i+1]-samples[i], sizeof(FeatExtIntT), compareFeatExtInt);
  }
  
  FILE* sampleFile=fopen(sampleFilename.c_str(), "wb");
  fwrite(samples, sizeof(long), sampleNr, sampleFile);
  fclose(sampleFile);
  free(samples);
  
  long *featuresExt=(long*)malloc(sizeof(long)*featureNr);
  long *featuresInt=(long*)malloc(sizeof(long)*featureNr);
  if(featuresExt==NULL||featuresInt==NULL) {
    fprintf(stderr, "Too few main memory. Program will terminate!\n");
    exit(-1);
  }
  for(long i=0L; i<featureNr; i++) {
    featuresExt[i]=featuresExtInt[i].extId;
    featuresInt[i]=featuresExtInt[i].intId;
    featureCounts[i]=featuresExtInt[i].count;
  }
  free(featuresExtInt);
  
  FILE* featureExtFile=fopen(featureExtFilename.c_str(), "wb");
  fwrite(featuresExt, sizeof(long), featureNr, featureExtFile);
  fclose(featureExtFile);
  free(featuresExt);
  
  FILE* featureIntFile=fopen(featureIntFilename.c_str(), "wb");
  fwrite(featuresInt, sizeof(long), featureNr, featureIntFile);
  fclose(featureIntFile);
  free(featuresInt);
  
  FILE* featureCountFile=fopen(featureCountFilename.c_str(), "wb");
  fwrite(featureCounts, sizeof(double), featureNr, featureCountFile);
  fclose(featureCountFile);
  free(featureCounts);
  
  printf(progName.c_str());
  printf(" terminated successfully!\n");
  
  return 0;
}
