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



typedef struct FeatExtInt {
  long extId;
  long intId;
  double count;
} FeatExtIntT;

int compareFeatExtInt(const void * a, const void * b) {
  if(( (*(FeatExtIntT*)a).intId - (*(FeatExtIntT*)b).intId )<0L)
    return -1;
  if(( (*(FeatExtIntT*)a).intId - (*(FeatExtIntT*)b).intId )>0L)
    return 1;
  return 0;
}

enum KernelType { TAN, TANS, TAN2, LIN, GAUSS};
enum KernelComb { TANCOMB, TANSCOMB, TANCOMB2, LINCOMB, GAUSSCOMB};



class SampleData {
  public:
    SampleData(long id, const std::string& sampleIdFilename) {
      this->id=id;
      this->sampleIdFilename=sampleIdFilename;
      load();
    }
    
    void load() {
      FILE* sampleIdFile=fopen(sampleIdFilename.c_str(), "rb");
      fseek(sampleIdFile, 0, SEEK_END);
      long sampleIdFileSize=ftell(sampleIdFile);
      rewind(sampleIdFile);
      char* sampleIdTxt=(char*)malloc(sampleIdFileSize+sizeof(char));
      if(sampleIdTxt==NULL) {
        fprintf(stderr, "Too few main memory. Program will terminate! - error %d\n", __LINE__);
        exit(-1);
      }
      fread(sampleIdTxt, 1, sampleIdFileSize, sampleIdFile);
      fclose(sampleIdFile);
      long nrChars=sampleIdFileSize/sizeof(char);
      sampleNr=0L;
      for(long i=0L; i<nrChars; i++)
        if(sampleIdTxt[i]=='\n') 
          sampleNr++;
      free(sampleIdTxt);
      
      sampleIndexToSampleId=new std::string[sampleNr];
      
      std::ifstream sampleIdFileStream(sampleIdFilename.c_str(), std::ifstream::in);
      std::string sampleId;
      for(long i=0L; i<sampleNr; i++) {
        getline(sampleIdFileStream, sampleId);
        sampleIndexToSampleId[i]=sampleId;
        sampleIdToSampleIndex.insert(std::pair<std::string, long>(sampleId, i));
      }
      sampleIdFileStream.close();    
    }
    
    void unload() {
      delete[] sampleIndexToSampleId;
    }
    
    ~SampleData() {
       unload();
    }
    
    long id;
    long sampleNr;
    std::string sampleIdFilename;
    std::string* sampleIndexToSampleId;
    std::map<std::string, long> sampleIdToSampleIndex;
};



class DenseFeatureData {
  public:
    DenseFeatureData(long ind, const std::string& dchemPathname, const std::string& dFeatureType) {
      this->id=ind;
      struct stat sb;
      int statRet;
      std::string dataTypeInfoFilename;
      std::ifstream dataTypeInfoFile;
      std::string dataType;
      std::string propertyVersionName(dFeatureType);
      
      std::string dfeatureDBPathname(dchemPathname); dfeatureDBPathname=dfeatureDBPathname+propertyVersionName+"/";
      propertyFilename=dfeatureDBPathname+"properties.bin";
      propertyIdFilename=dfeatureDBPathname+"PropertyIdTable.txt";
      dataTypeInfoFilename=dfeatureDBPathname+"dt.txt";
      statRet=(stat(propertyFilename.c_str(), &sb)!=-1);
      statRet=statRet||(stat(propertyIdFilename.c_str(), &sb)!=-1);  
      if(statRet==-1) {
        fprintf(stderr, "The database '%s' does either not exist or is not complete! - error %d\n", dfeatureDBPathname.c_str(), __LINE__);
        exit(-1);
      }
      dataTypeInfoFile.open(dataTypeInfoFilename.c_str(), std::ifstream::in);
      getline(dataTypeInfoFile, dataType);
      dataTypeInfoFile.close();
      if(sizeof(long)!=atoi(dataType.c_str())) {
        fprintf(stderr, "The system, which was used to create the database is not compatible with the current system! - error %d", __LINE__);
        exit(-1);
      }
      load();
    }
    
    void load() {
      FILE* propertyFile=fopen(propertyFilename.c_str(), "rb");
      fseek(propertyFile, 0, SEEK_END);
      long propertySize=ftell(propertyFile);
      rewind(propertyFile);
      propertiesStore=(double*)malloc(propertySize);
      if(propertiesStore==NULL) {
        fprintf(stderr, "Too few main memory. Program will terminate! - error %d\n", __LINE__);
        exit(-1);
      }
      fread(propertiesStore, 1, propertySize, propertyFile);
      fclose(propertyFile);
      
      FILE* propertyIdFile=fopen(propertyIdFilename.c_str(), "rb");
      fseek(propertyIdFile, 0, SEEK_END);
      long propertyIdSize=ftell(propertyIdFile);
      rewind(propertyIdFile);
      char* propertyId=(char*)malloc(propertyIdSize);
      if(propertyId==NULL) {
        fprintf(stderr, "Too few main memory. Program will terminate!\n");
        exit(-1);
      }
      fread(propertyId, 1, propertyIdSize, propertyIdFile);
      fclose(propertyIdFile);
      long propertyIdNr=(propertyIdSize/sizeof(char));
      
      propertyNr=0L;
      for(long i=0L; i<propertyIdNr; i++) {
        if(propertyId[i]=='\n') 
          propertyNr++;
      }
      free(propertyId);
      
      sampleNr=(propertySize/sizeof(double))/(propertyNr);
      properties=(double**)malloc(sizeof(double*)*sampleNr);
      for(long i=0L; i<sampleNr; i++) {
      //  if(id==1L)
      //    properties[i]=&(propertiesStore[propertyNr*i+167]);
      //  else
          properties[i]=&(propertiesStore[propertyNr*i]);
      }
      /*if(id==1L)
        propertyNr=881;*/
      
      propertyIndexToPropertyId=new std::string[propertyNr];
      std::ifstream propertyIdFileStream(propertyIdFilename.c_str(), std::ifstream::in);
      for(long i=0L; i<propertyNr; i++) {
        std::string mypropertyId;
        getline(propertyIdFileStream, mypropertyId);
        propertyIndexToPropertyId[i]=mypropertyId;
        propertyIdToPropertyIndex.insert(std::pair<std::string, long>(mypropertyId, i));
      }
      propertyIdFileStream.close();
      
      
      
    }
    
    void unload() {
      delete[] propertyIndexToPropertyId;
      free(properties);
      free(propertiesStore);
    }
    
    ~DenseFeatureData() {
       unload();
    }
    
    
    
    void restrict(const std::vector<long>& sampleNrVec, const std::string& dchemPathnameNew, const std::string& dFeatureType) {
      long sampleNrNew=sampleNrVec.size();
      std::string propertyVersionName(dFeatureType);
      std::string dfeatureDBPathnameNew(dchemPathnameNew); dfeatureDBPathnameNew=dfeatureDBPathnameNew+propertyVersionName+"/";
      if(mkdir(dfeatureDBPathnameNew.c_str(), S_IRWXU)==-1) {
        fprintf(stderr, "The directory '%s' either already exists or it is an invalid directory!\n", dfeatureDBPathnameNew.c_str());
        exit(-1);
      }
      std::string propertyFilenameNew=dfeatureDBPathnameNew+"properties.bin";
      std::string propertyIdFilenameNew=dfeatureDBPathnameNew+"PropertyIdTable.txt";
      std::string dataTypeInfoFilenameNew=dfeatureDBPathnameNew+"dt.txt";
      
      std::ofstream dataTypeInfoFile;
      dataTypeInfoFile.open(dataTypeInfoFilenameNew.c_str());
      dataTypeInfoFile << sizeof(long) << std::endl;
      dataTypeInfoFile.close();
      
      double* propertiesStoreNew=(double*)malloc(sizeof(double)*sampleNrNew*propertyNr);
      
      long propertyStoreLengthNew=0L;
      for(long i=0L; i<sampleNrNew; i++) {
        for(long j=0L; j<propertyNr; j++) {
          propertiesStoreNew[i*propertyNr+j]=propertiesStore[sampleNrVec[i]*propertyNr+j];
          propertyStoreLengthNew++;
        }
      }
      
      FILE* propertyFile=fopen(propertyFilenameNew.c_str(), "wb");
      fwrite(propertiesStoreNew, sizeof(double), propertyStoreLengthNew, propertyFile);
      fclose(propertyFile);
      
      free(propertiesStoreNew);
      
      
      
      FILE *fd1=fopen(propertyIdFilename.c_str(), "rb");
      FILE *fd2=fopen(propertyIdFilenameNew.c_str(), "wb");

      size_t l1;
      unsigned char buffer[8192]; 

      while((l1 = fread(buffer, 1L, sizeof(unsigned char)*8192L, fd1)) > 0) {
        fwrite(buffer, 1, l1, fd2);
      }
      
      fclose(fd1);
      fclose(fd2);
    }
    
    
    
    long id;
    double* propertiesStore;
    double** properties;
    long propertyNr;
    long sampleNr;
    std::string propertyFilename;
    std::string propertyIdFilename;
    std::map<std::string, long> propertyIdToPropertyIndex;
    std::string* propertyIndexToPropertyId;
};

class SparseFeatureData {
  public:
  
    SparseFeatureData(long ind, const std::string& schemPathname, const std::string& sFeatureType) {
      this->id=ind;
      struct stat sb;
      int statRet;
      std::string dataTypeInfoFilename;
      std::ifstream dataTypeInfoFile;
      std::string dataType;
      
      std::string fpVersionName(sFeatureType);
      std::string sfeatureDBPathname(schemPathname); sfeatureDBPathname=sfeatureDBPathname+fpVersionName+"/";
      intToExtMappingFilename=sfeatureDBPathname+"IntToExtMappingTable.bin";
      sampleFilename=sfeatureDBPathname+"fpSampleTable.bin";
      featureIntFilename=sfeatureDBPathname+"fpFeatureTableInt.bin";
      featureExtFilename=sfeatureDBPathname+"fpFeatureTableExt.bin";
      featureCountFilename=sfeatureDBPathname+"fpFeatureCountTableD.bin";
      dataTypeInfoFilename=sfeatureDBPathname+"dt.txt";
      statRet=(stat(intToExtMappingFilename.c_str(), &sb)==-1);
      statRet=statRet||(stat(sampleFilename.c_str(), &sb)==-1);
      statRet=statRet||(stat(featureIntFilename.c_str(), &sb)==-1);
      statRet=statRet||(stat(featureExtFilename.c_str(), &sb)==-1);
      statRet=statRet||(stat(featureCountFilename.c_str(), &sb)==-1);
      if(statRet==-1) {
        fprintf(stderr, "The database '%s' does either not exist or is not complete! - error %d\n", sfeatureDBPathname.c_str(), __LINE__);
        exit(-1);
      }
      dataTypeInfoFile.open(dataTypeInfoFilename.c_str(), std::ifstream::in);
      getline(dataTypeInfoFile, dataType);
      dataTypeInfoFile.close();
      if(sizeof(long)!=atoi(dataType.c_str())) {
        fprintf(stderr, "The system, which was used to create the database is not compatible with the current system! - error %d", __LINE__);
        exit(-1);
      }
      load();
    }
    
    void load() {
      FILE* featureFile=fopen(featureIntFilename.c_str(), "rb");
      fseek(featureFile, 0, SEEK_END);
      long featureSize=ftell(featureFile);
      rewind(featureFile);
      features=(long*)malloc(featureSize);
      if(features==NULL) {
        fprintf(stderr, "Too few main memory. Program will terminate! - error %d\n", __LINE__);
        exit(-1);
      }
      fread(features, 1, featureSize, featureFile);
      fclose(featureFile);
      
      FILE* featureCountFile=fopen(featureCountFilename.c_str(), "rb");
      fseek(featureCountFile, 0, SEEK_END);
      long featureCountSize=ftell(featureCountFile);
      rewind(featureCountFile);
      featureCounts=(double*)malloc(featureCountSize);
      if(featureCounts==NULL) {
        fprintf(stderr, "Too few main memory. Program will terminate! - error %d\n", __LINE__);
        exit(-1);
      }
      fread(featureCounts, 1, featureCountSize, featureCountFile);
      fclose(featureCountFile);
      
      FILE* sampleFile=fopen(sampleFilename.c_str(), "rb");
      fseek(sampleFile, 0, SEEK_END);
      long sampleSize=ftell(sampleFile);
      rewind(sampleFile);
      samples=(long*)malloc(sampleSize+(long)(sizeof(long)));
      if(samples==NULL) {
        fprintf(stderr, "Too few main memory. Program will terminate! - error %d\n", __LINE__);
        exit(-1);
      }
      fread(samples, 1, sampleSize, sampleFile);
      fclose(sampleFile);

      FILE* intToExtMappingFile=fopen(intToExtMappingFilename.c_str(), "rb");
      fseek(intToExtMappingFile, 0, SEEK_END);
      long intToExtMappingSize=ftell(intToExtMappingFile);
      rewind(intToExtMappingFile);
      internalFeatNrToExternalFeatNr=(long*)malloc(intToExtMappingSize);
      if(internalFeatNrToExternalFeatNr==NULL) {
        fprintf(stderr, "Too few main memory. Program will terminate! - error %d\n", __LINE__);
        exit(-1);
      }
      fread(internalFeatNrToExternalFeatNr, 1, intToExtMappingSize, intToExtMappingFile);
      fclose(intToExtMappingFile);
      
      featureNr=featureSize/sizeof(long);
      sampleNr=sampleSize/sizeof(long);
      samples[sampleNr]=featureNr;
      nrDistinctFeatures=intToExtMappingSize/sizeof(long);
    }
    
    void unload() {
      free(internalFeatNrToExternalFeatNr);
      free(samples);
      free(featureCounts);
      free(features);
    }
    
    ~SparseFeatureData() {
       unload();
    }
    
    void align(SparseFeatureData* sf) {
      std::map<long, long> externalFeatNrToInternalFeatNr;
      for(long i=0; i<nrDistinctFeatures; i++)
        externalFeatNrToInternalFeatNr.insert(std::pair<long, long>(internalFeatNrToExternalFeatNr[i], i));
      
      long* internalFeatNrToInternalFeatNrNew=(long*)malloc(sf->nrDistinctFeatures*sizeof(long));
      long nrDistinctFeaturesNew=nrDistinctFeatures;
      std::list<long> newExternalFeatNrList;
      for(long i=0; i<sf->nrDistinctFeatures; i++) {
        std::map<long,long>::iterator it = externalFeatNrToInternalFeatNr.find(sf->internalFeatNrToExternalFeatNr[i]);
        if (it == externalFeatNrToInternalFeatNr.end()) { 
          internalFeatNrToInternalFeatNrNew[i]=nrDistinctFeaturesNew;
          nrDistinctFeaturesNew++;
          newExternalFeatNrList.push_back(sf->internalFeatNrToExternalFeatNr[i]);
        }
        else {
          internalFeatNrToInternalFeatNrNew[i]=it->second;
        }
      }
      std::vector<long> newExternalFeatNr(newExternalFeatNrList.begin(), newExternalFeatNrList.end());
      
      long* internalFeatNrToExternalFeatNrNew=(long*)malloc(nrDistinctFeaturesNew*sizeof(long));
      memcpy(internalFeatNrToExternalFeatNrNew, internalFeatNrToExternalFeatNr, nrDistinctFeatures*sizeof(long));
      for(long i=nrDistinctFeatures; i<nrDistinctFeaturesNew; i++)
        internalFeatNrToExternalFeatNrNew[i]=newExternalFeatNr[i-nrDistinctFeatures];      
      
      FeatExtIntT* featuresExtInt=(FeatExtIntT*)malloc(sizeof(FeatExtIntT)*sf->featureNr);
      for(long i=0L; i<sf->featureNr; i++) {
        featuresExtInt[i].intId=internalFeatNrToInternalFeatNrNew[sf->features[i]];
        featuresExtInt[i].extId=internalFeatNrToExternalFeatNrNew[featuresExtInt[i].intId];
        featuresExtInt[i].count=sf->featureCounts[i];
      }
      for(long i=0L; i<sf->sampleNr; i++)
        qsort(&featuresExtInt[sf->samples[i]], sf->samples[i+1]-sf->samples[i], sizeof(FeatExtIntT), compareFeatExtInt);      
      
      for(long i=0L; i<sf->featureNr; i++) {
        sf->features[i]=featuresExtInt[i].intId;
        sf->featureCounts[i]=featuresExtInt[i].count;
      }
      free(featuresExtInt);
      
      free(internalFeatNrToInternalFeatNrNew);
      free(sf->internalFeatNrToExternalFeatNr);
      sf->internalFeatNrToExternalFeatNr=internalFeatNrToExternalFeatNrNew;
      sf->nrDistinctFeatures=nrDistinctFeaturesNew;
    }
    
    
    
    void restrict(const std::vector<long>& sampleNrVec, const std::string& schemPathnameNew, const std::string& sFeatureType) {
      long sampleNrNew=sampleNrVec.size();
      std::string fpVersionName(sFeatureType);
      std::string sfeatureDBPathnameNew(schemPathnameNew); sfeatureDBPathnameNew=sfeatureDBPathnameNew+fpVersionName+"/";
      if(mkdir(sfeatureDBPathnameNew.c_str(), S_IRWXU)==-1) {
        fprintf(stderr, "The directory '%s' either already exists or it is an invalid directory!\n", sfeatureDBPathnameNew.c_str());
        exit(-1);
      }
      std::string intToExtMappingFilenameNew=sfeatureDBPathnameNew+"IntToExtMappingTable.bin";
      std::string sampleFilenameNew=sfeatureDBPathnameNew+"fpSampleTable.bin";
      std::string featureIntFilenameNew=sfeatureDBPathnameNew+"fpFeatureTableInt.bin";
      std::string featureExtFilenameNew=sfeatureDBPathnameNew+"fpFeatureTableExt.bin";
      std::string featureCountFilenameNew=sfeatureDBPathnameNew+"fpFeatureCountTableD.bin";
      std::string dataTypeInfoFilenameNew=sfeatureDBPathnameNew+"dt.txt";
      
      std::ofstream dataTypeInfoFile;
      dataTypeInfoFile.open(dataTypeInfoFilenameNew.c_str());
      dataTypeInfoFile << sizeof(long) << std::endl;
      dataTypeInfoFile.close();
      
      long featureNrNew=0L;
      std::list<long> featureNrList;
      for(long i=0L; i<sampleNrNew; i++) {
        long beginFeat=samples[sampleNrVec[i]];
        long endFeat=samples[sampleNrVec[i]+1L];
        for(long j=beginFeat; j<endFeat; j++) {
          featureNrList.push_back(features[j]);
          featureNrNew++;
        }
      }
      std::vector<long> featureNrVec(featureNrList.begin(), featureNrList.end());
      std::sort(featureNrVec.begin(), featureNrVec.end());
      std::vector<long>::iterator it=std::unique(featureNrVec.begin(), featureNrVec.end());
      featureNrVec.resize(std::distance(featureNrVec.begin(), it));
      long nrDistinctFeaturesNew=featureNrVec.size();
      long* oldFeatureNrToNewFeatureNr=(long*)malloc(sizeof(long)*nrDistinctFeatures);
      if((oldFeatureNrToNewFeatureNr==NULL)) {
        fprintf(stderr, "Too few main memory. Program will terminate!\n");
        exit(-1);
      }
      for(long i=0L; i<nrDistinctFeatures; i++)
        oldFeatureNrToNewFeatureNr[i]=-1L;
      for(long i=0L; i<nrDistinctFeaturesNew; i++)
        oldFeatureNrToNewFeatureNr[featureNrVec[i]]=i;
      
      long* featuresNew=(long*)malloc(sizeof(long)*featureNrNew);
      long* featuresExtNew=(long*)malloc(sizeof(long)*featureNrNew);
      double* featureCountsNew=(double*)malloc(sizeof(double)*featureNrNew);
      FeatExtIntT* featExtInt=(FeatExtIntT*)malloc(sizeof(FeatExtIntT)*featureNr);
      long* samplesNew=(long*)malloc(sizeof(long)*sampleNrNew+(long)(sizeof(long)));
      long* internalFeatNrToExternalFeatNrNew=(long*)malloc(sizeof(long)*nrDistinctFeaturesNew);
      if((featuresNew==NULL)||(featureCountsNew==NULL)||(featExtInt==NULL)||(samplesNew==NULL)||(internalFeatNrToExternalFeatNrNew==NULL)) {
        fprintf(stderr, "Too few main memory. Program will terminate!\n");
        exit(-1);
      }
      
      featureNrNew=0L;
      for(long i=0; i<sampleNrNew; i++) {
        samplesNew[i]=featureNrNew;
        long beginFeat=samples[sampleNrVec[i]];
        long endFeat=samples[sampleNrVec[i]+1L];

        for(long j=beginFeat; j<endFeat; j++) {
          featExtInt[j].intId=oldFeatureNrToNewFeatureNr[features[j]];
          featExtInt[j].extId=internalFeatNrToExternalFeatNr[features[j]];
          featExtInt[j].count=featureCounts[j];
        }
        qsort(&featExtInt[beginFeat], endFeat-beginFeat, sizeof(FeatExtIntT), compareFeatExtInt);
        for(long j=beginFeat; j<endFeat; j++) {
          featuresNew[featureNrNew]=featExtInt[j].intId;
          featuresExtNew[featureNrNew]=featExtInt[j].extId;
          featureCountsNew[featureNrNew]=featExtInt[j].count;
          featureNrNew++;
        }
      }
      
      for(long i=0L; i<nrDistinctFeaturesNew; i++)
        internalFeatNrToExternalFeatNrNew[i]=internalFeatNrToExternalFeatNr[featureNrVec[i]];
      
      FILE* featureFile=fopen(featureIntFilenameNew.c_str(), "wb");
      fwrite(featuresNew, sizeof(long), featureNrNew, featureFile);
      fclose(featureFile);
      
      FILE* featureExtFile=fopen(featureExtFilenameNew.c_str(), "wb");
      fwrite(featuresExtNew, sizeof(long), featureNrNew, featureExtFile);
      fclose(featureExtFile);
      
      FILE* featureCountFile=fopen(featureCountFilenameNew.c_str(), "wb");
      fwrite(featureCountsNew, sizeof(double), featureNrNew, featureCountFile);
      fclose(featureCountFile);
      
      FILE* sampleFile=fopen(sampleFilenameNew.c_str(), "wb");
      fwrite(samplesNew, sizeof(long), sampleNrNew, sampleFile);
      fclose(sampleFile);
      
      FILE* intToExtMappingFile=fopen(intToExtMappingFilenameNew.c_str(), "wb");
      fwrite(internalFeatNrToExternalFeatNrNew, sizeof(long), nrDistinctFeaturesNew, intToExtMappingFile);
      fclose(intToExtMappingFile);
      
      free(internalFeatNrToExternalFeatNrNew);
      free(samplesNew);
      free(featExtInt);
      free(featureCountsNew);
      free(featuresExtNew);
      free(featuresNew);
      free(oldFeatureNrToNewFeatureNr);
    }
    
    
    
    long id;
    long* features;
    double* featureCounts;
    long* samples;
    long* internalFeatNrToExternalFeatNr;
    long featureNr;
    long sampleNr;
    long nrDistinctFeatures;
    std::string intToExtMappingFilename;
    std::string sampleFilename;
    std::string featureIntFilename;
    std::string featureExtFilename;
    std::string featureCountFilename;
};

class SparseFeatureKernel {
  public:    
    SparseFeatureKernel(const long nr, SparseFeatureData* feat, const KernelType mykt) {
      id=nr;
      feature=feat;
      kt=mykt;
    }
    
    SparseFeatureKernel(const SparseFeatureKernel &obj) {
      id=obj.id;
      feature=obj.feature;
      kt=obj.kt;
    }
    
    SparseFeatureKernel& operator= (const SparseFeatureKernel &obj) {
      id=obj.id;
      feature=obj.feature;
      kt=obj.kt;
      return *this;
    }
    
    long id;
    SparseFeatureData* feature;
    KernelType kt;
};

class DenseFeatureKernel {
  public:    
    DenseFeatureKernel(const long nr, DenseFeatureData* feat, const KernelType mykt) {
      id=nr;
      feature=feat;
      kt=mykt;
    }
    
    DenseFeatureKernel(const DenseFeatureKernel &obj) {
      id=obj.id;
      feature=obj.feature;
      kt=obj.kt;
    }
    
    DenseFeatureKernel& operator= (const DenseFeatureKernel &obj) {
      id=obj.id;
      feature=obj.feature;
      kt=obj.kt;
      return *this;
    }
    
    long id;
    DenseFeatureData* feature;
    KernelType kt;
};

class FeatureKernelCollection {
  public:
    FeatureKernelCollection() { }
    
    FeatureKernelCollection(const long nr, const std::vector<DenseFeatureKernel*>& dFeatureTypes, const std::vector<SparseFeatureKernel*>& sFeatureTypes, KernelComb mycomb, const long& norm, const long& mysim) {
      id=nr;
      for(long i=0L; i<dFeatureTypes.size(); i++)
        this->dFeatureKernels.push_back(dFeatureTypes[i]);
      for(long i=0L; i<sFeatureTypes.size(); i++)
        this->sFeatureKernels.push_back(sFeatureTypes[i]);
      kc=mycomb;
      kn=norm;
      sim=mysim;
    }
    
    FeatureKernelCollection(const FeatureKernelCollection &obj) {
      id=obj.id;
      dFeatureKernels=obj.dFeatureKernels;
      sFeatureKernels=obj.sFeatureKernels;
      kc=obj.kc;
      kn=obj.kn;
      sim=obj.sim;
    }
    
    FeatureKernelCollection& operator= (const FeatureKernelCollection &obj) {
      id=obj.id;
      dFeatureKernels=obj.dFeatureKernels;
      sFeatureKernels=obj.sFeatureKernels;
      kc=obj.kc;
      kn=obj.kn;
      sim=obj.sim;
      return *this;
    }
    
    long id;
    std::vector<DenseFeatureKernel*> dFeatureKernels;
    std::vector<SparseFeatureKernel*> sFeatureKernels;
    KernelComb kc;
    long kn;
    long sim;
};

class SVMHyperParam {
  public:
    SVMHyperParam(const FeatureKernelCollection& featureParam, const long& svmGen, const double& rbf) {
      this->fkc=featureParam;
      this->svmGen=svmGen;
      this->rbf=rbf;
    }
    
    SVMHyperParam(const SVMHyperParam &obj) {
      this->fkc=obj.fkc;
      this->svmGen=obj.svmGen;
      this->rbf=obj.rbf;
    }
    
    SVMHyperParam& operator= (const SVMHyperParam &obj) {
      this->fkc=obj.fkc;
      this->svmGen=obj.svmGen;
      this->rbf=obj.rbf;
      return *this;
    }
    
    FeatureKernelCollection fkc;
    long svmGen;
    double rbf;
};

bool paramSort(SVMHyperParam p1, SVMHyperParam p2) {
  if (p1.fkc.id<p2.fkc.id)
    return true;
  else if (p1.fkc.id>p2.fkc.id)
    return false;
  
  if(p1.rbf<p2.rbf)
    return true;
  else if(p1.rbf>p2.rbf)
    return false;
  
  if(p1.svmGen<p2.svmGen)
    return true;
  else if(p1.svmGen>p2.svmGen)
    return false;
  
  return false;
}

class KNNHyperParam {
  public:
    KNNHyperParam(const FeatureKernelCollection& featureParam, const long& knn) {
      this->fkc=featureParam;
      this->knn=knn;
    }
    
    KNNHyperParam(const KNNHyperParam &obj) {
      this->fkc=obj.fkc;
      this->knn=obj.knn;
    }
    
    KNNHyperParam& operator= (const KNNHyperParam &obj) {
      this->fkc=obj.fkc;
      this->knn=obj.knn;
      return *this;
    }
    
    FeatureKernelCollection fkc;
    long knn;
};

bool paramSortKNN(KNNHyperParam p1, KNNHyperParam p2) {
  if (p1.fkc.id<p2.fkc.id)
    return true;
  else if (p1.fkc.id>p2.fkc.id)
    return false;
  
  if(p1.knn<p2.knn)
    return true;
  else if(p1.knn>p2.knn)
    return false;
  
  return false;
}

class NBHyperParam {
  public:
    NBHyperParam(const FeatureKernelCollection& featureParam) {
      this->fkc=featureParam;
    }
    
    NBHyperParam(const NBHyperParam &obj) {
      this->fkc=obj.fkc;
    }
    
    NBHyperParam& operator= (const NBHyperParam &obj) {
      this->fkc=obj.fkc;
      return *this;
    }
    
    FeatureKernelCollection fkc;
};

bool paramSortNB(NBHyperParam p1, NBHyperParam p2) {
  if (p1.fkc.id<p2.fkc.id)
    return true;
  else if (p1.fkc.id>p2.fkc.id)
    return false;
    
  return false;
}

class SEAHyperParam {
  public:
    SEAHyperParam(const FeatureKernelCollection& featureParam) {
      this->fkc=featureParam;
    }
    
    SEAHyperParam(const SEAHyperParam &obj) {
      this->fkc=obj.fkc;
    }
    
    SEAHyperParam& operator= (const SEAHyperParam &obj) {
      this->fkc=obj.fkc;
      return *this;
    }
    
    FeatureKernelCollection fkc;
};

bool paramSortSEA(SEAHyperParam p1, SEAHyperParam p2) {
  if (p1.fkc.id<p2.fkc.id)
    return true;
  else if (p1.fkc.id>p2.fkc.id)
    return false;
    
  return false;
}





