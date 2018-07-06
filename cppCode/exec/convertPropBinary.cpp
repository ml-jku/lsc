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
    fprintf(stderr, "Usage: convertPropBinary project rawDataFolder propertyVersion\n");
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
  
  string propertyVersionName(argv[3]);
  
  string propertyTxtFilename=rawDataPathname+propertyVersionName+".csv";
  statRet=(stat(propertyTxtFilename.c_str(), &sb)==-1);
  if(statRet==-1) {
    fprintf(stderr, "The files '%s' does either not exist or is not complete!\n", propertyTxtFilename.c_str());
    exit(-1);
  }
    
  string dfeatureDBPathname(dchemPathname); dfeatureDBPathname=dfeatureDBPathname+propertyVersionName+"/";
  if(mkdir(dfeatureDBPathname.c_str(), S_IRWXU)==-1) {
    fprintf(stderr, "The directory '%s' either already exists or it is an invalid directory!\n", dfeatureDBPathname.c_str());
    exit(-1);
  }
  string propertyFilename=dfeatureDBPathname+"properties.bin";
  string propertyIdFilename=dfeatureDBPathname+"PropertyIdTable.txt";
  string dataTypeInfoFilename=dfeatureDBPathname+"dt.txt";
  
  printf("Creating binary database with the following parameters:\n");
  printf("project:         %30s\n", projectName.c_str());
  printf("rawdataFolder:   %30s\n", rawDataFoldername.c_str());
  printf("propertyVersion: %30s\n", propertyVersionName.c_str());
  
  
  
  ofstream dataTypeInfoFile;
  dataTypeInfoFile.open(dataTypeInfoFilename.c_str());
  dataTypeInfoFile << sizeof(long) << endl;
  dataTypeInfoFile.close();
  
  long sampleNr=0L;
  map<string, long> sampleIdToSampleIndex;
  ifstream sampleIdFile(sampleIdFilename.c_str(), ifstream::in);
  string sampleId;
  while(getline(sampleIdFile, sampleId)) {
    sampleIdToSampleIndex.insert(pair<string, long>(sampleId, sampleNr));
    sampleNr++;
  }
  sampleIdFile.close();
  fprintf(stderr, "sampleNr: %ld\n", sampleNr);
  
  FILE* propertyTxtFile=fopen(propertyTxtFilename.c_str(), "rb");
  fseek(propertyTxtFile, 0, SEEK_END);
  long propertyTxtSize=ftell(propertyTxtFile);
  rewind(propertyTxtFile);
  char* propertyTxt=(char*)malloc(propertyTxtSize+sizeof(char));
  if(propertyTxt==NULL) {
    fprintf(stderr, "Too few main memory. Program will terminate!\n");
    exit(-1);
  }
  fread(propertyTxt, 1, propertyTxtSize, propertyTxtFile);
  fclose(propertyTxtFile);
  long propertyTxtNr=(propertyTxtSize/sizeof(char));
  propertyTxt[propertyTxtNr]=0L;
  string propertyTxtStr(propertyTxt);
  
  std::list<string> propertyNames;  
  long endlInd=propertyTxtStr.find('\n');
  if(!(endlInd==std::string::npos)) {
    long startInd=0L;
    startInd=propertyTxtStr.find(',', startInd)+1L;
    for(;;) {
      long commaInd=propertyTxtStr.find(',', startInd);
      string field;
      if(commaInd==std::string::npos) {
        field=propertyTxtStr.substr(startInd);
        propertyNames.push_back(field);
        break;
      }
      if(commaInd>endlInd) {
        field=propertyTxtStr.substr(startInd, endlInd-startInd);
        propertyNames.push_back(field);
        break;
      }
      field=propertyTxtStr.substr(startInd, commaInd-startInd);
      propertyNames.push_back(field);
      startInd=commaInd+1L;
    }
  }
  
  long lines=0L;
  for(long i=0L; i<propertyTxtNr; i++) {
    if(propertyTxtStr[i]=='\n') 
      lines++;
  }
  
  double* propertiesStore=(double*)malloc(sizeof(double)*propertyNames.size()*lines);
  double** properties=(double**)malloc(sizeof(double*)*lines);
  for(long i=0L; i<lines; i++) {
    properties[i]=&(propertiesStore[propertyNames.size()*i]);
  }
  
  long propertyNr=propertyNames.size();
  if((propertyNr>0)&&(lines>1L)) {
    for(long i=1L; i<lines; i++) {
      string field;
      long startInd=endlInd+1L;
      endlInd=propertyTxtStr.find('\n', startInd);
      
      long commaInd=propertyTxtStr.find(',', startInd);
      field=propertyTxtStr.substr(startInd, commaInd-startInd);
      startInd=commaInd+1L;
      long storeIndex=sampleIdToSampleIndex[field];
      for(long j=0L; j<(propertyNr-1L); j++) {
        long commaInd=propertyTxtStr.find(',', startInd);
        field=propertyTxtStr.substr(startInd, commaInd-startInd);
        startInd=commaInd+1L;
        properties[storeIndex][j]=atof(field.c_str());
      }
      field=propertyTxtStr.substr(startInd, endlInd-startInd);
      properties[storeIndex][propertyNr-1L]=atof(field.c_str());
    }
  }
  
  FILE* propertyFile=fopen(propertyFilename.c_str(), "wb");
  fwrite(propertiesStore, sizeof(long), propertyNames.size()*(lines-1L), propertyFile);
  fclose(propertyFile);
  free(properties);
  free(propertiesStore);
  
  ofstream propertyIdFile;
  propertyIdFile.open(propertyIdFilename.c_str());
  for(std::list<string>::iterator it=propertyNames.begin(); it!=propertyNames.end(); ++it)
    propertyIdFile << (*it) << endl;
  propertyIdFile.close();
  
  printf(progName.c_str());
  printf(" terminated successfully!\n");
  
  return 0;
}
