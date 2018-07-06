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
#include <set>

using namespace std;



int main(int argc, char** argv) {
  if(sizeof(long)<8) {
    printf("This program is optimized for machines with at least 8 byte longs!  The program will terminate!\n");
    return -1;
  }
  string progName(argv[0]);
  string basePathname(getenv("HOME")); basePathname=basePathname+"/mydata/trgpred/";
  
  
  
  if(argc!=3) {
    printf("Usage: genDirStructure projectName rawDataFolder\n");
    return -1;
  }

  string projectName(argv[1]);
  string projectPathname=basePathname+projectName+"/";
  if(mkdir(projectPathname.c_str(), S_IRWXU)==-1) {
    fprintf(stderr, "The directory '%s' either already exists or it is an invalid directory!\n", projectPathname.c_str());
    exit(-1);
  }
  string chemPathname=projectPathname+"chemFeatures/";
  string clusterPathname=chemPathname+"cl";
  string dchemPathname=chemPathname+"d";
  string schemPathname=chemPathname+"s";
  string trainPathname=projectPathname+"train";
  string runPathname=projectPathname+"run";
  string destSampleIdFilename=chemPathname+"SampleIdTable.txt";
  
  string rawDataFoldername(argv[2]);
  string rawDataPathname(getenv("HOME"));
  rawDataPathname=rawDataPathname+"/mydata/raw/"+rawDataFoldername+"/";
  string sourceSampleIdFilename=rawDataPathname+"SampleIdTable.txt";
  
  
  
  mkdir(chemPathname.c_str(), S_IRWXU);
  mkdir(clusterPathname.c_str(), S_IRWXU);
  mkdir(dchemPathname.c_str(), S_IRWXU);
  mkdir(schemPathname.c_str(), S_IRWXU);
  mkdir(trainPathname.c_str(), S_IRWXU);
  mkdir(runPathname.c_str(), S_IRWXU);

  
  FILE *fd1=fopen(sourceSampleIdFilename.c_str(), "rb");
  FILE *fd2=fopen(destSampleIdFilename.c_str(), "wb");

  size_t l1;
  unsigned char buffer[8192]; 

  while((l1 = fread(buffer, 1L, sizeof(unsigned char)*8192L, fd1)) > 0) {
    fwrite(buffer, 1, l1, fd2);
  }
  
  fclose(fd1);
  fclose(fd2);

  printf(progName.c_str());
  printf(" terminated successfully!\n");
  
  return 0;
}
