/*
Copyright (C) 2018 Andreas Mayr
Licensed under GNU General Public License v3.0 (see http://www.bioinf.jku.at/research/lsc/LICENSE and https://github.com/ml-jku/lsc/blob/master/LICENSE)
*/



#include "targetPipeline.h"
#include <gsl/gsl_randist.h>



using namespace std;



int compareDiscT(const void* a, const void* b) {
  double val1=((DiscT*)a)->discValue;
  double val2=((DiscT*)b)->discValue;
  if(val1>val2)
    return 1;
  if(val1<val2)
    return -1;
  return 0;
}

/*
post-analysis after publication showed slight differences to python AUC function;
most probable because multiple exactly same discrimation values not considered 
esspecially for ranking here;
may lead to slight changes in values, but no different conclusions
or changes in the order of the leading methods [DNN, SVM, RF, GC]/feature encodings;
for non-leading methods (which may have also been more 
likely to prodcue exactly same discrimation values) at overall also 
only slight changes and no different conclusions
*/

double compAUC(long* labels, double* disc, long length) {
  double pos=0.0;
  double neg=0.0;
  DiscT* discVec=(DiscT*)std::malloc(sizeof(DiscT)*length);
  for(long i=0L; i<length; i++) {
    if(labels[i]==0L) neg=neg+1.0;
    if(labels[i]==1L) pos=pos+1.0;
    discVec[i].discValue=disc[i];
    discVec[i].origIndex=i;
  }
  qsort(discVec, length, sizeof(DiscT), compareDiscT);
  double R=0.0;
  for(long i=0L; i<length; i++) {
    long origInd=discVec[i].origIndex;
    if(labels[origInd]==1L) {
      R=R+(double)(i+1L);
    }
  }
  double auc=(1.0/(pos*neg))*(R-(pos*(pos+1.0))/(2.0));
  std::free(discVec);
  return auc;
}



int compareTrainRec(const void* av, const void* bv) {
  TrainRec* a=(TrainRec*) av;
  TrainRec* b=(TrainRec*) bv;
  
  int cmp=(a->target).compare(b->target);
  if(cmp!=0) return cmp;
  
  if((a->sampleNr)>(b->sampleNr))
    cmp=1;
  else if((a->sampleNr)<(b->sampleNr))
    cmp=-1;
  else
    cmp=0;
  if(cmp!=0) return cmp;
  
  if((a->label)>(b->label))
    cmp=1;
  else if((a->label)<(b->label))
    cmp=-1;
  else
    cmp=0;
  if(cmp!=0) return cmp;
  
  return cmp;
}

int compareSampleId(const void* av, const void* bv) {
  IndTrainRec* ia=(IndTrainRec*) av;
  IndTrainRec* ib=(IndTrainRec*) bv;
  TrainRec* a=ia->rec;
  TrainRec* b=ib->rec;
  
  int cmp;
  if((a->sampleNr)>(b->sampleNr))
    cmp=1;
  else if((a->sampleNr)<(b->sampleNr))
    cmp=-1;
  else
    cmp=0;
  if(cmp!=0) return cmp;
  
  cmp=(a->target).compare(b->target);
  if(cmp!=0) return cmp;
  
  if((a->label)>(b->label))
    cmp=1;
  else if((a->label)<(b->label))
    cmp=-1;
  else
    cmp=0;
  if(cmp!=0) return cmp;
  
  return (cmp);
}

long* getSampleIndices(TrainRec* rec, long length) {
  IndTrainRec* indRec=new IndTrainRec[length];
  for(long i=0; i<length; i++) {
    indRec[i].ind=i;
    indRec[i].rec=&(rec[i]);
  }
  qsort(indRec, length, sizeof(IndTrainRec), compareSampleId);
  long* indices=(long*)malloc(sizeof(long)*length);
  for(long i=0; i<length; i++) {
    indices[i]=indRec[i].ind;
  }
  delete[] indRec;
  return indices;
}

int compareClusterRec(const void* av, const void* bv) {
  ClusterRec* a=(ClusterRec*) av;
  ClusterRec* b=(ClusterRec*) bv;
  
  int cmp;
  if((a->clusterNr)>(b->clusterNr))
    cmp=1;
  else if((a->clusterNr)<(b->clusterNr))
    cmp=-1;
  else
    cmp=0;
  if(cmp!=0) return cmp;
  
  if((a->sampleNr)>(b->sampleNr))
    cmp=1;
  else if((a->sampleNr)<(b->sampleNr))
    cmp=-1;
  else
    cmp=0;
  if(cmp!=0) return cmp;
  
  return cmp;
}
