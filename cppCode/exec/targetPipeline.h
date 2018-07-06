/*
Copyright (C) 2018 Andreas Mayr
Licensed under GNU General Public License v3.0 (see http://www.bioinf.jku.at/research/lsc/LICENSE and https://github.com/ml-jku/lsc/blob/master/LICENSE)
*/



#ifndef TARGETPIPELINE_H
#define TARGETPIPELINE_H

#include <string>
#include <cstdlib>

using namespace std;


typedef struct {
  long origIndex;
  double discValue;
} DiscT;

int compareDiscT(const void* a, const void* b);

/*see comment for compAUC at function body (comparison to Python AUC function)*/
double compAUC(long* labels, double* disc, long length);



class TrainRec {
public:
  long sampleNr;
  long label;
  string target;
};

int compareTrainRec(const void* av, const void* bv);


class IndTrainRec {
public:
  long ind;
  TrainRec* rec;
};

int compareSampleId(const void* av, const void* bv);

long* getSampleIndices(TrainRec* rec, long length);


class ClusterRec {
public:
  long clusterNr;
  long sampleNr;
};

int compareClusterRec(const void* av, const void* bv);



#endif
