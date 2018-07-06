/*
Copyright (C) 2018 Andreas Mayr
Licensed under GNU General Public License v3.0 (see http://www.bioinf.jku.at/research/lsc/LICENSE and https://github.com/ml-jku/lsc/blob/master/LICENSE)
*/



#ifndef SUMS_H
#define SUMS_H

#include <cstdlib>
#include <cmath>

using namespace std;

typedef struct sumRet {
  double sum1;
  double sum2;
} sumRet_t;

sumRet_t tanimotoSumSparseBinary(long* featuresPred, long beginPredFeat, long endPredFeat, long* featuresTrain, long beginTrainFeat, long endTrainFeat);
sumRet_t tanimotoSumSparseCount(long* featuresPred, double* featureCountsPred, long beginPredFeat, long endPredFeat, long* featuresTrain, double* featureCountsTrain, long beginTrainFeat, long endTrainFeat);
template <typename T> sumRet_t tanimotoSumDenseOrig(T* propertiesPred, T* propertiesTrain, long length);
template <typename T> sumRet_t tanimotoSumDenseOrigSplit(T* propertiesPred, T* propertiesTrain, long length);
sumRet_t tanimotoSumSparseBinary2(long* featuresPred, long beginPredFeat, long endPredFeat, long* featuresTrain, long beginTrainFeat, long endTrainFeat);
sumRet_t tanimotoSumSparseCount2(long* featuresPred, double* featureCountsPred, long beginPredFeat, long endPredFeat, long* featuresTrain, double* featureCountsTrain, long beginTrainFeat, long endTrainFeat);
template <typename T> sumRet_t tanimotoSumDenseOrig2(T* propertiesPred, T* propertiesTrain, long length);
double linearSumSparseBinary(long* featuresPred, long beginPredFeat, long endPredFeat, long* featuresTrain, long beginTrainFeat, long endTrainFeat);
double linearSumSparseCount(long* featuresPred, double* featureCountsPred, long beginPredFeat, long endPredFeat, long* featuresTrain, double* featureCountsTrain, long beginTrainFeat, long endTrainFeat);
template <typename T> double linearSumDense(T* propertiesPred, T* propertiesTrain, long length);
double gaussianSumSparseBinary(long* featuresPred, long beginPredFeat, long endPredFeat, long* featuresTrain, long beginTrainFeat, long endTrainFeat);
double gaussianSumSparseCount(long* featuresPred, double* featureCountsPred, long beginPredFeat, long endPredFeat, long* featuresTrain, double* featureCountsTrain, long beginTrainFeat, long endTrainFeat);
template <typename T> double gaussianSumDense(T* propertiesPred, T* propertiesTrain, long length);
#endif
