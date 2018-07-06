/*
Copyright (C) 2018 Andreas Mayr
Licensed under GNU General Public License v3.0 (see http://www.bioinf.jku.at/research/lsc/LICENSE and https://github.com/ml-jku/lsc/blob/master/LICENSE)
*/



#include "sums.h"

sumRet_t tanimotoSumSparseBinary(long* featuresPred, long beginPredFeat, long endPredFeat, long* featuresTrain, long beginTrainFeat, long endTrainFeat) {
  sumRet_t ret;
  long countPred=beginPredFeat;
  long countTrain=beginTrainFeat;
  long out=0;

  if((countPred==endPredFeat)&&(countTrain==endTrainFeat)) {
    ret.sum1=out;
    ret.sum2=0.0;
    return ret;
  }

  while((countPred<endPredFeat)&&(countTrain<endTrainFeat)) {
    if(featuresTrain[countTrain]==featuresPred[countPred]) {
      out++;
      countTrain++;
      countPred++;
    }
    else if(featuresTrain[countTrain]<featuresPred[countPred]) {
      countTrain++;
    }
    else if(featuresTrain[countTrain]>featuresPred[countPred]) {
      countPred++;
    }
  }
  ret.sum1=out;
  ret.sum2=(endTrainFeat+endPredFeat-beginTrainFeat-beginPredFeat-out);
  return ret;
}

sumRet_t tanimotoSumSparseCount(long* featuresPred, double* featureCountsPred, long beginPredFeat, long endPredFeat, long* featuresTrain, double* featureCountsTrain, long beginTrainFeat, long endTrainFeat) {
  sumRet_t ret;
  long countPred=beginPredFeat;
  long countTrain=beginTrainFeat;
  double minmaxSum=0.0;
  long commonSum=0;

  if((countPred==endPredFeat)&&(countTrain==endTrainFeat)) {
    ret.sum1=minmaxSum;
    ret.sum2=0.0;
    return ret;
  }

  while((countPred<endPredFeat)&&(countTrain<endTrainFeat)) {
    if(featuresTrain[countTrain]==featuresPred[countPred]) {
      if(featureCountsTrain[countTrain]>featureCountsPred[countPred]) {
        minmaxSum=minmaxSum+((double)(featureCountsPred[countPred]))/((double)featureCountsTrain[countTrain]);
        commonSum=commonSum+1;
      }
      else {
        minmaxSum=minmaxSum+((double)featureCountsTrain[countTrain])/((double)(featureCountsPred[countPred]));
        commonSum=commonSum+1;
      }
      countTrain++;
      countPred++;
    }
    else if(featuresTrain[countTrain]<featuresPred[countPred]) {
      commonSum=commonSum+1;
      countTrain++;
    }
    else if(featuresTrain[countTrain]>featuresPred[countPred]) {
      commonSum=commonSum+1;
      countPred++;
    }
  }
  while((countTrain<endTrainFeat)) {
    commonSum=commonSum+1;
    countTrain++;
  }
  while((countPred<endPredFeat)) {
    commonSum=commonSum+1;
    countPred++;
  }
  ret.sum1=minmaxSum;
  ret.sum2=commonSum;
  return ret;
}

template <typename T> sumRet_t tanimotoSumDenseOrig(T* propertiesPred, T* propertiesTrain, long length) {
  sumRet_t ret;
  double minmaxSum=0.0;
  long commonSum=0;
  for(long i=0L; i<length; i++) {
    double propPred=propertiesPred[i];
    double propTrain=propertiesTrain[i];
    if(fmax(propPred,propTrain)>0.00000) {
      minmaxSum=minmaxSum+((double)fmin(propPred,propTrain))/((double)fmax(propPred,propTrain));
      commonSum=commonSum+1;
    }
  }
  ret.sum1=minmaxSum;
  if(commonSum>0L)
    ret.sum2=commonSum;
  else
    ret.sum2=0.0;
  return ret;
}

template <typename T> sumRet_t tanimotoSumDenseOrigSplit(T* propertiesPred, T* propertiesTrain, long length) {
  sumRet_t ret;
  double minmaxSum=0.0;
  long commonSum=0;
  for(long i=0L; i<length; i++) {
    double propPred=propertiesPred[i];
    double propTrain=propertiesTrain[i];
    double propPredPos;
    double propPredNeg;
    double propTrainPos;
    double propTrainNeg;
    if(propPred>0) {
      propPredPos=propPred;
      propPredNeg=0.0;
    }
    else {
      propPredPos=0.0;
      propPredNeg=-propPred;
    }
    if(propTrain>0) {
      propTrainPos=propTrain;
      propTrainNeg=0.0;
    }
    else {
      propTrainPos=0.0;
      propTrainNeg=-propTrain;
    }
    if(fmax(propPredPos,propTrainPos)>0.00000) {
      minmaxSum=minmaxSum+((double)fmin(propPredPos,propTrainPos))/((double)fmax(propPredPos,propTrainPos));
      commonSum=commonSum+1;
    }
    if(fmax(propPredNeg,propTrainNeg)>0.00000) {
      minmaxSum=minmaxSum+((double)fmin(propPredNeg,propTrainNeg))/((double)fmax(propPredNeg,propTrainNeg));
      commonSum=commonSum+1;
    }
  }
  ret.sum1=minmaxSum;
  if(commonSum>0L)
    ret.sum2=commonSum;
  else
    ret.sum2=0.0;
  return ret;
}












sumRet_t tanimotoSumSparseBinary2(long* featuresPred, long beginPredFeat, long endPredFeat, long* featuresTrain, long beginTrainFeat, long endTrainFeat) {
  sumRet_t ret;
  long countPred=beginPredFeat;
  long countTrain=beginTrainFeat;
  long out=0;

  if((countPred==endPredFeat)&&(countTrain==endTrainFeat)) {
    ret.sum1=0.0;
    ret.sum2=0.0;
    return ret;
  }

  while((countPred<endPredFeat)&&(countTrain<endTrainFeat)) {
    if(featuresTrain[countTrain]==featuresPred[countPred]) {
      out++;
      countTrain++;
      countPred++;
    }
    else if(featuresTrain[countTrain]<featuresPred[countPred]) {
      countTrain++;
    }
    else if(featuresTrain[countTrain]>featuresPred[countPred]) {
      countPred++;
    }
  }
  ret.sum1=out;
  ret.sum2=(endTrainFeat+endPredFeat-beginTrainFeat-beginPredFeat-out);
  return ret;
}

sumRet_t tanimotoSumSparseCount2(long* featuresPred, double* featureCountsPred, long beginPredFeat, long endPredFeat, long* featuresTrain, double* featureCountsTrain, long beginTrainFeat, long endTrainFeat) {
  sumRet_t ret;
  long countPred=beginPredFeat;
  long countTrain=beginTrainFeat;
  double minmaxSum=0.0;
  double commonSum=0;

  if((countPred==endPredFeat)&&(countTrain==endTrainFeat)) {
    ret.sum1=0.0;
    ret.sum2=0.0;
    return ret;
  }

  while((countPred<endPredFeat)&&(countTrain<endTrainFeat)) {
    if(featuresTrain[countTrain]==featuresPred[countPred]) {
      if(featureCountsTrain[countTrain]>featureCountsPred[countPred]) {
        minmaxSum=minmaxSum+((double)(featureCountsPred[countPred]));
        commonSum=commonSum+((double)featureCountsTrain[countTrain]);
      }
      else {
        minmaxSum=minmaxSum+((double)featureCountsTrain[countTrain]);
        commonSum=commonSum+((double)(featureCountsPred[countPred]));
      }
      countTrain++;
      countPred++;
    }
    else if(featuresTrain[countTrain]<featuresPred[countPred]) {
      commonSum=commonSum+((double)featureCountsTrain[countTrain]);
      countTrain++;
    }
    else if(featuresTrain[countTrain]>featuresPred[countPred]) {
      commonSum=commonSum+((double)(featureCountsPred[countPred]));
      countPred++;
    }
  }
  while((countTrain<endTrainFeat)) {
    commonSum=commonSum+((double)featureCountsTrain[countTrain]);
    countTrain++;
  }
  while((countPred<endPredFeat)) {
    commonSum=commonSum+((double)(featureCountsPred[countPred]));
    countPred++;
  }
  ret.sum1=minmaxSum;
  ret.sum2=commonSum;
  return ret;
}

template <typename T> sumRet_t tanimotoSumDenseOrig2(T* propertiesPred, T* propertiesTrain, long length) {
  sumRet_t ret;
  double minmaxSum=0.0;
  double commonSum=0.0;
  for(long i=0L; i<length; i++) {
    double propPred=propertiesPred[i];
    double propTrain=propertiesTrain[i];
    minmaxSum=minmaxSum+((double)fmin(propPred,propTrain));
    commonSum=commonSum+((double)fmax(propPred,propTrain));
  }
  ret.sum1=minmaxSum;
  ret.sum2=commonSum;
  return ret;
}



















double linearSumSparseBinary(long* featuresPred, long beginPredFeat, long endPredFeat, long* featuresTrain, long beginTrainFeat, long endTrainFeat) {
  long countPred=beginPredFeat;
  long countTrain=beginTrainFeat;
  long out=0;

  if((countPred==endPredFeat)&&(countTrain==endTrainFeat))
    return(0.0);

  while((countPred<endPredFeat)&&(countTrain<endTrainFeat)) {
    if(featuresTrain[countTrain]==featuresPred[countPred]) {
      out++;
      countTrain++;
      countPred++;
    }
    else if(featuresTrain[countTrain]<featuresPred[countPred]) {
      countTrain++;
    }
    else if(featuresTrain[countTrain]>featuresPred[countPred]) {
      countPred++;
    }
  }
  return(((double)(out)));
}

double linearSumSparseCount(long* featuresPred, double* featureCountsPred, long beginPredFeat, long endPredFeat, long* featuresTrain, double* featureCountsTrain, long beginTrainFeat, long endTrainFeat) {
  long countPred=beginPredFeat;
  long countTrain=beginTrainFeat;
  double out=0;

  if((countPred==endPredFeat)&&(countTrain==endTrainFeat))
    return(0.0);

  while((countPred<endPredFeat)&&(countTrain<endTrainFeat)) {
    if(featuresTrain[countTrain]==featuresPred[countPred]) {
      out=out+featureCountsTrain[countTrain]*featureCountsPred[countPred];
      countTrain++;
      countPred++;
    }
    else if(featuresTrain[countTrain]<featuresPred[countPred]) {
      countTrain++;
    }
    else if(featuresTrain[countTrain]>featuresPred[countPred]) {
      countPred++;
    }
  }
  return(((double)(out)));
}

template <typename T> double linearSumDense(T* propertiesPred, T* propertiesTrain, long length) {
  double sum=0.0;
  for(long i=0L; i<length; i++)
    sum=sum+propertiesPred[i]*propertiesTrain[i];
  return sum;
}












double gaussianSumSparseBinary(long* featuresPred, long beginPredFeat, long endPredFeat, long* featuresTrain, long beginTrainFeat, long endTrainFeat) {
  long countPred=beginPredFeat;
  long countTrain=beginTrainFeat;
  long out=0;

  if((countPred==endPredFeat)&&(countTrain==endTrainFeat))
    return(0.0);

  while((countPred<endPredFeat)&&(countTrain<endTrainFeat)) {
    if(featuresTrain[countTrain]==featuresPred[countPred]) {
      countTrain++;
      countPred++;
    }
    else if(featuresTrain[countTrain]<featuresPred[countPred]) {
      out++;
      countTrain++;
    }
    else if(featuresTrain[countTrain]>featuresPred[countPred]) {
      out++;
      countPred++;
    }
  }
  return(-((double)(out)));
}

double gaussianSumSparseCount(long* featuresPred, double* featureCountsPred, long beginPredFeat, long endPredFeat, long* featuresTrain, double* featureCountsTrain, long beginTrainFeat, long endTrainFeat) {
  long countPred=beginPredFeat;
  long countTrain=beginTrainFeat;
  double out=0;

  if((countPred==endPredFeat)&&(countTrain==endTrainFeat))
    return(0.0);

  while((countPred<endPredFeat)&&(countTrain<endTrainFeat)) {
    if(featuresTrain[countTrain]==featuresPred[countPred]) {
      out=out+(featureCountsTrain[countTrain]-featureCountsPred[countPred])*(featureCountsTrain[countTrain]-featureCountsPred[countPred]);
      countTrain++;
      countPred++;
    }
    else if(featuresTrain[countTrain]<featuresPred[countPred]) {
      out=out+(featureCountsTrain[countTrain])*(featureCountsTrain[countTrain]);
      countTrain++;
    }
    else if(featuresTrain[countTrain]>featuresPred[countPred]) {
      out=out+(featureCountsPred[countPred])*(featureCountsPred[countPred]);
      countPred++;
    }
  }
  return(-((double)(out)));
}

template <typename T> double gaussianSumDense(T* propertiesPred, T* propertiesTrain, long length) {
  double sum=0.0;
  for(long i=0L; i<length; i++)
    sum=sum+(propertiesPred[i]-propertiesTrain[i])*(propertiesPred[i]-propertiesTrain[i]);
  return -sum;
}













template sumRet_t tanimotoSumDenseOrig<float>(float* propertiesPred, float* propertiesTrain, long length);
template sumRet_t tanimotoSumDenseOrig<double>(double* propertiesPred, double* propertiesTrain, long length);
template sumRet_t tanimotoSumDenseOrigSplit<float>(float* propertiesPred, float* propertiesTrain, long length);
template sumRet_t tanimotoSumDenseOrigSplit<double>(double* propertiesPred, double* propertiesTrain, long length);
template sumRet_t tanimotoSumDenseOrig2<float>(float* propertiesPred, float* propertiesTrain, long length);
template sumRet_t tanimotoSumDenseOrig2<double>(double* propertiesPred, double* propertiesTrain, long length);
template double linearSumDense<float>(float* propertiesPred, float* propertiesTrain, long length);
template double linearSumDense<double>(double* propertiesPred, double* propertiesTrain, long length);
template double gaussianSumDense<float>(float* propertiesPred, float* propertiesTrain, long length);
template double gaussianSumDense<double>(double* propertiesPred, double* propertiesTrain, long length);