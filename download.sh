#!/bin/bash

#Copyright (C) 2018 Andreas Mayr
#Licensed under GNU General Public License v3.0 (see http://www.bioinf.jku.at/research/lsc/LICENSE and https://github.com/ml-jku/lsc/blob/master/LICENSE)

createDir=$1

mkdir -p $createDir/chemFeatures/
mkdir -p $createDir/chemFeatures/cl
mkdir -p $createDir/chemFeatures/d
mkdir -p $createDir/chemFeatures/s
mkdir -p $createDir/train
mkdir -p $createDir/dataPython
mkdir -p $createDir/dataPythonReduced

wget http://bioinf.jku.at/research/lsc/chembl20/chemFeatures/SampleIdTable.txt --directory=$createDir/chemFeatures
wget http://bioinf.jku.at/research/lsc/chembl20/chemFeatures/cl/clusterMinFull.zip --directory=$createDir/chemFeatures/cl
wget http://bioinf.jku.at/research/lsc/chembl20/chemFeatures/cl/cl1.info --directory=$createDir/chemFeatures/cl
wget http://bioinf.jku.at/research/lsc/chembl20/chemFeatures/d/dense.zip --directory=$createDir/chemFeatures/d
wget http://bioinf.jku.at/research/lsc/chembl20/chemFeatures/d/semisparse.zip --directory=$createDir/chemFeatures/d
wget http://bioinf.jku.at/research/lsc/chembl20/chemFeatures/d/toxicophores.zip --directory=$createDir/chemFeatures/d
wget http://bioinf.jku.at/research/lsc/chembl20/chemFeatures/s/ECFC4.zip --directory=$createDir/chemFeatures/s
wget http://bioinf.jku.at/research/lsc/chembl20/chemFeatures/s/ECFC6_ES.zip --directory=$createDir/chemFeatures/s
wget http://bioinf.jku.at/research/lsc/chembl20/chemFeatures/s/DFS8_ES.zip --directory=$createDir/chemFeatures/s
wget http://bioinf.jku.at/research/lsc/chembl20/chemFeatures/s/semisparse.zip --directory=$createDir/chemFeatures/s
wget http://bioinf.jku.at/research/lsc/chembl20/chemFeatures/s/toxicophores.zip --directory=$createDir/chemFeatures/s
wget http://bioinf.jku.at/research/lsc/chembl20/chemFeatures/s/toxicophores.zip --directory=$createDir/chemFeatures/s
wget http://bioinf.jku.at/research/lsc/chembl20/train.zip --directory=$createDir
wget http://bioinf.jku.at/research/lsc/chembl20/dataPython.zip --directory=$createDir
wget http://bioinf.jku.at/research/lsc/chembl20/dataPythonReduced.zip --directory=$createDir
wget http://bioinf.jku.at/research/lsc/chembl20/dataPythonReduced/chembl20LSTM.pckl --directory=$createDir/dataPythonReduced
wget http://bioinf.jku.at/research/lsc/chembl20/dataPythonReduced/chembl20Smiles.pckl --directory=$createDir/dataPythonReduced
wget http://bioinf.jku.at/research/lsc/chembl20/dataPythonReduced/chembl20MACCS.pckl --directory=$createDir/dataPythonReduced
wget http://bioinf.jku.at/research/lsc/chembl20/dataPythonReduced/chembl20Deepchem.pckl --directory=$createDir/dataPythonReduced
wget http://bioinf.jku.at/research/lsc/chembl20/dataPythonReduced/chembl20Conv.pckl --directory=$createDir/dataPythonReduced
wget http://bioinf.jku.at/research/lsc/chembl20/dataPythonReduced/chembl20Weave.pckl --directory=$createDir/dataPythonReduced

unzip $createDir/chemFeatures/cl/clusterMinFull.zip -d $createDir/chemFeatures/cl
unzip $createDir/chemFeatures/d/dense.zip -d $createDir/chemFeatures/d
unzip $createDir/chemFeatures/d/semisparse.zip -d $createDir/chemFeatures/d
unzip $createDir/chemFeatures/d/toxicophores.zip -d $createDir/chemFeatures/d
unzip $createDir/chemFeatures/s/ECFC4.zip -d $createDir/chemFeatures/s
unzip $createDir/chemFeatures/s/ECFC6_ES.zip -d $createDir/chemFeatures/s
unzip $createDir/chemFeatures/s/DFS8_ES.zip -d $createDir/chemFeatures/s
unzip $createDir/chemFeatures/s/semisparse.zip -d $createDir/chemFeatures/s
unzip $createDir/chemFeatures/s/toxicophores.zip -d $createDir/chemFeatures/s
unzip $createDir/train.zip -d $createDir
unzip $createDir/dataPython.zip -d $createDir
unzip $createDir/dataPythonReduced.zip -d $createDir

