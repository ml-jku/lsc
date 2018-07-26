#!/bin/bash

#Copyright (C) 2018 Andreas Mayr
#Licensed under GNU General Public License v3.0 (see http://www.bioinf.jku.at/research/lsc/LICENSE and https://github.com/ml-jku/lsc/blob/master/LICENSE)

mkdir -p $HOME/mydata/raw/chembl20
mkdir -p $HOME/mydata/trgpred/chembl20
mkdir -p $HOME/myprogs

svn export https://github.com/ml-jku/lsc/trunk mycode
$HOME/mycode/cppCode/exec/install.sh

wget ftp://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_20/chembl_20.sdf.gz --directory=$HOME/mydata/raw/chembl20
gunzip $HOME/mydata/raw/chembl20/chembl_20.sdf.gz
wget http://www.bioinf.jku.at/research/lsc/training/SampleIdTable.txt --directory=$HOME/mydata/raw/chembl20
wget http://www.bioinf.jku.at/research/lsc/compounds/ECFC4.fpf --directory=$HOME/mydata/raw/chembl20
wget http://www.bioinf.jku.at/research/lsc/compounds/ECFC6_ES.fpf --directory=$HOME/mydata/raw/chembl20
wget http://www.bioinf.jku.at/research/lsc/compounds/DFS8_ES.fpf --directory=$HOME/mydata/raw/chembl20
wget http://www.bioinf.jku.at/research/lsc/compounds/dense.csv --directory=$HOME/mydata/raw/chembl20
wget http://www.bioinf.jku.at/research/lsc/compounds/semisparse.csv --directory=$HOME/mydata/raw/chembl20
wget http://www.bioinf.jku.at/research/lsc/compounds/toxicophores.csv --directory=$HOME/mydata/raw/chembl20
wget http://www.bioinf.jku.at/research/lsc/training/cl1.info --directory=$HOME/mydata/raw/chembl20
wget http://www.bioinf.jku.at/research/lsc/training/clusterMinFull.zip --directory=$HOME/mydata/raw/chembl20
wget http://www.bioinf.jku.at/research/lsc/training/train.info --directory=$HOME/mydata/raw/chembl20
wget http://www.bioinf.jku.at/research/lsc/training/tocompute.info --directory=$HOME/mydata/raw/chembl20



#####only for reproducibility of fingerprint features#####
#If the following wgets do not work immediately, try to start it multiple times after each other

#wget https://sourceforge.net/code-snapshots/svn/j/jc/jcompoundmapper/code/jcompoundmapper-code-r55.zip --directory=$HOME/myprogs
#unzip $HOME/myprogs/jcompoundmapper-code-r55.zip -d $HOME/myprogs/
svn export svn://svn.code.sf.net/p/jcompoundmapper/code/ $HOME/myprogs/jcompoundmapper-code-r55 -r 55
sed -i "59isubstructureHash=false;" $HOME/myprogs/jcompoundmapper-code-r55/src/de/zbit/jcmapper/fingerprinters/topological/features/ECFPFeature.java
sed -i "36inewHash=seed;" $HOME/myprogs/jcompoundmapper-code-r55/src/de/zbit/jcmapper/io/writer/ExporterHelper.java
ant -buildfile $HOME/myprogs/jcompoundmapper-code-r55/build.xml all

wget https://sourceforge.net/projects/jcompoundmapper/files/jCMapperCLI.jar --directory=$HOME/myprogs

wget https://bitbucket.org/jskDr/pcfp/get/39eb310c1b95.zip --directory=$HOME/myprogs
unzip $HOME/myprogs/39eb310c1b95.zip -d $HOME/myprogs/
ant -buildfile $HOME/myprogs/jskDr-pcfp-39eb310c1b95/build.xml

$HOME/mycode/callChemblScript1.sh #sparse features
$HOME/mycode/callChemblScript2.sh #part of semisparse features
$HOME/mycode/callChemblScript3.sh #part of semisparse features
python3 $HOME/mycode/createRDKit.py #part of semisparse features
#####------------------------------------------------#####



#####convert data to binary format#####

#First make sure the directory structures and all necessary data is there (cluster/training/target files, etc.)
#$HOME/mycode/createSampleIdTable.sh
$HOME/mycode/cppCode/exec/genDirStructure.exec chembl20 chembl20
cp $HOME/mydata/raw/chembl20/train.info $HOME/mydata/trgpred/chembl20/train
cp $HOME/mydata/raw/chembl20/tocompute.info $HOME/mydata/trgpred/chembl20/train
cp $HOME/mydata/raw/chembl20/cl1.info $HOME/mydata/trgpred/chembl20/chemFeatures/cl
unzip $HOME/mydata/raw/chembl20/clusterMinFull.zip -d $HOME/mydata/trgpred/chembl20/chemFeatures/cl #only necessary, if cluster file cl1.info should be reproduced by R script

#get data from csv to sparse format
#takes extremely much memory at least for semisparse and may result in a strange malloc error (on certain platforms only?), but writes out file
python3 $HOME/mycode/csvToSparse.py -propertiesName semisparse
python3 $HOME/mycode/csvToSparse.py -propertiesName toxicophores

$HOME/mycode/cppCode/exec/convertFPFBinary.exec chembl20 chembl20 ECFC4
$HOME/mycode/cppCode/exec/convertFPFBinary.exec chembl20 chembl20 ECFC6_ES
$HOME/mycode/cppCode/exec/convertFPFBinary.exec chembl20 chembl20 DFS8_ES
$HOME/mycode/cppCode/exec/convertPropBinary.exec chembl20 chembl20 dense
$HOME/mycode/cppCode/exec/convertPropBinary.exec chembl20 chembl20 semisparse
$HOME/mycode/cppCode/exec/convertPropBinary.exec chembl20 chembl20 toxicophores
$HOME/mycode/cppCode/exec/convertFPFBinary.exec chembl20 chembl20 semisparse
$HOME/mycode/cppCode/exec/convertFPFBinary.exec chembl20 chembl20 toxicophores
#####-----------------------------#####



#####only for reproducibility of clustering#####
#make sure to have enough processors and memory
$HOME/mycode/cppCode/exec/clusterMinFull.exec chembl20 ECFC4 200

#####--------------------------------------#####




#For methods implemented in Python prepare data first to python format
#requires that data is available in binary (C/C++) format
python3 $HOME/mycode/pythonCode/readDataset.py
python3 $HOME/mycode/pythonCode/readMoleculesDeepchem.py
python3 $HOME/mycode/pythonCode/readMoleculesLSTM.py

python3 $HOME/mycode/pythonCode/reduceDataset.py
python3 $HOME/mycode/pythonCode/reduceMoleculesDeepchem.py
python3 $HOME/mycode/pythonCode/reduceMoleculesLSTM.py
python3 $HOME/mycode/pythonCode/genAdditional.py

python3 $HOME/mycode/pythonCode/genAdditional.py



#Estimate sizes on GPU before training networks
python3 $HOME/mycode/pythonCode/apred/estGPUSize.py
python3 $HOME/gc/pythonCode/apred/estGPUSize.py
python3 $HOME/mycode/pythonCode/weave/estGPUSize.py


#Train inner loop networks
python3 $HOME/mycode/pythonCode/apred/step1.py -maxProc 8 -availableGPUs 0 1 2 3 -saveComputations
python3 $HOME/mycode/pythonCode/gc/step1.py -maxProc 8 -availableGPUs 0 1 2 3 -saveComputations
python3 $HOME/mycode/pythonCode/weave/step1.py -maxProc 8 -availableGPUs 0 1 2 3 -saveComputations

#Train outer loop networks (depend on inner loop networks)
python3 $HOME/mycode/pythonCode/apred/step2.py -maxProc 3 -availableGPUs 0 1 2 -saveComputations
python3 $HOME/mycode/pythonCode/gc/step2.py -maxProc 3 -availableGPUs 0 1 2 -saveComputations
python3 $HOME/mycode/pythonCode/weave/step2.py -maxProc 3 -availableGPUs 0 1 2 -saveComputations
python3 $HOME/mycode/pythonCode/lstm/step2.py -maxProc 3 -availableGPUs 0 1 2 -saveComputations

#Summarize Python results
#analysis/extract/*.py scripts 

#Methods implemented in C/C++

#for svm extremely compute intensive for some assays ==> increase number of processors, below 5 (better are possibly > 100-500)
$HOME/mycode/cppCode/exec/svmHyper.exec $HOME/mycode/cppCode/call/chemblConfSVM.txt 5
$HOME/mycode/cppCode/exec/knnHyper.exec $HOME/mycode/cppCode/call/chemblConfKNN.txt 5
$HOME/mycode/cppCode/exec/nbHyper.exec $HOME/mycode/cppCode/call/chemblConfNB.txt 5

#SEA is split in 3 parts, execute after each other
$HOME/mycode/cppCode/exec/seaHyper1.exec $HOME/mycode/cppCode/call/chemblConfSEA.txt 5
$HOME/mycode/cppCode/exec/seaHyper2.exec $HOME/mycode/cppCode/call/chemblConfSEA.txt 5
$HOME/mycode/cppCode/exec/seaHyper3.exec $HOME/mycode/cppCode/call/chemblConfSEA.txt 5

#Summarize all results
#analysis/summarizeResults.R

#Wilcoxon test comparisons
#analysis/cmpAnalysis.R

#Statistics table
#analysis/stat.py

#Summarize predictions of folds
#analysis/summarizeClusterToPredTab.py

#Comparison virtual assay, surrogate assay
#analysis/surassAnalysis.R

#Assay classes and types:
#assign CHEMBLIds to indivudual classes and types
#generateAnalysedAssayClasses.R
#generateAnalysedAssayTypes.R

#Permutation of samples, such that fold assignment is destroyed:
#perm.py
