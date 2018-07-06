#!/bin/bash

#Copyright (C) 2018 Andreas Mayr
#Licensed under GNU General Public License v3.0 (see http://www.bioinf.jku.at/research/lsc/LICENSE and https://github.com/ml-jku/lsc/blob/master/LICENSE)



rawDataDir=chembl20
sdfFile=chembl_20.sdf



mkdir $HOME/mydata/raw/$rawDataDir/ECFC4
$HOME/mycode/chemblScript1.sh ECFC 2 DAYLIGHT_INVARIANT_RING $HOME/mydata/raw/$rawDataDir/$sdfFile $HOME/mydata/raw/$rawDataDir/ECFC4 chembl_id LIBSVM_SPARSE_FREQUENCY

mkdir $HOME/mydata/raw/$rawDataDir/DFS8
$HOME/mycode/chemblScript1.sh DFS 8 ELEMENT_SYMBOL $HOME/mydata/raw/$rawDataDir/$sdfFile $HOME/mydata/raw/$rawDataDir/DFS8 chembl_id LIBSVM_SPARSE_FREQUENCY

mkdir $HOME/mydata/raw/$rawDataDir/ECFC6
$HOME/mycode/chemblScript1.sh ECFC 3 ELEMENT_SYMBOL $HOME/mydata/raw/$rawDataDir/$sdfFile $HOME/mydata/raw/$rawDataDir/ECFC6 chembl_id LIBSVM_SPARSE_FREQUENCY



dirName=$HOME/mydata/raw/$rawDataDir/ECFC4
outFile=ECFC4.fpf

rm $dirName/../$outFile
for i in `ls $dirName/myout*.res`; do
  echo $i
  head -n`wc -l $i |cut -f 1 -d " "` $i >> $dirName/../$outFile
done



dirName=$HOME/mydata/raw/$rawDataDir/DFS8
outFile=DFS8_ES.fpf

rm $dirName/../$outFile
for i in `ls $dirName/myout*.res`; do
  echo $i
  head -n`wc -l $i |cut -f 1 -d " "` $i >> $dirName/../$outFile
done



dirName=$HOME/mydata/raw/$rawDataDir/ECFC6
outFile=ECFC6_ES.fpf

rm $dirName/../$outFile
for i in `ls $dirName/myout*.res`; do
  echo $i
  head -n`wc -l $i |cut -f 1 -d " "` $i >> $dirName/../$outFile
done
