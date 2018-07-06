#!/bin/bash

#Copyright (C) 2018 Andreas Mayr
#Licensed under GNU General Public License v3.0 (see http://www.bioinf.jku.at/research/lsc/LICENSE and https://github.com/ml-jku/lsc/blob/master/LICENSE)



rawDataDir=chembl20
sdfFile=chembl_20.sdf



mkdir $HOME/mydata/raw/$rawDataDir/SHED
$HOME/mycode/chemblScript2.sh SHED ELEMENT_SYMBOL $HOME/mydata/raw/$rawDataDir/$sdfFile $HOME/mydata/raw/$rawDataDir/SHED chembl_id STRING_PATTERNS

mkdir $HOME/mydata/raw/$rawDataDir/CATS2D
$HOME/mycode/chemblScript2.sh CATS2D ELEMENT_SYMBOL $HOME/mydata/raw/$rawDataDir/$sdfFile $HOME/mydata/raw/$rawDataDir/CATS2D chembl_id STRING_PATTERNS





dirName=$HOME/mydata/raw/$rawDataDir/SHED
outFile=SHED_ES.fpf

rm $dirName/../$outFile
for i in `ls $dirName/myout*.res`; do
  echo $i
  head -n`wc -l $i |cut -f 1 -d " "` $i >> $dirName/../$outFile
done



dirName=$HOME/mydata/raw/$rawDataDir/CATS2D
outFile=CATS2D_ES.fpf

rm $dirName/../$outFile
for i in `ls $dirName/myout*.res`; do
  echo $i
  head -n`wc -l $i |cut -f 1 -d " "` $i >> $dirName/../$outFile
done
