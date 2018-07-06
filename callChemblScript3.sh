#!/bin/bash

#Copyright (C) 2018 Andreas Mayr
#Licensed under GNU General Public License v3.0 (see http://www.bioinf.jku.at/research/lsc/LICENSE and https://github.com/ml-jku/lsc/blob/master/LICENSE)



rawDataDir=chembl20
sdfFile=chembl_20.sdf



mkdir $HOME/mydata/raw/$rawDataDir/pubchem
$HOME/mycode/chemblScript3.sh $HOME/mydata/raw/$rawDataDir/$sdfFile $HOME/mydata/raw/$rawDataDir/pubchem





dirName=$HOME/mydata/raw/$rawDataDir/pubchem
outFile=pubchem.fpf

rm $dirName/../$outFile
for i in `ls $dirName/myout*.res`; do
  echo $i
  head -n`wc -l $i |cut -f 1 -d " "` $i >> $dirName/../$outFile
done
