#!/bin/bash

#Copyright (C) 2018 Andreas Mayr
#Licensed under GNU General Public License v3.0 (see http://www.bioinf.jku.at/research/lsc/LICENSE and https://github.com/ml-jku/lsc/blob/master/LICENSE)



optionInputFile=$1
optionOutputDir=$2

echo $1
echo $2

nrLines=`wc -l $optionInputFile|cut -f 1 -d " "`

cp $optionInputFile "$optionOutputDir/inputFile.sdf"
inputFile="$optionOutputDir/inputFile.sdf"
fileNr=1
outputFile="$optionOutputDir/myout$fileNr.res"
java -Xmx512m -classpath $HOME/myprogs/jskDr-pcfp-39eb310c1b95/build:$HOME/myprogs/jskDr-pcfp-39eb310c1b95/lib/jchem.jar tripod.fingerprint.FPTest $inputFile > $outputFile
cnt=1

oldFile=$outputFile
fileNr=`expr $fileNr + 1`
outputFile="$optionOutputDir/myout$fileNr.res"
newInchi=`tail -n1 $oldFile|cut -f 1 -d " "`

while [ 1 -gt 0 ]; do
  echo $outputFile
  cnt=1
  while [ \( ! -e "$outputFile" \) -o \( ! -s "$outputFile" \) ]; do
    echo $cnt
    cat -n $optionInputFile | grep -E "($newInchi|\\$\\$\\$\\$)" | grep -A$cnt $newInchi|tail -n1|cut -f1 > $optionOutputDir/lineFile
    myLine=`cat $optionOutputDir/lineFile`
    if [ -z "$myLine" ]; then
      exit
    fi
    catLines=`expr $nrLines - $myLine`
    echo $catLines
    if [ $catLines -lt 1 ]; then
      exit
    fi
    tail -n $catLines $optionInputFile > $inputFile
    java -Xmx512m -classpath $HOME/myprogs/jskDr-pcfp-39eb310c1b95/build:$HOME/myprogs/jskDr-pcfp-39eb310c1b95/lib/jchem.jar tripod.fingerprint.FPTest $inputFile > $outputFile
    cnt=`expr $cnt + 1`
  done
  oldFile=$outputFile
  fileNr=`expr $fileNr + 1`
  outputFile="$optionOutputDir/myout$fileNr.res"
  newInchi=`tail -n1 $oldFile|cut -f 1 -d " "`
done
