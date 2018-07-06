#!/bin/bash

#Copyright (C) 2018 Andreas Mayr
#Licensed under GNU General Public License v3.0 (see http://www.bioinf.jku.at/research/lsc/LICENSE and https://github.com/ml-jku/lsc/blob/master/LICENSE)

optionAlgorithm=$1
optionAtomType=$2
optionInputFile=$3
optionOutputDir=$4
optionLabel=$5
optionOutputFormat=$6

echo $1
echo $2
echo $3
echo $4
echo $5
echo $6
echo $7

nrLines=`wc -l $optionInputFile|cut -f 1 -d " "`

cp $optionInputFile "$optionOutputDir/inputFile.sdf"
inputFile="$optionOutputDir/inputFile.sdf"
fileNr=1
outputFile="$optionOutputDir/myout$fileNr.res"
java -Xmx64g -XX:hashCode=5 -jar $HOME/myprogs/jCMapperCLI.jar -c $optionAlgorithm -hs 2147483647 -a $optionAtomType -f $inputFile -o $outputFile  -l $optionLabel -ff $optionOutputFormat
cnt=1

oldFile=$outputFile
fileNr=`expr $fileNr + 1`
outputFile="$optionOutputDir/myout$fileNr.res"
newInchi=`tail -n2 $oldFile|head -n1|cut -f 1`

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
    java -Xmx64g -XX:hashCode=5 -jar $HOME/myprogs/jCMapperCLI.jar -c $optionAlgorithm -hs 2147483647 -a $optionAtomType -f $inputFile -o $outputFile -l $optionLabel -ff $optionOutputFormat
    cnt=`expr $cnt + 1`
  done
  oldFile=$outputFile
  fileNr=`expr $fileNr + 1`
  outputFile="$optionOutputDir/myout$fileNr.res"
  newInchi=`tail -n2 $oldFile|head -n1|cut -f 1`
done

