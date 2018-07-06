#Copyright (C) 2018 Andreas Mayr
#Licensed under GNU General Public License v3.0 (see http://www.bioinf.jku.at/research/lsc/LICENSE and https://github.com/ml-jku/lsc/blob/master/LICENSE)

import pandas as pd
import numpy as np
import os
import functools
import argparse



parser = argparse.ArgumentParser()
parser.add_argument("-propertiesName", help="propertiesName to convert from csv to sparse", type=str)
args = parser.parse_args()



rawDataFoldername="chembl20"
propertiesName=args.propertiesName
#propertiesName="toxicophores"
#propertiesName="semisparse"

rawDataPathname=os.environ.get("HOME")+"/mydata/raw"
inputDataDir=os.path.join(rawDataPathname, rawDataFoldername)

propertiesFileNameInp=os.path.join(inputDataDir, propertiesName+".csv")
propertiesFileNameOutp=os.path.join(inputDataDir, propertiesName+".fpf")



def compare(item1, item2):
  if item1[0]<item1[0]:
    return -1
  elif item1[0]>item1[0]:
    return 1
  if item2[1]<item2[1]:
    return -1
  elif item2[1]>item2[1]:
    return 1
  return 0

#mypd=pd.read_table(propertiesFileNameInp, sep=",")
#mypd=mypd.set_index("Unnamed: 0")
mypd=pd.read_table(propertiesFileNameInp, sep=",", index_col=0)

#bigz=np.where(mypd>0.0)
bigz=np.where(mypd!=0.0)

myind=list(zip(bigz[0], bigz[1]))
myind.sort(key=functools.cmp_to_key(compare))
myindArr=np.vstack(myind)
mysplitArr=np.split(myindArr, np.where(np.diff(myindArr[:,0]))[0]+1)

mydictArr1={mysplitArr[i][0,0] : mysplitArr[i][:,1].copy() for i in range(len(mysplitArr)) }
mykeys=np.array(list(mydictArr1.keys()))
mydictArr2={i : np.array(mypd.iloc[i, mydictArr1[i]].tolist()) for i in mykeys }

cpNames=mypd.index.values[mykeys]
myArr1=[mydictArr1[mykey] for mykey in mykeys]
myArr2=[mydictArr2[mykey] for mykey in mykeys]
myArr=list(zip(cpNames, myArr1, myArr2))

mystrs=[myArr[i][0]+" "+" ".join((pd.Series(myArr[i][1]).astype(np.int64).astype(np.str)+":"+pd.Series(myArr[i][2]).astype(np.str)).tolist())+"\n" for i in range(len(myArr))]

myf=open(propertiesFileNameOutp, "w")
myf.writelines(mystrs)
myf.close()


