#Copyright (C) 2018 Andreas Mayr
#Licensed under GNU General Public License v3.0 (see http://www.bioinf.jku.at/research/lsc/LICENSE and https://github.com/ml-jku/lsc/blob/master/LICENSE)

import os
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprint
from rdkit import DataStructs
#from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker
from rdkit.Chem import AllChem

ms = [x for x in Chem.SDMolSupplier(os.environ.get("HOME")+"/mydata/raw/chembl20/chembl_20.sdf")]

f = open(os.environ.get("HOME")+"/mydata/raw/chembl20/rdkitFP", 'a', buffering=1024*1024*1024)
for i in range(len(ms)):
  if i%1000==0:
    print(i)
  if not(ms[i] is None):
    chemblId=ms[i].GetProp("chembl_id")
    bitString=Chem.RDKFingerprint(ms[i], maxPath=6).ToBitString()
    f.write(chemblId+" "+bitString+"\n")
f.close()