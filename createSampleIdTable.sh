#!/bin/bash

#Copyright (C) 2018 Andreas Mayr
#Licensed under GNU General Public License v3.0 (see http://www.bioinf.jku.at/research/lsc/LICENSE and https://github.com/ml-jku/lsc/blob/master/LICENSE)

cat $HOME/mydata/raw/chembl20/chembl_20.sdf|grep -A1 chembl_id|grep -v chembl_id|grep -v \\-\\- > $HOME/mydata/raw/chembl20/SampleIdTable.txt
