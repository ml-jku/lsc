--Copyright (C) 2018 Andreas Mayr
--Licensed under GNU General Public License v3.0 (see http://www.bioinf.jku.at/research/lsc/LICENSE and https://github.com/ml-jku/lsc/blob/master/LICENSE)

use chembl_20;

SELECT
ass.chembl_id,
pc.protein_class_id,
pc.parent_id,
pc.pref_name
FROM assays ass
JOIN target_dictionary td ON td.tid = ass.tid
JOIN target_components tc ON tc.tid = td.tid
JOIN component_sequences cs ON cs.component_id = tc.component_id
JOIN component_class cc ON cc.component_id = cs.component_id
JOIN protein_classification pc ON pc.protein_class_id = cc.protein_class_id
INTO OUTFILE 'trgInfo.txt'

SELECT *
FROM protein_classification pc
INTO OUTFILE 'pcInfo.txt'



SELECT
ass.assay_id,
ass.chembl_id,
ass.assay_type
FROM assays ass
INTO OUTFILE 'assayInfo.txt'

SELECT *
FROM assay_type
INTO OUTFILE 'assayTypeInfo.txt'


