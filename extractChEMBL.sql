--Copyright (C) 2018 Andreas Mayr
--Licensed under GNU General Public License v3.0 (see http://www.bioinf.jku.at/research/lsc/LICENSE and https://github.com/ml-jku/lsc/blob/master/LICENSE)

use chembl_20;

SELECT
md.chembl_id,
ass.chembl_id,
act.standard_relation,
act.standard_value,
act.standard_units,
act.standard_type,
act.activity_comment,
act.doc_id,
td.tid,
tt.parent_type,
tt.target_type,
ass.confidence_score
FROM target_dictionary td
JOIN target_type tt ON tt.target_type = td.target_type
JOIN assays ass ON td.tid = ass.tid
JOIN activities act ON ass.assay_id = act.assay_id
JOIN molecule_dictionary md ON act.molregno = md.molregno
INTO OUTFILE 'chembl.txt'
