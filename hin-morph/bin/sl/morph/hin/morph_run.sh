$setu/bin/sl/morph/hin/morph_hin.exe --logfilepath morph.log --pdgmfilepath $setu/data_bin/sl/morph/hin/ --uwordpath $setu/data_bin/sl/morph/hin/dict_final --dictfilepath $setu/data_bin/sl/morph/hin/dict/  -ULDWH --inputfile $1 --outputfile morph_output

# uncomment below line if want to print nukta in root word


cat morph_output
