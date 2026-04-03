$setu/bin/sl/morph/hin/morph_hin.exe --logfilepath morph.log --pdgmfilepath $setu/data_bin/sl/morph/hin/ --uwordpath $setu/data_bin/sl/morph/hin/dict_final --dictfilepath $setu/data_bin/sl/morph/hin/dict/  -ULDWH --inputfile $1 --outputfile morph_output

# uncomment below line if want to print nukta in root word

python $setu/bin/sl/morph/hin/nukta-adder.py  morph_output | perl $setu/bin/sl/morph/hin/adj-gen-1.1/adj_gen.pl

# uncomment below line and comment above line if do not want nukta in root word

#cat morph_output
