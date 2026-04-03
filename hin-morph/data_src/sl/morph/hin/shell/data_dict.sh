rm -f ../../../../../data_bin/sl/morph/hin/dict_final
sort ../dict/dict.final | ./join_dict.pl > dict_final.dbm.c
rm -f dict_final
./create_dbm_dict.pl dict_final < dict_final.dbm.c
rm -f dict_final.dbm.c
chmod 644 dict_final
cp ../dict/prop_noun_lex.dic ../../../../../data_bin/sl/morph/hin/dict/
cp ../dict/uword.dic ../../../../../data_bin/sl/morph/hin/dict/
mv dict_final ../../../../../data_bin/sl/morph/hin/
