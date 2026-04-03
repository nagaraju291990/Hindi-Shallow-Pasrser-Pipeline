export LC_ALL=C
sh info.sh
sh lext.sh
sh main2.sh

rm -f ../../../../../data_bin/sl/morph/hin/suff 
rm -f ../../../../../data_bin/sl/morph/hin/uword.dbm
sh mk_dbm_suffix
sh mk_dbm_suff
sh mk_dbm_uword
chmod 644 ../../../../../data_bin/sl/morph/hin/suff  ../../../../../data_bin/sl/morph/hin/uword.dbm
