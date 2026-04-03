# How to run this script: sh run_pos_chunk_morph_models_for_hindi.sh <input_folder> <features_folder> <output_folder> <lcat_model_path> <gender_model_path> <number_model_path> <person_model_path> <case_model_path> <vib_model_path> <pos_model_path> <chunk_model_path>
# For the input_folder, features_folder and output_folder, provide the absolute paths. For the model paths, provide the relative paths from the current directory.
input_folder=$1
features_folder=$2
output_folder=$3
lcat_model_path=$4
gender_model_path=$5
number_model_path=$6
person_model_path=$7
case_model_path=$8
vib_model_path=$9
pos_model_path=${10}
chunk_model_path=${11}
if [ ! -d $features_folder ];then
	mkdir $features_folder
fi
if [ ! -d $output_folder ];then
	mkdir $output_folder
fi
for fl in $(ls $input_folder);do
	file_name=$(echo $fl | sed 's/\.txt//')
	file_path=$input_folder"/"$fl
	token_features_path=$features_folder"/"$file_name"-tokens+features.txt"
	token_features_pos_path=$features_folder"/"$file_name"-tokens+features+pos.txt"
	token_path=$features_folder"/"$file_name"-tokens.txt"
	pos_path=$features_folder"/"$file_name"-pos.txt"
	token_features_lcat_path=$features_folder"/"$file_name"-tokens+features+lcat.txt"
	lcat_path=$features_folder"/"$file_name"-lcat.txt"
	token_pos_path=$features_folder"/"$file_name"-token-pos.txt"
	token_pos_chunk_path=$features_folder"/"$file_name"-token-pos-chunk.txt"
	token_features_gender_path=$features_folder"/"$file_name"-tokens+features+gender.txt"
	gender_path=$features_folder"/"$file_name"-gender.txt"
	token_features_number_path=$features_folder"/"$file_name"-tokens+features+number.txt"
	number_path=$features_folder"/"$file_name"-number.txt"
	token_features_person_path=$features_folder"/"$file_name"-tokens+features+person.txt"
	person_path=$features_folder"/"$file_name"-person.txt"
	token_features_case_path=$features_folder"/"$file_name"-tokens+features+case.txt"
	case_path=$features_folder"/"$file_name"-case.txt"
	token_features_vib_path=$features_folder"/"$file_name"-tokens+features+vib.txt"
	vib_path=$features_folder"/"$file_name"-vib.txt"
	token_pos_chunk_morph_path=$features_folder"/"$file_name"-token-pos-chunk-all-morph.txt"
	python3 create_features_for_testing_crf_for_raw_sentences.py --input $file_path --output $token_features_path
	cut -f1 $token_features_path > $token_path
	crf_test -m $pos_model_path $token_features_path > $token_features_pos_path
	cut -f14 $token_features_pos_path > $pos_path
	paste $token_path $pos_path > $token_pos_path
	crf_test -m $chunk_model_path $token_pos_path > $token_pos_chunk_path
	crf_test -m $lcat_model_path $token_features_path > $token_features_lcat_path
	cut -f14 $token_features_lcat_path > $lcat_path
	crf_test -m $gender_model_path $token_features_path > $token_features_gender_path
	cut -f14 $token_features_gender_path > $gender_path
	crf_test -m $number_model_path $token_features_path > $token_features_number_path
	cut -f14 $token_features_number_path > $number_path
	crf_test -m $person_model_path $token_features_path > $token_features_person_path
	cut -f14 $token_features_person_path > $person_path
	crf_test -m $case_model_path $token_features_path > $token_features_case_path
	cut -f14 $token_features_case_path > $case_path
	crf_test -m $vib_model_path $token_features_path > $token_features_vib_path
	cut -f14 $token_features_vib_path > $vib_path
	paste $token_pos_chunk_path $lcat_path $gender_path $number_path $person_path $case_path $vib_path > $token_pos_chunk_morph_path
	mv $token_pos_chunk_morph_path $output_folder/$file_name"-output.txt"
done
rm -rf $features_folder
