# how to run the code
# python3 create_features_for_testing_crf_for_raw_sentences.py --input Input_Folder_Path --output Output_File_Path
# works at both file and folder level. If the input is a folder, it will read all the files in the folder and create features for each file and save it in the output folder with the same name as the input file. If the input is a file, it will create features for that file and save it in the output file path provided.
import argparse
import os


def read_files_from_folder_find_features_and_write_to_files(input_folder_path, output_folder_path):
    '''
    :param input_folder_path: Input Folder Path containing all the files which you want to tag
    :param output_folder_path: Output Folder Path where the features will be saved
    :return None
    '''
    if os.path.isdir(input_folder_path):
        for root, dirs, files in os.walk(input_folder_path):
            features_string = ''
            for fl in files:
                input_file_path = os.path.join(root, fl)
                lines_read = open(input_file_path, 'r', encoding='utf-8').readlines()
                features_string = find_features_from_sentences(lines_read)
                output_file_path = os.path.join(output_folder_path, fl)
                write_file(output_file_path, features_string)
    else:
        features_string = ''
        lines_read = open(input_folder_path, 'r', encoding='utf-8').readlines()
        features_string = find_features_from_sentences(lines_read)
        write_file(output_folder_path, features_string)


def find_features_from_sentences(sentences):
    '''
    :param sentences: Sentences read from file
    :return features: Features of all tokens for each sentence combined for all the sentences
    '''
    prefix_len = 4
    suffix_len = 7
    features = ''
    for sentence in sentences:
        sentence_features = ''
        if sentence.strip():
            tokens_split = [token for token in sentence.strip().split() if token.strip()]
            for token in tokens_split:
                sentence_features += token + '\t'
                for i in range(1, prefix_len + 1):
                    sentence_features += affix_feats(token, i, 0) + '\t'
                for i in range(1, suffix_len + 1):
                    sentence_features += affix_feats(token, i, 1) + '\t'
                sentence_features = sentence_features + 'LESS\n' if len(token) <= 4 else sentence_features + 'MORE\n'
        if sentence_features.strip():
            features += sentence_features + '\n'
    return features


def affix_feats(token, length, type_aff):
    '''
    :param line: extract the token and its corresponding suffix list depending on its length
    :param token: the token in the line
    :param length: length of affix
    :param type: 0 for prefix and 1 for suffix
    :return suffix: returns the suffix
    '''
    if len(token) < length:
        return 'NULL'
    else:
        if type_aff == 0:
            return token[:length]
        else:
            return token[len(token) - length:]


def write_file(out_path, data):
    '''
    :param out_path: Enter the path of the output file
    :param data: Enter the token features of sentence separated by a blank line
    :return: None
    '''
    with open(out_path, 'w', encoding='utf-8') as fout:
        fout.write(data + '\n')


def main():
    '''main function to run the code.'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', dest='inp', help="Add the input path from where tokens and its features will be extracted")
    parser.add_argument('--output', dest='out', help="Add the output file where the features will be saved")
    args = parser.parse_args()
    read_files_from_folder_find_features_and_write_to_files(args.inp, args.out)


if __name__ == '__main__':
    main()
