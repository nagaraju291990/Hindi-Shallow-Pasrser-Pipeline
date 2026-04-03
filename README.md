- This repository contains code for running CRF based POS, Chunk, and Morphological models for Hindi language. The code is written in Python and uses the CRF++ toolkit (https://taku910.github.io/crfpp/) for training and testing the models. The code is organized in a way that it can be easily adapted for other languages as well. The main script to run the models is `run_pos_chunk_morph_models_for_hindi.sh` which takes the input folder containing raw sentences, the features folder where the intermediate files will be stored, and the output folder where the final output will be stored. The script also takes the paths of the trained models for POS, Chunk, and Morphological features as arguments.
- The script processes each file in the input folder, creates the necessary features for testing the CRF models, and then runs the models to get the POS, Chunk, and Morphological features. Finally, it combines all the features and stores the output in the specified output folder.
- If the output of any model is "unk", you can replace it with blank or "".
- How to run the script: 
 1. Make sure you have the CRF++ toolkit installed and properly set up in your environment.
 2. Use the saved POS, Chunk, and Morphological models and store them in the current directory.
 3. Prepare the input folder with raw sentences that you want to process.
 4. Run the script using the following command: 
    ```
    bash run_pos_chunk_morph_models_for_hindi.sh Input_Folder Features_Folder Output_Folder model-lcat.m model-gender.m model-number.m model-person.m model-case.m model-vib.m model-pos.m model-chunk.m
    ```
 5. For the input_folder, features_folder and output_folder, provide the absolute paths. For the model paths, provide the relative paths from the current directory.
 6. The output will be in conll format where the columns will be token, POS, Chunk, lcat, gender, number, person, case and vibhakti. The output files will be stored in the output folder with the name `<input_file_name>-output.txt`.
- To cite this work, please use the following BibTeX entry:
```
@article{mishra2024multi,
  title={Multi task learning based shallow parsing for indian languages},
  author={Mishra, Pruthwik and Mujadia, Vandan and Sharma, Dipti Misra},
  journal={ACM Transactions on Asian and Low-Resource Language Information Processing},
  volume={23},
  number={9},
  pages={1--18},
  year={2024},
  url={https://dl.acm.org/doi/10.1145/3664620}
  publisher={ACM New York, NY}
}
```# Hindi-Shallow-Pasrser-Pipeline
