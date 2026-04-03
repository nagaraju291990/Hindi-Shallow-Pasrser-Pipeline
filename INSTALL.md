
## CRF++ 0.51+
Download: https://drive.google.com/file/d/1ytMgBCdU0E3JihL8K24XbogEmne_OLte/view?usp=drive_link

### Follow below steps to install CRF++

````
cd CRF++-0.51
./configure
make
$make install
````

### Check if the installtion was successful using below command

````
crf_test --version
````

**Note if you get an error as below**
if **libcrfpp.so.0** not found after CRF installation than execute the below command

````
ln -s /usr/local/lib/libcrfpp.so.0 /usr/lib/libcrfpp.so.0
````

## Morph Analyszer dependencies

````
sudo yum install gcc gdbm libgdbm-dev glib-2.0 libglib2.0-dev pkg-config
````

## for ubuntu 20+ to install glib2.0
````
sudo apt-get install libgtk2.0-dev
````

*For ubuntu use following command*

````
sudo apt-get install gcc libgdbm-dev libglib2.0-dev g++ pkg-config
````

## Install


# install morph
cd hin-morph

````
export setu=$PWD
cd src/sl/morph/hin/
make clean
make compile
make install
````

## to test pipeline upto morph,pos, chunk
If morph and crf are installed properly below step should run without fail: input folder contain files with sentences
````
sh bash run_pos_chunk_morph_models_for_hindi.sh Input_Folder Features_Folder Output_Folder model-lcat.m model-gender.m model-number.m model-person.m model-case.m model-vib.m model-pos.m model-chunk.m
````

## then follow clause boundary installation steps
[CLause boundary README](path)

## to run full pipeline
python3 run_shallow_parser.py <input_dir> <output_dir>