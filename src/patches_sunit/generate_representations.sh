#!/bin/bash

[ "$1" == "-h" -o "$1" == "-help" ] && echo "
Run the script with following arguments:

a. <path_of_data> : This is a compulsory arument that specifies where the representation data would be generated from

b. <path_of_virtual_environment> : This is an optional argument that specifies the path of the virtual environment

Run the script as:

generate_representations.sh <path_of_data> <path_of_virtual_environment>
" && exit

HOME=$(pwd)

mkdir $HOME/huggingface_cache


# Locating the data
if [ "$1" != "-h" -o "$1" != "-help" ]
then 
    data=$1
fi
[ ! -d $data ] && echo "Data Directory $data DOES NOT exist. Terminating" && exit


# Locating the virtual environment
if [ -z "$2" ]
  then
    virtual=/home/bhattacharya/personal_work_troja/MML/virtual
    [ -d $virtual ] && source $virtual/bin/activate    
else
    virtual=$2
    source $virtual/bin/activate
fi

python3 $HOME/src/representation_generator.py --data $data

rm -rf $HOME/.cache
rm -rf $HOME/huggingface_cache
