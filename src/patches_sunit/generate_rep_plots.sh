#!/bin/bash
[ "$1" == "-h" -o "$1" == "-help" ] && echo "
Run the script with following arguments:

a. <path_of_virtual_environment> : This is an optional argument that specifies the path of the virtual environment

Run the script as:

generate_rep_plots.sh <path_of_virtual_environment>
" && exit

HOME=$(pwd)

# Locating the virtual environment
if [ -z "$1" ]
  then
    virtual=/home/bhattacharya/personal_work_troja/MML/virtual
    [ -d $virtual ] && source $virtual/bin/activate    
else
    virtual=$1
    source $virtual/bin/activate
fi



Amb_LOC=$HOME/Representations/Ambiguity
for d in $Amb_LOC/*; do
    echo $d
    # ./src/classification_mlp.py --data $d 
    ./src/fig_classification.py --data $d
done

GRAM_LOC=$HOME/Representations/Grammaticality
for d in $GRAM_LOC/*; do
    n_loc=$d
    bert_repr=$n_loc/BERT.pkl
    if [ -f "$bert_repr" ]; then
        echo $n_loc
        # ./src/classification_mlp.py --data $n_loc 
        ./src/fig_classification.py --data $n_loc
    else
        for case in $n_loc/*/; do
            echo $case
            # ./src/classification_mlp.py --data $case 
            ./src/fig_classification.py --data $case
        done
    fi
done

COMPLEXITY_LOC=$HOME/Representations/Complexity
for d in $COMPLEXITY_LOC/*; do
    echo $d
    # ./src/classification_mlp.py --data $d 
    ./src/fig_classification.py --data $d
done


# rm -rf .cache
# rm .python_history

