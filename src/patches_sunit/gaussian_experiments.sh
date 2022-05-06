#!/bin/bash
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

python3 src/gaussian_approx.py
