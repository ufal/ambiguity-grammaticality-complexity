# Ambiguity, Complexity and Grammaticality in Pretrained Language Models

A project at Institute of Formal and Applied Linguistics at Charles University.
Contact the authors for more info.

## Extracting the representations
To extract the representations from the data, run: 

```
./src/patches_sunit/generate_representations.sh [location-of-source-data] (optional: [location-of-virtual-environment])
```

For example:

```
./src/patches_sunit/generate_representations.sh ../folder_where_data_is_stored/data 
./src/patches_sunit/generate_representations.sh ../folder_where_data_is_stored/data ../location_of_virtualenv 
```

## TF-IDF baseline:

To run the TF-IDF baselines, run:

```
./src/patches_sunit/baseline_generator.sh (optional: [location-of-virtual-environment])
```

For example:

```
./src/patches_sunit/baseline_generator.sh 
./src/patches_sunit/baseline_generator.sh ../location_of_virtualenv 
```

## MLP classifier on the extracted representations:

```
./src/patches_sunit/generate_rep_plots.sh (optional: [location-of-virtual-environment]) 
```

For example:

```
./src/patches_sunit/generate_rep_plots.sh  
./src/patches_sunit/generate_rep_plots.sh ../location_of_virtualenv 
```
