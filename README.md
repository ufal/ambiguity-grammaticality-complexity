# Representation Probing


## Extracting the representations
To extract the representations from the data, run: 

./src/patches_sunit/generate_representations.sh [location-of-source-data] (optional: [location-of-virtual-environment])

### Example:    
./src/patches_sunit/generate_representations.sh ../folder_where_data_is_stored/data 

./src/patches_sunit/generate_representations.sh ../folder_where_data_is_stored/data ../location_of_virtualenv 

## Extracting the tf-idf baselines

To extract the tf-idf baselines, run:

./src/patches_sunit/baseline_generator.sh (optional: [location-of-virtual-environment])

### Example:

./src/patches_sunit/baseline_generator.sh 

./src/patches_sunit/baseline_generator.sh ../location_of_virtualenv 

## Running MLP classifier on the extracted representations:

./src/patches_sunit/generate_rep_plots.sh (optional: [location-of-virtual-environment]) 

### Example:    
./src/patches_sunit/generate_rep_plots.sh  

./src/patches_sunit/generate_rep_plots.sh ../location_of_virtualenv 

