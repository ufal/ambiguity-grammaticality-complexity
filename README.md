# Sentence Ambiguity, Grammaticality and Complexity Probes

A project at Institute of Formal and Applied Linguistics at Charles University to be presented at BlackboxNLP @ EMNLP 2022.
Read the paper [here](https://arxiv.org/pdf/2210.06928.pdf) and cite as:

```
@article{bhattacharya2022sentence,
  title={Sentence Ambiguity, Grammaticality and Complexity Probes},
  author={Bhattacharya, Sunit and Zouhar, Vil{\'e}m and Bojar, Ond{\v{r}}ej},
  journal={arXiv preprint arXiv:2210.06928},
  year={2022}
}
```

[Contact the authors](mailto:bhattacharya@ufal.mff.cuni.cz,zouhar@ufal.mff.cuni.cz,bojar@ufal.mff.cuni.cz) for more information about this work :blush:.

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

# Data
The data used in the experiments are publicly available. Some pointers to the data:
* BLiMP: https://github.com/alexwarstadt/blimp
* CoLA: https://nyu-mll.github.io/CoLA/
* Complexity Data: http://www.italianlp.it/resources/corpus-of-sentences-rated-with-human-complexity-judgments/
* MSCOCO: http://www.italianlp.it/resources/corpus-of-sentences-rated-with-human-complexity-judgments/
* Ambiguous COCO: https://www.quest.dcs.shef.ac.uk/wmt17_files_mmt/source_mscoco.task1
