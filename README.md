# Representing Schema Structure with Graph Neural Networks for Text-to-SQL Parsing

Author implementation of this [ACL 2019 paper](https://arxiv.org/abs/1905.06241).

## Install & Configure

1. Install the required packages
    ```
    pip install -r requirements.txt
    ```

2. Download the dataset from the official Spider website

3. Edit the config file `train_configs/defaults.jsonnet` to update the location of the dataset:
```
local dataset_path = "dataset/";
```

## Training

1. Use the following AllenNLP command to train:
```
train train_configs/defaults.jsonnet -s experiments/name_of_experiment \
--include-package dataset_readers.spider \ 
--include-package models.semantic_parsing.spider_parser
``` 

First time loading of the dataset might take a while (a few hours) since the model first loads values from tables and calculates similarity features with the relevant question. It will then be cached.

You should get results similar to the following:
```
  "best_validation__match/exact_match": 0.31431334622823986,
  "best_validation_sql_match": 0.41876208897485495,
  "best_validation__others/action_similarity": 0.5249016759914995,
  "best_validation__match/match_single": 0.5359712230215827,
  "best_validation__match/match_hard": 0.2824267782426778,
  "best_validation_beam_hit": 0.5513937282229965,
  "best_validation_loss": 7.764545440673828
```

## Inference

Trained model files will be uploaded here soon.