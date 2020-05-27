# Representing Schema Structure with Graph Neural Networks for Text-to-SQL Parsing

Author implementation of this [ACL 2019 paper](https://arxiv.org/abs/1905.06241).

Please also see the [follow-up repository](https://github.com/benbogin/spider-schema-gnn-global) with improved results, for this [EMNLP paper](https://www.aclweb.org/anthology/D19-1378.pdf).

## Install & Configure

1. Install pytorch version 1.0.1.post2 that fits your CUDA version 
   
   (this repository should probably work with the latest pytorch version, but wasn't tested for it. If you use another version, you'll need to also update the versions of packages in `requirements.txt`)
    ```
    pip install https://download.pytorch.org/whl/cu100/torch-1.0.1.post2-cp37-cp37m-linux_x86_64.whl # CUDA 10.0 build
    ```
    
2. Install the rest of required packages
    ```
    pip install -r requirements.txt
    ```
    
3. Run this command to install NLTK punkt.
```
python -c "import nltk; nltk.download('punkt')"
```

4. Download the dataset from the [official Spider dataset website](https://yale-lily.github.io/spider)

5. Edit the config file `train_configs/defaults.jsonnet` to update the location of the dataset:
```
local dataset_path = "dataset/";
```

## Training

1. Use the following AllenNLP command to train:
```
allennlp train train_configs/defaults.jsonnet -s experiments/name_of_experiment \
--include-package dataset_readers.spider \ 
--include-package models.semantic_parsing.spider_parser
``` 

First time loading of the dataset might take a while (a few hours) since the model first loads values from tables and calculates similarity features with the relevant question. It will then be cached for subsequent runs.

You should get results similar to the following (the `sql_match` is the one measured in the official evaluation test):
```
  "best_validation__match/exact_match": 0.3715686274509804,
  "best_validation_sql_match": 0.47549019607843135,
  "best_validation__others/action_similarity": 0.5731271471206189,
  "best_validation__match/match_single": 0.6254612546125461,
  "best_validation__match/match_hard": 0.3054393305439331,
  "best_validation_beam_hit": 0.6070588235294118,
  "best_validation_loss": 7.383035182952881
  "best_epoch": 32
```

Note that the hyper-parameters used in `defaults.jsonnet` are different than those mentioned in the paper
(most importantly, 3 timesteps are used instead of 2), thanks to the [following contribution from @wlhgtc](https://github.com/benbogin/spider-schema-gnn/pull/13).
The original training config file is still available in `train_configs/paper_Defaults.jsonnet`.

## Inference

Use the following AllenNLP command to output a file with the predicted queries:

```
allennlp predict experiments/name_of_experiment dataset/dev.json \
--predictor spider \
--use-dataset-reader \
--cuda-device=0 \
--output-file experiments/name_of_experiment/prediction.sql \
--silent \
--include-package models.semantic_parsing.spider_parser \
--include-package dataset_readers.spider \
--include-package predictors.spider_predictor \
--weights-file experiments/name_of_experiment/best.th \
-o "{\"dataset_reader\":{\"keep_if_unparsable\":true}}"
```
