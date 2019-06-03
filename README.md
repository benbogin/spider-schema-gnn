# Representing Schema Structure with Graph Neural Networks for Text-to-SQL Parsing

Author implementation of this [ACL 2019 paper](https://arxiv.org/abs/1905.06241).

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
--weights-file experiments/name_of_experiment/best.th
```

Trained model files will be uploaded here soon.