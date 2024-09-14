# A pytorch implementation for "PEACE: Probability-enhanced Relational Learning with Invariance for Test-time Adaptive Retrieval"

## ENVIRONMENTS

1. pytorch 2.4.0

2. loguru

3. scikit-learn

## DATASETS
[OFFICE-HOME]

## Before running the code, the dataset path has to be modified based on the dataset's path in lines 86 and 87 in run.py and lines 188 and 196 in officehome.py.

How to train a hashing model:

python run.py --train

It will run the test-time adaptive retrieval task on Real_World → Art from dataset OFFICE-HOME with 64-bit hash codes.

To change the task and hash code length, the corresponding arguments need to be modified in run.py.
