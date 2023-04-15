import os
import pandas as pd
from variables import train_dataset_path, cleaned_train_dataset_path

if not os.path.isfile(train_dataset_path):
    print('First download and unzip data from: https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis')
    quit(1)

train_dataset = pd.read_csv(train_dataset_path).rename(columns={'2401': 'id', 'Borderlands': 'entity', 'Positive': 'sentiment', 'im getting on borderlands and i will murder you all ,': 'tweet'})

train_dataset.to_csv(cleaned_train_dataset_path)
print('Training dataset cleaned and saved.')