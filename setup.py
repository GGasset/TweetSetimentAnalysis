import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from variables import train_dataset_path, cleaned_train_dataset_path

def main():
    if not os.path.isfile(train_dataset_path):
        print('First download and unzip data from: https://www.kaggle.com/datasets/kazanova/sentiment140')
        print('Make sure to rename the uncompressed dataset to', train_dataset_path)
        quit(1)

    columns_data: dict = {'sentiment': [], 'id': [], 'date': [], 'query': [], 'user': [], 'tweet': []}
    with open(train_dataset_path) as train_dataset:
        train_dataset_csv = csv.reader(train_dataset, delimiter=",", strict=True)
        for row_i, row in enumerate(train_dataset_csv):
            row: list[str] = row
            i = 0
            for col_name in columns_data.keys():
                col_name: str = col_name
                if col_name == 'sentiment':
                    columns_data[col_name].append(numeric_sentiment_to_str(int(row[i])))
                else:
                    columns_data[col_name].append(row[i])
                i += 1
            if not row_i % (10 ** 3):
                print(f'{row_i}/{(10 ** 6) * 1.6} reader rows')

    train_dataset = pd.DataFrame(data=columns_data)
    print(train_dataset.head())

    train_dataset.to_csv(cleaned_train_dataset_path)
    print('Training dataset cleaned and saved.')

    sns.catplot(x='sentiment', data=train_dataset, kind='count')
    plt.show()

def numeric_sentiment_to_str(sentiment: float):
    if sentiment == 0:
        return 'Negative'
    if sentiment == 2:
        return 'Neutral'
    if sentiment == 4:
        return 'Positive'
    
    raise NotImplementedError("Numeric sentiment isn't added")


if __name__ == '__main__':
    main()