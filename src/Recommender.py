import pandas as pd
import numpy as np
from pandas import DataFrame
from pandas import Series
import surprise

import os

df = pd.read_csv('../resources/altered-subset.csv', sep=',', header=0, encoding="latin")


def string2numeric_hash(text):
    import hashlib
    return int(hashlib.md5(text.encode('utf-8')).hexdigest()[:8], 16)

def constraint_value(value):
    if value > 0:
        return 1
    else:
        return 0


users = Series()
items = Series()
rating = Series()

for column in df.filter(regex="^201612").columns:
    users = users.append(df['Unnamed: 0'], ignore_index=True)
    items = items.append(Series(string2numeric_hash(column) for i in range(df.shape[0])), ignore_index=True)
    rating = rating.append(df[column].apply(constraint_value), ignore_index=True)

rtings = DataFrame({'user': users, 'item': items, 'rating': rating})

print(rtings.head())



reader=surprise.dataset.Reader(line_format='user item rating',rating_scale=(0,1))

mr_train=surprise.dataset.Dataset.load_from_df(rtings,reader=reader)
mr_trainset=mr_train.build_full_trainset()

import surprise.prediction_algorithms.knns as knns
knnbasic=knns.KNNBasic(k=40,min_k=1,sims_options={'name':'cosine','user_based':True})
knnbasic.fit(mr_trainset)

print(rtings.head(15))



print(knnbasic.predict(uid=13,iid=765270874))









# for row in dataset.columns:
#     userid = row[dataset.columns.get_loc('user')]
# dataset.insert(0, 'user', df['Unnamed: 0'])
