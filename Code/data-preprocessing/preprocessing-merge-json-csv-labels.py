from __future__ import print_function, division
import pandas as pd

import json

data = []
with open('yelp_academic_dataset_photo.json') as f:
    for line in f:
        data.append(json.loads(line))

yelp_frame = pd.read_csv('yelp_academic_dataset_photo_features.csv')


yelp_frame.insert(loc=0, column='type', value=0)

yelp_frame.iloc[0,0] = "TEST"


yelp_size = len(yelp_frame)

values = set()

index = 0

for i in data:
    values.add(i['label'])
    photo_id = i['photo_id']
    label = i['label']
    #print (label)
    #print (i['photo_id'])
    for j in range(yelp_size):
        yelp_frame_ind = yelp_frame.iloc[j,1]
        if photo_id == yelp_frame_ind:
            yelp_frame.iloc[j,0] = label
            #break
            #index += 1
            break

yelp_frame.to_csv("yelp_dataset_preprocessed.csv")
