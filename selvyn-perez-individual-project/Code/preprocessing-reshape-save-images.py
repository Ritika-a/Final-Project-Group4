import pandas as pd
import numpy as np
import cv2
import urllib


def normalize(x):
    min_val = np.min(x)
    max _val = np.max(x)
    x = (x-min_val) / (max_val-min_val)
    return x

dir = 'https://storage.googleapis.com/ml2-group4-project/yelp_academic_dataset_photos/'
new_dir = '< directory where to store reshaped images >'

df = pd.read_csv('yelp_dataset_preprocessed.csv')
df_size = 100000

x = np.zeros(shape=(df_size, 32, 32, 3))

labels = []

for i in range(df_size):
    try:
        df_label = df.iloc[i, 0]
        labels.append(df_label)
        df_img = df.iloc[i, 1]
        df_img_path = dir + df_img + ".jpg"

        url_response = urllib.urlopen(df_img_path)
        df_img_path = np.array(bytearray(url_response.read()), dtype=np.uint8)
        df_orig_img = cv2.imdecode(df_img_path, -1)
        height, width, depth = df_orig_img.shape

        df_new_img = cv2.resize(df_orig_img, (32, 32))

        x[i] = df_new_img

        # Save reshaped images if necessary
        # cv2.imwrite(new_dir + df_img + ".jpg", df_new_img)
    except:
        print ('error - problem with image, will skip')
        continue

label = pd.get_dummies(labels)
y = label.values
x = normalize(x)

np.save("all_labels.npy", y)
np.save("all_images.npy", x)
