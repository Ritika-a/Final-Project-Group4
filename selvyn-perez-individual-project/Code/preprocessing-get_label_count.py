import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import time


df = pd.read_csv('yelp_dataset_preprocessed.csv')
df_size = 100000

x = np.zeros(shape=(df_size, 32, 32, 3))
food = []
drink = []
outside = []
inside = []
menu = []

start_time = time.time()
for i in range(df_size):
    try:
        df_label = df.iloc[i, 0]

        if df_label == "food":
            food.append(df_label)
        if df_label == "drink":
            drink.append(df_label)
        if df_label == "outside":
            outside.append(df_label)
        if df_label == "inside":
            inside.append(df_label)
        if df_label == "menu":
            menu.append(df_label)
    except:
        print ('error')
        continue

objects = ('food', 'drink', 'outside', 'inside', 'menu')
y_pos = np.arange(len(objects))
performance = [len(food), len(drink), len(outside), len(inside), len(menu)]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Usage')
plt.title('Programming language usage')
plt.show()
