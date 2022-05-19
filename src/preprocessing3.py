import numpy as np
import pandas as pd
import time
from datetime import datetime as dt

start = dt.now()


r_dtypes = {'user_id': str, 'business_id': str, 'stars': int, 'text': str}
b_pandas = []


df = pd.read_json("src/data/yelp_ridotto2.json", dtype=r_dtypes)

countsUser = df["user_id"].value_counts()

user_id_over50 = countsUser[countsUser > 50].index.to_list()
df2 = df[df.user_id.isin(user_id_over50)]

df2.to_json('src/data/yelp_ridotto3.json')


end = dt.now()
elapsed = end-start
print("Tempo di esecuzione: %02d:%02d:%02d:%02d" % (elapsed.days,
      elapsed.seconds // 3600, elapsed.seconds // 60 % 60, elapsed.seconds % 60))
