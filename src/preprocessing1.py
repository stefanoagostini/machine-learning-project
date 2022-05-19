import numpy as np
import pandas as pd
from datetime import datetime as dt

start = dt.now()

r_dtypes = {'review_id': str, 'user_id': str,
            'business_id': str, 'stars': int,
            'date': str, 'text': str, 'useful': int,
            'funny': int, 'cool': int}
b_pandas = []


reader = pd.read_json("src/data/yelp_academic_dataset_review.json", orient="records", lines=True,
                      dtype=r_dtypes, chunksize=100000)
l = 0
for chunk in reader:
    l += len(chunk)
    reduced_chunk = chunk.drop(
        columns=['review_id', 'useful', 'funny', 'cool', 'date'])
    b_pandas.append(reduced_chunk)

b_pandas = pd.concat(b_pandas, ignore_index=True)

b_pandas.to_json('src/data/yelp_ridotto.json')

end = dt.now()
elapsed = end-start
print("Tempo di esecuzione: %02d:%02d:%02d:%02d" % (elapsed.days,
      elapsed.seconds // 3600, elapsed.seconds // 60 % 60, elapsed.seconds % 60))
