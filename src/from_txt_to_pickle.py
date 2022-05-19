import pickle
import numpy as np
from datetime import datetime as dt

start = dt.now()

txtPath = 'src/data/glove.6B.300d.txt'
picklePath = 'src/data/glove.6B.300d.pickle'

f = open(txtPath, 'r', encoding="utf8")
embeddings_index = {}
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

g = open(picklePath, 'wb')
pickle.dump({'embeddings_index': embeddings_index}, g)
g.close()

end = dt.now()
elapsed = end-start
print("Tempo di esecuzione: %02d:%02d:%02d:%02d" % (elapsed.days,
      elapsed.seconds // 3600, elapsed.seconds // 60 % 60, elapsed.seconds % 60))
