import pickle
import time
import math
from collections import Counter, defaultdict

print("Loading text_search_index.pkl...")
t0 = time.time()
with open("data/input/input/index/text_search_index.pkl", "rb") as f:
    data = pickle.load(f)

documents = data["documents"]
tokenized = data["tokenized"]
N = len(documents)
print(f"Loaded {N} documents in {time.time()-t0:.2f}s")

print("Building BM25 data structures...")
t1 = time.time()
doc_lengths = [len(t) for t in tokenized]
avgdl = sum(doc_lengths) / max(N, 1)

df = Counter()
for tokens in tokenized:
    df.update(set(tokens))

idf = {}
for term, count in df.items():
    idf[term] = math.log((N - count + 0.5) / (count + 0.5) + 1.0)

inverted = defaultdict(list)
for doc_idx, tokens in enumerate(tokenized):
    tf = Counter(tokens)
    for term, count in tf.items():
        inverted[term].append((doc_idx, count))

print(f"Built BM25 in {time.time()-t1:.2f}s")

print("Saving bm25_index.pkl...")
t2 = time.time()
with open("data/input/input/index/bm25_index.pkl", "wb") as f:
    pickle.dump({
        "doc_lengths": doc_lengths,
        "avgdl": avgdl,
        "idf": idf,
        "inverted": dict(inverted)
    }, f)

print(f"Saved in {time.time()-t2:.2f}s")
