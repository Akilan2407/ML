import pandas as p
from collections import Counter
data=p.read_csv('IMDB Dataset.csv')
print(data.head(10))
#number of sentences
text = " ".join(data['review'].dropna())
sent1=text.split('.')
sent2 = [s.strip() for s in sent1 if s.strip() != ""]
print("Number of sentences:", len(sent2))
#number of words
words=text.split()
print("Number of words:", len(words))
#10 most common words
w=text.lower().split()
wc=Counter(w)
top=wc.most_common(10)
print("top 10 words : ")
for w,count in top:
    print(f" '{w}'  with frequency {count}")
