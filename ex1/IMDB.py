import pandas as p
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as pl
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
#vocabulary size
s=set(words)
vsize = len(s)
print("Vocabulary size :", vsize)
#indentifying the stop words
stop_words = {
    'the','is','in','and','to','of','a','for','on','with',
    'this','that','it','as','an','are','was','were','be'
}
fwords = [w for w in words if w not in stop_words]
fseries = p.Series(fwords)
t = fseries.value_counts().head(10)
print("\nTop 10 AFTER removing stop words:\n", t)
#average line length
alen=len(words)/len(sent2)
print("\nAverage line length :", alen)
#wordcloud
clean_text = " ".join(fwords)

wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='white'
).generate(clean_text)

pl.figure(figsize=(10,5))
pl.imshow(wordcloud)
pl.axis("off")
pl.show()