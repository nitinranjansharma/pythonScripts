
# coding: utf-8

# In[2]:

from collections import Counter


# In[3]:

import numpy as np
import pandas as pd


# In[4]:

headlines = [
    "PretzelBros, airbnb for people who like pretzels, raises $2 million",
    "Top 10 reasons why Go is better than whatever language you use.",
    "Why working at apple stole my soul (I still love it though)",
    "80 things I think you should do immediately if you use python.",
    "Show HN: carjack.me -- Uber meets GTA"
]


# In[5]:

words = list(set(" ".join(headlines).split(" ")))


# In[50]:

matrix = []
def makematrix(headlines,words):
    matrix = []
    for line in headlines:
        count = Counter(line)
        row = [count.get(w,0) for w in words]
        matrix.append(row)
    df = pd.DataFrame(matrix)
    df.columns = words
    return df
    


# In[ ]:




# In[21]:

print(makematrix(headlines, words))


# In[22]:

import re


# In[53]:

new_headlines1 = [re.sub(r'[^\w\s\d]','',h.lower()) for h in headlines]


# In[54]:

new_headlines = [re.sub("\s+", " ", h) for h in new_headlines]


# In[55]:

words = list(set(" ".join(new_headlines).split(" ")))


# In[56]:


print(makematrix(new_headlines, unique_words))


# In[73]:

#removing stopwords
stopwords = ["the",
            "you",
            "2",
            "is",
            "if",
            "for",
            "why",
            "hn",
            "at",
            "it",
            "go",
            "still",
            "do"]


# In[74]:

unique_words = list(set(" ".join(stopwords).split(" ")))


# In[75]:

unique_words = [re.sub(r'[^\s\w\d]','',h.lower())for h in unique_words]
unique_words = [re.sub("\s+"," ",h) for h in unique_words]


# In[76]:

words = [w for w in words if w not in unique_words]


# In[77]:

print(makematrix(new_headlines,words))


# In[78]:

from sklearn.feature_extraction.text import CountVectorizer


# In[79]:

vectorizer = CountVectorizer(lowercase = True, stop_words = "english")


# In[80]:

matrix = vectorizer.fit_transform(headlines)


# In[82]:

print(matrix.todense())


# In[88]:

subject = pd.read_csv("/Users/nitinranjansharma/Documents/Nitin/Files/stories.csv", header = None)


# In[103]:

subject.head()


# In[105]:

subject.isnull().sum()


# In[108]:

subject = subject.dropna(axis = 0, how = 'any')


# In[109]:

subject.isnull().sum()


# In[97]:

subject.columns = ["test","date","test1","test2","upvotes","url","test3","headline"]


# In[100]:

subject.drop(subject.columns[[0,2,3,6]], inplace =True, axis = 1)


# In[102]:

subject['full_test'] = subject['headline'] + " " + subject['url']


# In[110]:

commatrix = vectorizer.fit_transform(subject['headline'])


# In[113]:

print(commatrix.shape)


# In[114]:

col = subject["upvotes"].copy(deep=True)
col_mean = col.mean()
col[col < col_mean] = 0
col[(col > 0) & (col > col_mean)] = 1


# In[120]:

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# In[122]:

selector = SelectKBest(chi2, k=1000)
selector.fit(commatrix, col)
top_words = selector.get_support().nonzero()


# In[123]:

chi_matrix = commatrix[:,top_words[0]]


# In[126]:

transform_functions = [
    lambda x: len(x),
    lambda x: x.count(" "),
    lambda x: x.count("."),
    lambda x: x.count("!"),
    lambda x: x.count("?"),
    lambda x: len(x) / (x.count(" ") + 1),
    lambda x: x.count(" ") / (x.count(".") + 1),
    lambda x: len(re.findall("\d", x)),
    lambda x: len(re.findall("[A-Z]", x)),
]


# In[127]:

columns = []
for func in transform_functions:
    columns.append(subject["headline"].apply(func))


# In[129]:

meta = np.asarray(columns).T


# In[131]:




# In[132]:

columns = []
submission_date = pd.to_datetime(subject['date'])



# In[134]:

transfer_function = [ lambda x: x.year , lambda x: x.month, lambda x: x.day, lambda x:x.hour, lambda x: x.minute]


# In[136]:

for func in transfer_function:
    columns.append(submission_date.apply(func))


# In[137]:

non_nlp_features = np.asarray(columns).T


# In[138]:

features = np.hstack([non_nlp_features,meta,chi_matrix.todense()])


# In[141]:

from sklearn.linear_model import Ridge


# In[143]:

import random


# In[151]:

train_rows = 1000500
random.seed(1)


# In[152]:

indices = list(range(features.shape[0]))


# In[153]:

random.shuffle(indices)


# In[154]:

train = features[indices[:train_rows],:]


# In[155]:

test = features[indices[train_rows:],:]


# In[158]:

train_upvotes = subject['upvotes'].iloc[indices[:train_rows]]


# In[159]:

test_upvotes = subject['upvotes'].iloc[indices[train_rows:]]


# In[160]:

train = np.nan_to_num(train)


# In[161]:

reg = Ridge(alpha = 0.1)


# In[162]:

reg.fit(train,train_upvotes)


# In[163]:

predictions = reg.predict(test)


# In[164]:

#accuracy
print(sum(abs(predictions - test_upvotes))/len(predictions))


# In[165]:

average_upvotes = sum(test_upvotes)/len(test_upvotes)
print(sum(abs(average_upvotes - test_upvotes)) / len(predictions))


# In[166]:

from sklearn.ensemble import RandomForestClassifier


# In[167]:

clf = RandomForestClassifier(max_depth=25, random_state=0)


# In[168]:

clf.fit(train,train_upvotes)


# In[169]:

predictions_rf = clf.predict(test)


# In[171]:

#accuracy
print(sum(abs(predictions_rf - test_upvotes))/len(predictions_rf))


# In[ ]:




# In[ ]:




# In[ ]:



