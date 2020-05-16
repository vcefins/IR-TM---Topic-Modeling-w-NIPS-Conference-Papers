import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.stem import WordNetLemmatizer

from nltk.tokenize import word_tokenize

num_topics = 20
n_top_words = 20

csv = pd.read_csv("papers.csv", sep=",", header=0)

text = csv.iloc[:, 6].to_frame().T

lemmatizer = WordNetLemmatizer()

text.iloc[0].apply(lemmatizer.lemmatize())


vectorizer = CountVectorizer(analyzer='word', max_features=10000)
doc_term_count = vectorizer.fit_transform(text.iloc[0])

transformer = TfidfTransformer(smooth_idf=False)
doc_term_tfidf = transformer.fit_transform(doc_term_count)

doc_term_tfidf_norm = normalize(doc_term_tfidf, norm='l1', axis=1)

nmf_model = NMF(n_components=num_topics, init='nndsvd')
nmf_model.fit(doc_term_tfidf_norm)

# function obtained from https://medium.com/ml2vec/topic-modeling-is-an-unsupervised-learning-approach-to-clustering-documents-to-discover-topics-fdfbf30e27df
def get_nmf_topics(model, n_top_words):
    # the word ids obtained need to be reverse-mapped to the words so we can print the topic names.
    feat_names = vectorizer.get_feature_names()

    word_dict = {}
    for i in range(num_topics):
        # for each topic, obtain the largest values, and add the words they map to into the dictionary.
        words_ids = model.components_[i].argsort()[:-n_top_words - 1:-1]
        words = [feat_names[key] for key in words_ids]
        word_dict['Topic # ' + '{:02d}'.format(i + 1)] = words

    return pd.DataFrame(word_dict)


topic_words = get_nmf_topics(nmf_model, n_top_words)

#doc_term_mat_norm = normalize(doc_term_mat, norm='l1', axis=1)

print(topic_words)