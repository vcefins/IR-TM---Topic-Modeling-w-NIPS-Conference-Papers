import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import json
import ast
from nltk.corpus import wordnet
import string
import nltk

# nltk.download('averaged_perceptron_tagger')
# nltk.download('punkt')
# nltk.download('stopwords')


lemmatizer = WordNetLemmatizer()


# Converts NLTK's POS tagger output into appropriate input for WordNetLemmatizer
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


# parameters for determining the amount of topics and displayed important words


# reading csv and creating data corpus
csv = pd.read_csv("papers.csv", sep=",", header=0)
text = csv.iloc[:, 6].to_frame().T

# Converting Dataframe to List
text_in_List = []
for index in range(text.size):
    text_in_List.append(text[index][0])

print(len(text_in_List), 'papers have found. \nInitiating POS tagging.')

rewrite_pos = False

# Part of Speech tagging
text_pos = []
index_counter = 0
if rewrite_pos:
    with open("file2.txt", "a+") as f:
        for t in text_in_List:
            string = nltk.pos_tag(word_tokenize(t))

            f.write(str(string) + '\n')

            index_counter += 1
            if index_counter % 100 == 0:
                print("File", index_counter, "saved to file: abstracts.txt")

print("\n\n\nPOS tagging completed.\nSaved POS tagged corpus to file.")

############################
# After POS Tagging and storing the dataset, data from file.txt can be read directly into a list.
# Since calculating everything in each iteration is very inefficient.
############################
stop_words = set(stopwords.words('english'))
text_lemmad = []
counter = 0
with open("file2.txt", "r") as f:
    with open("file2_lemma.txt", "a+") as lemmafile:
        for paper in f:
            wordnet_tagged = map(lambda x: (x[0], get_wordnet_pos(x[1])), ast.literal_eval(paper))
            lemmatized_sentence = []
            for word, tag in wordnet_tagged:
                # Filtering STOP WORDS
                if word.lower() not in stop_words:
                    # Lemmatize
                    if tag is None:
                        lemmatized_sentence.append(word.lower())
                    else:
                        lemmatized_sentence.append(lemmatizer.lemmatize(word.lower(), tag))

            lemmafile.write(
                "\n" + str(lemmatized_sentence).translate((str.maketrans('', '', string.punctuation))) + ";")

            counter += 1
            if counter % 100 == 0:
                print(counter)

print("Downloading POS tagged dataset complete.\nInitiating Lemmatization.")

# Lemmatizing (while getting rid of stop words)


# for paper in text_posd:
#     # POS Tags translated.
#     wordnet_tagged = map(lambda x: (x[0], get_wordnet_pos(x[1])), paper)
#     lemmatized_sentence = []
#     for word, tag in wordnet_tagged:
#         # Filtering STOP WORDS
#         if word not in stop_words:
#             # Lemmatize
#             if tag is None:
#                 lemmatized_sentence.append(word)
#             else:
#                 lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
#
#     with open("lemma.txt", "a+") as f:
#         f.write("\n" + str(lemmatized_sentence))
#     print("Document", delete_this_counter, "has been lemmad & stored in file: lemma.txt")

# x = 0
# for paper in text2:
#     papers_tokenized.append([i.lower() for i in word_tokenize(paper) if i.lower() not in stop_words])
#     x += 1
#     print(x)

"""

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

# doc_term_mat_norm = normalize(doc_term_mat, norm='l1', axis=1)

print(topic_words)
"""
