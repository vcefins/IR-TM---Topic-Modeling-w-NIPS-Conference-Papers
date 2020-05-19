from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

import nltk


lem = WordNetLemmatizer()


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

def lemmatize_sentence(sentence):
    wordnet_tagged = map(lambda x: (x[0], get_wordnet_pos(x[1])), sentence)
    print("Tagged.")
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            #if there is no available tag, append the token as is
            lemmatized_sentence.append(word)
        else:
            #else use the tag to lemmatize the token
            lemmatized_sentence.append(lem.lemmatize(word, tag))
    return " ".join(lemmatized_sentence)


list = [('balls', 'CD'), ('running','NN'), ('following', 'VBZ'), ('crying', 'CONJ'), ('fucking', 'V')]
list2 = [[('balls', 'CD'), ('running','NN'), ('following', 'CC'), ('crying', 'CONJ'), ('fucking', 'V')],[('suckmicock', 'NIG')]]


stop_words = set(stopwords.words('english'))


for word in list:
    if word[0] not in stop_words:
        print(word, word[0])

a = lemmatize_sentence(list)
print(type(a))


# dick = [lem.lemmatize(word[0], word[1]) for word in list]


"""
for word in wordnet_tagged:
    print("Word:", word[0], "  POS:", word[1])
    print("POS tag translated to:", get_wordnet_pos(word[1]))
    print("Lemmatized:", lem.lemmatize(word[0], get_wordnet_pos(word[1])))
"""