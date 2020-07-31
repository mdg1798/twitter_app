"""[summary]
this module provide preprocessing functions for analyzing tweet
"""
import pandas as pd
import numpy as np
import re
import nltk
import pickle
from nltk.corpus import stopwords
from numpy import array
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

# loading stop words from local
# stop words is pre download from nltk
STOPWORDS = set(stopwords.words('english'))
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
    
maxlen=1000

def preprocess_text(sen):
    # Removing html tags
    sentence = remove_tags(sen)

    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)
    
    sentence = ' '.join(word for word in sentence.split() if word not in STOPWORDS)
    

    return sentence

TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    return TAG_RE.sub('', text)

def apply(df,col):
    X=[]
    sentences = list(df[col])
    for sen in sentences:
            X.append(preprocess_text(sen))
    return X     

def token(X,maxlen=maxlen,tokenizer=tokenizer):
    
    X_1= tokenizer.texts_to_sequences(X)
    X_1=pad_sequences(X_1, padding='post', maxlen=maxlen)
    
    return X_1

def pipeline(df,col):
    X=apply(df,col)
    X_final=token(X)
    return X_final

# class Preprocessor(object):

#     def __init__(
#             self, path, stop_words=stop_words, tokenizer=tknzr, 
#             max_length_dictionary=10000, max_length_tweet=100
#         ):
#         self.tokenizer = tokenizer
#         self.stop_words = stop_words
#         self.embeddingMap = self.load_embedding(path, max_length_dictionary)
#         self.max_length_tweet = max_length_tweet
#         self.pad_seq = None


#     def pipeline(self, t):
#         strings = self.clean_text(t)
#         stringsList = self.tokenize_text(
#             strings, tokenizer=self.tokenizer, stop_words=self.stop_words
#         )
#         tokenList = self.replace_token_with_index(stringsList, self.embeddingMap)
#         self.pad_seq = self.pad_sequence(tokenList, self.max_length_tweet)
#         self.pad_seq = np.array(self.pad_seq)
#         self.pad_seq = self.pad_seq.reshape(1,self.max_length_tweet)
#         return self.pad_seq


#     def clean_text(self, t):
#         """
#         remove redundant text from tweets
#         only return alphbets, numbers, !, and ?

#         Arguments:
#             t {[str]} -- any tweets

#         Returns:
#             t [str] -- clean tweets
#         """
#         p.set_options(
#             p.OPT.URL,
#             p.OPT.MENTION,
#             p.OPT.HASHTAG,
#             p.OPT.RESERVED,
#             p.OPT.EMOJI,
#             p.OPT.SMILEY,
#         )
#         t = p.clean(t)
#         t = re.sub(r"[\\//_,;.:*+\-\>\<)(%^$|~&`'\"\[\]\=]+", '', t)
#         t = re.sub(r'[^\x00-\x7F]+', ' ', t)
#         return t.lower()


#     def tokenize_text(self, t, tokenizer=tknzr, stop_words=stop_words):
#         """
#         tokenize preprocessed tweets

#         Arguments:
#             t {[str]} -- clean tweets

#         Keyword Arguments:
#             tokenizer {[nltk tokenizer]} -- one of nltk tokenizer (default: {TweetTokenizer})
#             stop_words {[dict]} -- stop words map (default: {stop words})

#         Returns:
#             [list of str] -- ex. ['I', 'am', 'Richard', '!']
#         """
#         tList = [i for i in tokenizer.tokenize(t) if i not in stop_words]
#         return tList


#     def load_embedding(self, path, max_length_dictionary=10000):
#         """
#         load embedding map

#         Arguments:
#             path {[str]} -- the absolute path of where embedding map is

#         Keyword Arguments:
#             max_length_dictionary {int} -- maximum length of words to be loaded (default: {10000})

#         Returns:
#             [dict] -- embedding map loaded as python dictionary
#         """
#         embeddings_dict = {}
#         i = 0

#         if ".zip/" in path:
#             archive_path = os.path.abspath(path)
#             split = archive_path.split(".zip/")
#             archive_path = split[0] + ".zip" 
#             path_inside = split[1]
#             archive = zipfile.ZipFile(archive_path, "r")
#             embeddings = archive.read(path_inside).decode("utf8").split("\n")
#             embeddings = ['<pad>', '<unknown>'] + embeddings

#             for i, words in enumerate(embeddings):
#                 if words == '<unknown>' or words == '<unk>':
#                     if i >= 2:
#                         continue

#                 embeddings_dict[words] = i

#                 if i == max_length_dictionary:
#                     break

#             return embeddings_dict
            

#         with open(path, 'r',encoding="utf-8") as f:
#             for line in f:
#                 values = line.split()
#                 if values[0].isalnum():
#                     embeddings_dict[values[0]] = i
#                     i += 1

#                 if i == max_length_dictionary:
#                     break

#         return embeddings_dict


#     def replace_token_with_index(self, tList, embeddingMap):
#         """
#         replace token with index in embedding map

#         Arguments:
#             tList {[list of str]} -- the output from tokenize_text
#             embeddingMap {[dict]} -- the output from load_embedding or a self-defined dictionary

#         Returns:
#             [list of int] -- replace the tokens with index, ex. [1, 892, 3, 2467]
#         """
#         tNewList = []
#         for t in tList:
#             # if t is not in EmbeddingMap continue the loop
#             indice = embeddingMap.get(t)
#             if indice:
#                 tNewList.append(indice)
#         return tNewList


#     def pad_sequence(self, tList, max_length_tweet=10):
#         """
#         construct pad sequence

#         Arguments:
#             tList {[list of int]} -- output from replace_token_with_index

#         Keyword Arguments:
#             max_length_tweet {int} -- the maximum length of tweet (default: {10})

#         Returns:
#             [list of int] -- return pad sequence list
#         """
#         reLength = max_length_tweet - len(tList)

#         if reLength > 0:
#             tList.extend([0] * reLength)
#         elif reLength < 0:
#             tList = tList[:max_length_tweet]

#         return tList
