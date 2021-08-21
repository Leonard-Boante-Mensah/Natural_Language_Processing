import os
import gensim
import spacy
from president_helper import read_file, process_speeches, merge_speeches, get_president_sentences, get_presidents_sentences, most_frequent_words
import numpy as np

# get list of all speech files
files = sorted([file for file in os.listdir() if file[-4:] == '.txt'])
# print(files)

# read each speech file
speeches = [read_file(f) for f in files]
# print(speeches[0])

# preprocess each speech
processed_speeches = process_speeches(speeches)


#Printing the first sentence of the first speech
# print(processed_speeches[0][0])

# merge speeches
all_sentences = merge_speeches(processed_speeches)
# print(all_sentences[0])

# view most frequently used words
most_freq_words = most_frequent_words(all_sentences)
# print(most_freq_words)

# create gensim model of all speeches
all_prez_embeddings = gensim.models.Word2Vec(all_sentences, size=96, window=5, min_count=1, workers=2, sg=1)

# view words similar to freedom
similar_to_freedom = all_prez_embeddings.most_similar('freedom', topn=20)
#print(similar_to_freedom)

my_similarword2 = all_prez_embeddings.most_similar('citizen', topn=20)

my_similarword3 = all_prez_embeddings.most_similar('country', topn=20)

# print(my_similarword2)
# print(my_similarword3)
# get President Roosevelt sentences
roosevelt_sentences = get_president_sentences('franklin-d-roosevelt')

# view most frequently used words of Roosevelt
roosevelt_most_freq_words = most_frequent_words(roosevelt_sentences)
# print(roosevelt_most_freq_words)

# create gensim model for Roosevelt
roosevelt_embeddings = gensim.models.Word2Vec(roosevelt_sentences, size=96, window=5, min_count=1, workers=2, sg=1)

# view words similar to freedom for Roosevelt
roosevelt_similar_to_freedom = all_prez_embeddings.most_similar('freedom', topn=20)
# print(roosevelt_similar_to_freedom)
# for word in similar_to_freedom:
#   if word in roosevelt_similar_to_freedom:
#     print(word)

# get sentences of multiple presidents
rushmore_prez_sentences = get_presidents_sentences(["washington","jefferson","lincoln","theodore-roosevelt"])


# view most frequently used words of presidents
rushmore_most_freq_words = most_frequent_words(rushmore_prez_sentences)
# print(rushmore_most_freq_words)

# create gensim model for the presidents
rushmore_embeddings = gensim.models.Word2Vec(rushmore_prez_sentences, size=96, window=5, min_count=1, workers=2, sg=1)


# view words similar to freedom for presidents
rushmore_similar_to_freedom = rushmore_embeddings.most_similar('freedom', topn=20)

print(rushmore_similar_to_freedom)


