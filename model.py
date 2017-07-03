'''
Authors: Anurag Prasad (anuragp1@bu.edU), Jarrod Lewis (jl101995@bu.edu)
File:    model.py
Purpose: Trains LDA model using Yelp dataset reviews of cafes
'''

from bs4 import BeautifulSoup
import sys
import time
import logging
import argparse
import requests
import codecs
import urllib
import os
import requests
import json
import nltk
from nltk.corpus import stopwords
import gensim
from gensim.corpora import BleiCorpus
from gensim import corpora
from gensim.models import LdaModel
import numpy as np

# YELP DATASET: train on business reviews for businesses within the "Coffee & Tea" category
coffee_places = set()

with open('yelp_academic_dataset_business.json') as businesses:
    for item in businesses:
        biz = json.loads(item)
        if biz['categories'] is not None:
            if "Coffee & Tea" in biz['categories']:
                coffee_places.add(biz['business_id'])
        
reviewData = {}

with open('yelp_academic_dataset_review.json') as reviews:
    for item in reviews:
        rev = json.loads(item)
        if rev['business_id'] in coffee_places:
            reviewData[rev['review_id']] = {'text':rev['text'], 'stars':rev['stars']}            

# Split review into sentences, remove stopwords, extract parts-of-speech tags
# (opt. if lots of reviews) store each review into MongoDB db called 'Reviews'
stopWords = set(stopwords.words('english'))

for revId in reviewData:
    reviewWords = []
    sentences = nltk.sent_tokenize(reviewData[revId]['text'].lower())
    
    for sentence in sentences:
        tokens = nltk.word_tokenize(sentence)
        text = [w for w in tokens if w not in stopWords]
        tagged_text = nltk.pos_tag(text)
        
        for word, tag in tagged_text:
            reviewWords.append({'word': word, 'pos': tag})
    
    reviewData[revId]['review_words'] = reviewWords

def lemmatize(reviewDict):
    '''Loop through the reviews, get nouns, and group them by lemma'''
    reviewCorpus = {}
    lemmatizer = nltk.WordNetLemmatizer()

    for review_count, review_content in reviewDict.items():
        nouns = []
        words = [w for w in review_content['review_words'] if w['pos'] in ['NN','NNS']]
        
        for w in words:
            nouns.append(lemmatizer.lemmatize(w['word']))
            
        reviewCorpus[review_count] = {'review_stars' : review_content['stars'], \
                                      'review_text' : review_content['text'], \
                                      'review_nouns' : nouns} 
    
    return reviewCorpus

def train(reviewDict, k):
	'''Feed reviews to LDA model using k topics'''

	id2word = corpora.Dictionary(reviewDict[review]["review_nouns"] for review in reviewDict)
	# id2word.filter_extremes(keep_n=10000)
    # id2word.compactify()

	corpora_dict = corpora.Dictionary(reviewDict[review]["review_nouns"] for review in reviewDict)
	corpora_dict.save('lda/dictionary.dict')

	corpus = [corpora_dict.doc2bow(reviewDict[review]["review_nouns"]) for review in reviewDict]
	corpora.BleiCorpus.serialize('lda/corpus.lda-c', corpus)
	corpus = corpora.BleiCorpus('lda/corpus.lda-c')

	if k == 50:
		# save lda model for 50 topics
		lda = gensim.models.LdaModel(corpus, num_topics=50, id2word=id2word)
		lda.save('lda/lda_50_topics.lda')

	elif k == 25:
		# save lda model for 25 topics
		lda = gensim.models.LdaModel(corpus, num_topics=25, id2word=id2word)
		lda.save('lda/lda_25_topics.lda')

	return lda


def main():
    global REVIEW_DICT 
    REVIEW_DICT = reviewData 

    # Check if the folder for the lda model exists. If it doesnt create the folder 
    if not os.path.exists('lda'):
        os.makedirs('lda')
    
    train(lemmatize(REVIEW_DICT), 25)

    # Get all 25 topics using K=25
    WEIGHT_TOPIC = []

    dictionary_path = "lda/dictionary.dict"
    corpus_path = "lda/corpus.lda-c"
    lda_model_path = "lda/lda_25_topics.lda"

    dictionary = corpora.Dictionary.load(dictionary_path)
    corpus = corpora.BleiCorpus(corpus_path)
    lda = LdaModel.load(lda_model_path)

    # Print topics
    TOPIC_DICT = dict(lda.show_topics(num_topics=25))

    for topicN, topicWeights in TOPIC_DICT.items():
	    print('Topic ' + str(topicN) + ' : \n' + str(topicWeights) + '\n')

if __name__ == '__main__':
	main()

