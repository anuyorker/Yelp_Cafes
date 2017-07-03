'''
Authors: Anurag Prasad (anuragp1@bu.edU), Jarrod Lewis (jl101995@bu.edu)
File:    implementation.py
Purpose: Implements LDA model and generates recommendations for cafes
		 based on topic weight, sentiment score, and star rating. Also
		 incorporates regression to analyze the relationship between
		 star rating and sentiment score, and to predict star rating
		 from sentiment score for an unseen review.
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
from scipy import stats
import corenlp
import subprocess
import numpy as np

# 25 topic labels interpreted from topic keywords/weights found in 'model.py'
TOPIC_LABELS = {
0 : 'Atmosphere (place, staff, love, selection, music, fun)',
1 : 'Latte/Chai/Milk Drinks (latte, chai, vanilla, mocha, milk)',
2 : 'Wait Time / Service (time, place, food, order, service, wait)',
3 : 'Staff (guy, smile, lady, girl, thank you, counter)',
4 : 'Doughnuts (donut, doughnut, dozen, apple, dunkin, shop, creme)',
5 : 'Visited on Vacation (vacation, kale, parfait, frappe, paradise)',
6 : 'Food / Meals / Restaurant (food, place, restaurant, service, meal)',
7 : 'Physical Space (place, table, area, spot, staff, parking lot, space)',
8 : 'Baked Bread Items (bagels, pastry, bakery, bread, tart, cheese)',
9 : 'Sandwiches, Lunch, and Breakfast Food (sandwich, breakfast, food, lunch, egg)',
10 : 'Small Dessserts (cream, ice, waffle, strawberry, dessert, macaroon, banana)',
11 : 'Bakery Style Cakes (cake, dessert, cupcake, brownie, cheesecake, birthday, bakery)',
12 : 'Alcoholic Beverages (beer, wine, bottle, club)',
13 : 'Sweet Flavors (gelato, butter, taste, caramel, flavor, peanut, pistachio)',
14 : 'Unknown (die, der, da, le, ist, man)',
15 : 'Chocolates (bar, raspberry, fountain, dark, truffle)',
16 : 'Coffee – General (coffee, shop, place, cup, bean, espresso)',
17 : 'Juice (juice, roll, orange, cinnamon)',
18 : 'Tea (tea, selection, store, cup)',
19 : 'Cookies and Pastries (cookie, muffin, chip, scone, macarons, blueberry, salt, cheddar)',
20 : 'Customer Service (starbucks, location, time, service,customer, order, line, employee, staff)',
21 : 'Specialty Tea/Milk Drinks (tea, milk, drink, boba, place)',
22 : 'Young People (kid, place, class, year, hip, student, college)',
23 : 'Room (cafe, wall, room, door, floor)',
24 : 'Misc.'
}

def predTopics(review_text):
    separated_text = review_text.lower().split()
    
    # apply LDA model
    dictionary_path = "lda/dictionary.dict"
    corpus_path = "lda/corpus.lda-c"

    lda_model_path = "lda/lda_25_topics.lda"

    dictionary = corpora.Dictionary.load(dictionary_path)
    corpus = corpora.BleiCorpus(corpus_path)
    lda = LdaModel.load(lda_model_path)
    
    review_bow = dictionary.doc2bow(separated_text)
    
    return lda[review_bow]

def sentimentScore(text):
    sentences = list(filter(None, text.replace('\n','').split('.')))
    sentimentVals = []
    for sentence in sentences:
        try:
            sentimentVals.append(corenlp.sentiment_analysis_on_sentence(sentence))
        except:
            continue
        
    return np.mean(sentimentVals)

def get_reviews(theurl):
    '''Get 20 reviews from first page of restaurant'''
    reviewInfo = {}
    
    stop = set(stopwords.words('english'))
    
    main_page = requests.get(theurl)
    if main_page.status_code == 200:
        soup = BeautifulSoup(main_page.content, "html.parser")
    else:
        print('Non-200 status code. Get request for url failed.')
    
    review_div = soup.findAll('div',{'itemprop':'review'})

    reviewCount = 1
    
    for i in review_div: # iterating through review_div 
        # get review star rating
        reviewStar = float(i.find('meta',{'itemprop':'ratingValue'}).get('content', None))

        # get review body text
        reviewBody = i.find('p',{'itemprop':'description'})
        for txt in reviewBody:
            if type(txt) != '<p>' and not str(txt).startswith('<p>'):
                reviewText = txt
        
        topics = predTopics(reviewText)
        sentiment = sentimentScore(reviewText)
        
        reviewInfo[reviewCount] = {'review_stars' : reviewStar, \
                                   'review_text': reviewText, \
                                   'review_topics' : topics, \
                                   'sentiment_score' : sentiment} 
        reviewCount += 1
        
    return reviewInfo

def getReviewData(resUrl):
    '''Return dictionary of reviews (with their extracted topics and sentiment scores)
       and the least frequent topics across all reviews for this restaurant
    '''
    resReviews = get_reviews(resUrl)
    
    resData = {}
    
    for count, items in resReviews.items():
        resData[count] = {'review_stars' : items['review_stars'],
                          'review_text' : items['review_text'],
                          'review_topics' : sorted([(t,TOPIC_LABELS[t],w) for (t,w) in items['review_topics']], key=lambda x: x[2], reverse=True),
                          'sentiment_score' : items['sentiment_score']}
        
    return resData

def getLeastFrequent(reviewData):
	'''Return least frequent reviews. These will be ignored later on to acconut for low frequency topics.'''
	topicWeightList = []
	for count, item in reviewData.items():
		this = [(i[0], i[2]) for i in item['review_topics']]
		topicWeightList.append(this)

	total = 0
	topicDistribution = [0 for i in range(0,25)]
	for review in topicWeightList:
		for item in review:
			index = item[0]
			topicDistribution[index] += item[1]
			total += item[1]
	topicDistribution
	topicDistDict = {}
	for i in range(len(topicDistribution)):
		topicDistDict[TOPIC_LABELS[i]] =  topicDistribution[i]/total

	leastFrequent = []
	for topic in topicDistDict.keys():
		if topicDistDict[topic] < .02:
			leastFrequent.append(topic)

	return leastFrequent

def topicsByWeight(data):
    '''Return list of ranked topics from highest to lowest aggregate weight on restaurant's reviews.'''
    topic_weights = {}
    
    for count, items in data.items():
        for topic in items['review_topics']:
            if topic[1] not in topic_weights:
                topic_weights[topic[1]] = topic[2]            
            else:
                topic_weights[topic[1]] += topic[2]
    
    ranked_topics = sorted(list(topic_weights.items()), key=lambda x: x[1], reverse=True)
    
    return ranked_topics

def getStats(data):
    '''Return ranked topics (by weight) and dictionary of stars/sentiment score for each topic'''
    topic_weights = {}
    stars_sentiments = {topic_name: {'sentiments':[], 'stars':[]} for n, topic_name in TOPIC_LABELS.items()}
    
    for count, items in data.items():
        for topic in items['review_topics']:
            stars_sentiments[topic[1]]['sentiments'].append(items['sentiment_score'])
            stars_sentiments[topic[1]]['stars'].append(items['review_stars'])
    
    # calculate average sentiment scores and star ratings for each topic
    for k in stars_sentiments:
        stars_sentiments[k]['sentiments'] = np.mean(stars_sentiments[k]['sentiments'])
        stars_sentiments[k]['stars'] = np.mean(stars_sentiments[k]['stars'])
        
    stars_sentiments = {topicItems[0]:topicItems[1] for topicItems in list(stars_sentiments.items()) \
                        if not np.isnan(topicItems[1]['sentiments']) and not topicItems[0] in getLeastFrequent(data)}
    
    return stars_sentiments
    
def rankBySentiment(ss_stats):
	'''Rank topics by topic group's average sentiment score'''
	topicSentimentsRanked = sorted([(i[0],i[1]['sentiments']) for i in list(ss_stats.items())], key=lambda x:x[1], reverse=True)
	return topicSentimentsRanked

def rankByStars(ss_stats):
	'''Rank topics by topic group's average star-rating'''
	topicStarsRanked = sorted([(i[0],i[1]['stars']) for i in list(ss_stats.items())], key=lambda x:x[1], reverse=True)
	return topicStarsRanked

def regSentStars(ss_stats):
    '''Do linear regression of sentiment score on star rating.
       Return slope, intercept, r_squared, p_value, std_err of regression
    '''
    topicSentiments = [i['sentiments'] for i in list(ss_stats.values())]
    topicStars = [i['stars'] for i in list(ss_stats.values())]
    slope, intercept, r_value, p_value, std_err = stats.linregress(topicSentiments, topicStars)
    
    return slope, intercept, r_value**2, p_value, std_err

def sentimentGroup(n):
    '''Takes sentiment score and classifies into sentiment group'''
    if round(n, 0) == 0:
        return 'very negative ({})'.format(str(round(n,2)))
    if round(n, 0) == 1:
        return 'negative ({})'.format(str(round(n,2)))
    elif round(n, 0) == 2:
        if n < 1.75:
            return 'slightly negative ({})'.format(str(round(n,2)))
        elif n > 2.25:
            return 'slightly positive ({})'.format(str(round(n,2)))
        else:
            return 'neutral ({})'.format(str(round(n,2)))
    elif round(n, 0) == 3:
        return 'positive ({})'.format(str(round(n,2)))
    elif round(n, 0) == 4:
        return 'very positive ({})'.format(str(round(n,2)))
 
def recommend(data, sentimentRank, starRank):
    '''Prints a recommendation based on the cafe's topic sentiments and stars '''
    
    # print('People feel good about these aspects of this cafe...')
    # posFeelings = [print('* ' + topic[0].split('(')[0] + ' ({})'.format(str(topic[1]))) for topic in sentimentRank[0:3]]
    # print()
    # print("However, they don't feel so good about these aspects of this cafe...")
    # negFeelings = [print('* ' + topic[0].split('(')[0] + ' ({})'.format(str(topic[1]))) for topic in sentimentRank[-3:]]
    # print()
    # print("Ratings for reviews about these topics are high...")
    # posStars = [print('* ' + topic[0].split('(')[0] + ' ({})'.format(str(topic[1]))) for topic in starRank[0:3]]
    # print()
    # print("Reviews for reviews about these topics are low...")
    # negStars = [print('* ' + topic[0].split('(')[0] + ' ({})'.format(str(topic[1]))) for topic in starRank[-3:]]
    # print()
    
    composite = sorted([(name, (stats['sentiments'], stats['stars'])) for name, stats in getStats(data).items()], key=lambda x:x[1], reverse=True)
    
    print('RECOMMENDATION: \n')
    print('Keep up the good work with...')
    posRecc = [print('---> ' + topic[0].split('(')[0] + '(avg. sentiment: {} / avg. stars: {})'.format(sentimentGroup(topic[1][0]), round(topic[1][1], 2))) for topic in composite[0:3]]
    print()
    print('May need to make improvements where performing lowest...')
    negRecc = [print('---> ' + topic[0].split('(')[0] + '(avg. sentiment: {} / avg. stars: {})'.format(sentimentGroup(topic[1][0]), round(topic[1][1], 2))) for topic in composite[-3:]]

def predictStars(data, unseenReview):
    '''Predict star rating for a based on a review's sentiment'''
    sentiment_score = sentimentScore(unseenReview)
    predStars = regSentStars(getStats(data))[1] + regSentStars(getStats(data))[0]*sentiment_score
    return predStars


def main():
	# NOTE: Before running corenlp, must start up NLP server using:
	# java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
	# subprocess.call(['cd','stanford-corenlp-full-2016-10-31'])
    subprocess.call(['java','-mx4g','-cp','*','edu.stanford.nlp.pipeline.StanfordCoreNLPServer','-port','9000','-timeout','15000'], cwd='stanford-corenlp-full-2016-10-31')
    print()

	# Example for an unseen cafe: Thinking Cup Boston
    # cafe_url = 'https://www.yelp.com/biz/thinking-cup-boston-5'
    cafe_url = 'https://www.yelp.com/biz/3-little-figs-somerville'

	# (1) Get reviews from cafe
    print("Getting review data for cafe...")
    cafe_data = getReviewData(cafe_url)
    print()

	# (2) Get topic info (stars and sentiment scores)
    cafe_stats = getStats(cafe_data)

    # (3) Rank topics by sentiment score
    cafe_topic_sentimentRank = rankBySentiment(cafe_stats)

	# (4) Rank topics by Yelp star rating
    cafe_topic_starRank = rankByStars(cafe_stats)

    # (5) Perform linear regression of sentiment stars on star rating
    print('Producing regression of cafe\'s sentiment on its star rating...')

    cafe_ss_reg = regSentStars(cafe_stats)

    print('slope : ', cafe_ss_reg[0])
    print('intercept : ', cafe_ss_reg[1])
    print('r_squared : ', cafe_ss_reg[2]**2)
    print('p_value : ', cafe_ss_reg[3])
    print('std_err : ', cafe_ss_reg[4])
    print()

	# (6) Generate recommendations for this restaurant
    print('Generating recommendation for cafe...')
    cafe_recommend = recommend(cafe_data, cafe_topic_sentimentRank, cafe_topic_starRank)
    print()

	# (7) Predict star rating from on an unseen review
    print('Predicting star rating from an unseen review...')
    #unseenReview = '''I ordered a hazelnut macchiato and it was delicious! Also ordered the breakfast burrito and it was alright. This place was small and cozy and also packed! It was a great environment. The food and drinks came out quick. Conveniently located just a few blocks from the hotel we stayed at. I would definitely recommend Thinking Cup and I'll be back!'''
    unseenReview = '''absolutely my favorite place to go to on weekdays (the lines go out the door on weekends) -- coffee is consistently great and the sandwiches are fab'''
    print('predicted star rating for unseen review : ' + str(predictStars(cafe_data, unseenReview)))
    print('actual star rating = 5')

if __name__ == '__main__':
    main()






