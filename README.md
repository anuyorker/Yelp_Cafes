# Generating Cafe Suggestions with Topic Extraction and Sentiment Analysis of Yelp Reviews

## Abstract
Reviews for businesses and restaurants on Yelp are typically used by consumers to determine where to best spend their money. However, reviews also offer important insights for businesses to leverage. Re- views can show what a business is doing right and what it needs to improve upon. Naive approaches to discovering insights from reviews require an impractical amount of text to iterate through, often skewed by text outliers that would distract a business from implementing meaningful changes. With hopes of circumventing these issues, this project aims to use Latent Dirichlet Allocation (LDA) to pinpoint areas of improvement relevant to a business. Along with topic extraction, we incorporate sentiment analysis to generate recommendations for cafes.

## Introduction
*Motivation*
Yelp currently has "tips" from users to help other users. However, we believe there is useful information that restaurants can obtain from their customers’ reviews. More specifically, features or subtopics can be extracted from reviews to discover the distinct topics most relevant to the restaurant (i.e. what its customers are saying about it).

The reviews of a business on Yelp can yield key ideas for business improvements. The discovery of these core concepts can help a business improve its ratings. Without the ability to acquire actionable in- sights from review text, however, businesses are left guessing about their flaws. The goal of this project’s analysis is to provide business owners a clear target for the pursuit of improving their ratings.

*Dataset*
The dataset that was utilized to address this problem was the Yelp Academic Dataset. This dataset includes 4.1 million reviews by users for businesses and 1.1 million business attributes. For this project, the businesses were filtered by the business category "Coffee & Tea", or cafes. We narrowed it down to cafes so that we could find meaningful subtleties between a particular category of restaurants, rather than extract generic topics across all restaurants (e.g. vague topics like "lunch" or "dinner"). Our hypothesis was that inspecting the cafe category would reveal influential aspects specific to cafe businesses such as barista wait time, specialty drinks, and bakery items.

*Methods*
The model we’ve devised to help businesses to improve their service incorporates two main components:
1. **Latent Dirichlet Allocation (LDA)**: The trained LDA model can then be used to determine the subtopics contained in input reviews. LDA is used to lower dimensionality of the large text data of the reviews and extract their latent subtopics.

2. **Sentiment Treebank**: The next step involves analyzing the reviews sentiment in order to assess how the individuals feel about the topics extracted from their reviews. The ranking of the topics based on the sentiment of the reviews can then be used to suggest improvements that businesses can make.
