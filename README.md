## a repo for everyday Natural Language Processing in Python
 
 
### nlp01_preproc.py
functions for preprocessing text in df
- 1.1 transform to lowercase
- 1.2 removing punctuation
- 1.3 removing specific chars
- 1.4 removing whitespace
- 1.5 removing numbers
- 1.6 replacing numeric digits with text
- 1.7 removing non-ASCI chars
- 1.8 removing stop words
- 1.9 removing custom words
- 1.10 adding word counts
- 1.11 lemmatizing (wordnet + pos tag)
- 1.12 stemming (snowball)
- 1.13 parts of speech filter

## nlp02_contractions.py
Contractions are words or combinations of words that are shortened by dropping letters and replacing them by an apostrophe.
- 2.1 contractions using dictionary
- 2.2 expanding using contractions library
- 2.3 expanding using pycontractions library

## nlp03_sentiment.py
- 3.1 TextBlob
   - Outputs Polarity and Subjectivity
   - Polarity is a float between [-1, 1].  -1 indicates negative sentiment, and +1 indicates positive sentiment
   - Subjectivity is a float between [0, 1], relating to the level of personal opinion, emotion or judgment in the text
- 3.2 VADER (Valence Aware Dictionary and Sentiment Reasoner)
   - Outputs pos, neu, neg which is proportion of text attributed to positive, neutral or negative sentiment
   - Also outputs compound in the range [-1, 1], which is the normalized score of all lexicon ratings

Notes on TextBlob and VADER sentiment
- TextBlob may be preferred for formal text, and VADER for more informal text or social media
- neither TextBlob or VADER differentiate based on upper/lower casing of text
- TextBlob will differentiate sentiment based on punctuation (e.g. !) but not for repeated punctuation (e.g. !!!)
- VADER will also differentiate sentiment based on repeated punctuation (e.g. ! vs !!!)
