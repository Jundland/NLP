# 3.1 TextBlob sentiment
def txt_sentiment_tb(df, col):
    """
    txt_sentiment_tb uses TextBlob sentiment analyis to add 'polarity' and 'subjectivity' columns to the df
    Polarity is a float between [-1, 1]. -1 indicates negative sentiment, and +1 indicates positive sentiment
    Subjectivity is a float between [0, 1], relating to the level of personal opinion, emotion or judgment in the text

    :param df: dataframe containing text to be transformed
    :param col: column in dataframe containing text to be transformed
    :return: returns the transformed dataframe
    """
    import nltk
    from textblob import TextBlob
    import pandas as pd

    df[["polarity", "subjectivity"]] = df[col].apply(
        lambda Text: pd.Series(TextBlob(Text).sentiment)
    )
    return df


# 3.1 VADER sentiment
def txt_sentiment_vd(df, col):
    """
    txt_sentiment_vd uses VADER sentiment analyis to add 'neg', 'neu', 'pos' and 'compound' sentiment columns to the df
    Outputs pos, neu, neg which is proportion of text attributed to positive, neutral or negative sentiment
    Also outputs compound in the range [-1, 1], which is the normalized score of all lexicon ratings

    :param df: dataframe containing text to be transformed
    :param col: column in dataframe containing text to be transformed
    :return: returns the transformed dataframe
    """
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    analyser = SentimentIntensityAnalyzer()
    df["neg"] = df[col].apply(lambda x: analyser.polarity_scores(x)["neg"])
    df["neu"] = df[col].apply(lambda x: analyser.polarity_scores(x)["neu"])
    df["pos"] = df[col].apply(lambda x: analyser.polarity_scores(x)["pos"])
    df["compound"] = df[col].apply(lambda x: analyser.polarity_scores(x)["compound"])
    return df
