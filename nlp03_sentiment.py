# 3.1 TextBlob sentiment
def txt_sentiment_tb(df, col):
    import nltk
    from textblob import TextBlob
    import pandas as pd

    df[["polarity", "subjectivity"]] = df[col].apply(
        lambda Text: pd.Series(TextBlob(Text).sentiment)
    )
    return df


# 3.1 VADER sentiment
def txt_sentiment_vd(df, col):
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    analyser = SentimentIntensityAnalyzer()
    df["sent_neg"] = df[col].apply(lambda x: analyser.polarity_scores(x)["neg"])
    df["sent_neu"] = df[col].apply(lambda x: analyser.polarity_scores(x)["neu"])
    df["sent_pos"] = df[col].apply(lambda x: analyser.polarity_scores(x)["pos"])
    df["compound"] = df[col].apply(lambda x: analyser.polarity_scores(x)["compound"])
    return df
