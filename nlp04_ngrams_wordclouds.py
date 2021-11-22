def txt_ngrams(df, col, ngram_size):
    """
    txt_ngrams uses word_tokenize from nltk to split a text column of a dataframe into n-gram combinations

    :param df: dataframe containing text to be transformed
    :param col: column in dataframe containing text to be transformed
    :ngram_size: number of words to tokenize/split (1 = unigram, 2 = bigram etc)
    :return: returns the transformed dataframe
    """
    from nltk import word_tokenize, ngrams

    if ngram_size == 1:
        df["ngrams"] = df[col].apply(word_tokenize)
    else:
        df["ngrams"] = df[col].apply(
            lambda x: ["_".join(ng) for ng in ngrams(word_tokenize(x), ngram_size)]
        )

    return df


def txt_wordcloud(df, col):
    """
    txt_wordcloud uses wordcloud and matplotlib to plot a wordcloud of ngram results

    :param df: dataframe containing text to be transformed
    :param col: column in dataframe containing text to be transformed
    :return: returns the wordcloud plot
    """
    from wordcloud import WordCloud
    from matplotlib.pyplot as plt

    ngrams=df[col]

    wordcloud = WordCloud(
        width = 2000,
        height = 1500,
        background_color = 'white').generate(str(ngrams))
    fig = plt.figure(
        figsize = (30, 20),
        facecolor = 'k',
        edgecolor = 'k')
    plt.imshow(wordcloud, interpolation = 'biliner')
    plt.axis('off')
    plt.tight_layout(pad = 0)

    return fig