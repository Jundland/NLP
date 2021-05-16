# 1.01 transform to lower case
def txt_to_lower(df, col):
    """
    txt_to_lower transforms all text to lowercase

    :param df: dataframe containing text to be transformed
    :param col: column in dataframe containing text to be transformed
    :return: returns the transformed dataframe
    """
    df[col] = df[col].str.lower()
    return df


# 1.02 remove punctuation
def txt_no_punct(df, col):
    """
    txt_no_punct replaces all punctuation with a space (leaving just words, numbers, spaces and dashes)
    it then replaces excessive whitespace (i.e. more than one space sequentially) with a single space
    and then trims the text to remove leading/trailing spaces

    :param df: dataframe containing text to be transformed
    :param col: column in dataframe containing text to be transformed
    :return: returns the transformed dataframe
    """
    df[col] = df[col].str.replace("[^\w\s\-]", " ")  # keep w=words s=spaces -= dash
    df[col] = df[col].str.strip()
    return df


# 1.03 replace a char
def txt_replace_char(df, col, char_old, char_new):
    """
    txt_replace_char replaces a specific character defined in the inputs with another character also defined in inputs

    :param df: dataframe containing text to be transformed
    :param col: column in dataframe containing text to be transformed
    :param char_old: character to be replaced in text (e.g. '_')
    :param char_new: character it will be replaced with (e.g. ' ')
    :return: returns the transformed dataframe
    """
    df[col] = df[col].str.replace(char_old, char_new)
    return df


# 1.04 remove whitespace
def txt_no_whitespace(df, col):
    """
    txt_no_whitespace replaces all whitespace (including tabs) with a single space
    and then trims the text to remove leading/trailing spaces

    :param df: dataframe containing text to be transformed
    :param col: column in dataframe containing text to be transformed
    :return: returns the transformed dataframe
    """

    df[col] = df[col].str.replace("\s+", " ")
    df[col] = df[col].str.strip()
    return df


# 1.05 remove numbers
def txt_no_numbers(df, col):
    """
    txt_no_numbers removes all numerical digits (i.e. 0,1,2,3,4,5,6,7,8,9) from text

    :param df: dataframe containing text to be transformed
    :param col: column in dataframe containing text to be transformed
    :return: returns the transformed dataframe
    """
    df[col] = df[col].str.replace("\d+", "")
    return df


# 1.06 replace number digits with text
def txt_replace_numbers(df, col):
    """
    txt_replace_numbers replaces numerical digits with the alphabetical version (e.g. replaces 500 with five hundred)

    :param df: dataframe containing text to be transformed
    :param col: column in dataframe containing text to be transformed
    :return: returns the transformed dataframe
    """
    import nltk
    import inflect

    def replace_numbers(sentence):
        p = inflect.engine()
        words = nltk.word_tokenize(sentence)
        new_words = []
        for word in words:
            if word.isdigit():
                new_word = p.number_to_words(word)
                new_words.append(new_word)
            else:
                new_words.append(word)
        return new_words

    df[col] = df[col].apply(lambda x: " ".join(replace_numbers(x)))
    return df


# 1.07 remove non-ASCII characters
def txt_ascii_only(df, col):
    """
    txt_ascii_only removes all non asci characters from the text

    :param df: dataframe containing text to be transformed
    :param col: column in dataframe containing text to be transformed
    :return: returns the transformed dataframe
    """
    from string import printable

    inc_set = set(printable)
    df[col] = df[col].apply(
        lambda x: "".join([" " if i not in inc_set else i for i in x])
    )
    return df


# 1.08 remove stop words
def txt_no_stopwords(df, col):
    """
    txt_no_stopwords removes all stopwords from the text

    :param df: dataframe containing text to be transformed
    :param col: column in dataframe containing text to be transformed
    :return: returns the transformed dataframe
    """
    from nltk.corpus import stopwords

    stop = stopwords.words("english")
    df[col] = df[col].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    return df


# 1.09 remove custom words
def txt_no_customwords(df, col, exc_set):
    """
    txt_no_customwords removes manually defined words contained in a set

    :param df: dataframe containing text to be transformed
    :param col: column in dataframe containing text to be transformed
    :param exc_set: set containing words to be excluded
    :return: returns the transformed dataframe
    """
    df[col] = df[col].apply(
        lambda x: " ".join(x for x in x.split() if x not in exc_set)
    )
    return df


# 1.10 word counts
def txt_word_counts(df, col):
    """
    txt_word_counts adds high level word count to the dataframe

    :param df: dataframe containing text to be transformed
    :param col: column in dataframe containing text to be transformed
    :return: returns the transformed dataframe
    """
    df["word_ct"] = df[col].apply(lambda x: len(str(x).split(" ")))
    return df


# 1.11 lemmatize (wordnet + POS)
def txt_lemmatize(df, col):
    """
    txt_lemmatize uses the WordNet lemmatizer to lemmatize text

    :param df: dataframe containing text to be transformed
    :param col: column in dataframe containing text to be transformed
    :return: returns the transformed dataframe
    """
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.stem.wordnet import WordNetLemmatizer
    from nltk.corpus import wordnet

    lemmatizer = nltk.stem.WordNetLemmatizer()
    wordnet_lemmatizer = WordNetLemmatizer()

    def nltk_tag_to_wordnet_tag(nltk_tag):
        if nltk_tag.startswith("J"):
            return wordnet.ADJ
        if nltk_tag.startswith("V"):
            return wordnet.VERB
        if nltk_tag.startswith("N"):
            return wordnet.NOUN
        if nltk_tag.startswith("R"):
            return wordnet.ADV
        else:
            return None

    def lemmatize_sentence(sentence):
        # tokenize the sentence and find the POS for each token
        nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
        # typle of token, wordnet tag
        wordnet_tagged = map(
            lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged
        )
        lemmatized_sentence = []
        for word, tag in wordnet_tagged:
            if tag is None:
                # if no tag then append word as-is
                lemmatized_sentence.append(word)
            else:
                lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
        return " ".join(lemmatized_sentence)

    df[col] = df[col].apply(lambda x: lemmatize_sentence(x))
    return df


# 1.12 stemming (snowball)
def txt_stemmer(df, col):
    """
    txt_stemmer uses the Snowball stemmer to stem the text

    :param df: dataframe containing text to be transformed
    :param col: column in dataframe containing text to be transformed
    :return: returns the transformed dataframe
    """

    def stem_sentence(sentence):
        import nltk
        from nltk.stem import SnowballStemmer

        stemmer = SnowballStemmer(language="english")
        words = nltk.word_tokenize(sentence)
        new_words = []
        for word in words:
            x = stemmer.stem(word)
            new_words.append(x)
        return new_words

    df[col] = df[col].apply(lambda x: " ".join(stem_sentence(x)))
    return df


# 1.13 filtering parts of speech
def txt_filter_pos(df, col, pos_set):
    """
    txt_filter_pos uses TextBlob part of speech tags to filter the text for specific pos (e.g. adjectives)

    :param df: dataframe containing text to be transformed
    :param col: column in dataframe containing text to be transformed
    :return: returns the transformed dataframe
    """

    def filter_pos(text, pos_set):
        from textblob import TextBlob

        blob = TextBlob(text)
        return [word for (word, tag) in blob.tags if tag in pos_set]

    df[col] = df[col].apply(lambda x: " ".join(filter_pos(x, pos_set)))
    return df
