# 1.1 transform to lower case
def txt_to_lower(df, col):
    df[col] = df[col].str.lower()
    return df


# 1.2 remove punctuation
def txt_no_punct(df, col):
    df[col] = df[col].str.replace("[^\w\s\-]", " ")  # keep w=words s=spaces -= dash
    return df


# 1.3 remove underscores
def txt_no_underscore(df, col):
    df[col] = df[col].str.replace("_", " ")
    return df


# 1.4 remove whitespace
def txt_no_whitespace(df, col):
    df[col] = df[col].str.replace("  ", " ")
    return df


# 1.5 remove numbers
def txt_no_numbers(df, col):
    df[col] = df[col].str.replace("\d+", "")
    return df


# 1.6 replace number digits with text
def txt_replace_numbers(df, col):
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


# 1.7 remove non-ASCI
def txt_asci_only(df, col):
    from string import printable

    inc_set = set(printable)
    df[col] = df[col].apply(
        lambda x: "".join([" " if i not in inc_set else i for i in x])
    )
    return df


# 1.8 remove stop words
def txt_no_stopwords(df, col):
    from nltk.corpus import stopwords

    stop = stopwords.words("english")
    df[col] = df[col].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    return df


# 1.9 remove custom words
def txt_no_customwords(df, col, exc_set):
    df[col] = df[col].apply(
        lambda x: " ".join(x for x in x.split() if x not in exc_set)
    )
    return df


# 1.10 word counts
def txt_word_counts(df, col):
    df["word_ct"] = df[col].apply(lambda x: len(str(x).split(" ")))
    return df


# 1.11 lemmatize (wordnet + POS)
def txt_lemmatize(df, col):
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
    def filter_pos(text, pos_set):
        from textblob import TextBlob

        blob = TextBlob(text)
        return [word for (word, tag) in blob.tags if tag in pos_set]

    df[col] = df[col].apply(lambda x: " ".join(filter_pos(x, pos_set)))
    return df
