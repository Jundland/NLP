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
        lambda x: " ".join(x for x in x.split() if x not in exc_list)
    )
    return df
