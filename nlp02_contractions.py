# 2.1 contractions using dictionary
def txt_contractions(df, col):

    cont_dict = {
        "ain't": "is not",
        "aint": "is not",
        "aren't": "are not",
        "arent": "are not",
        "can't": "cannot",
        "cant": "cannot",
        "can't've": "cannot have",
        "cant've": "cannot have",
        "'cause": "because",
        "could've": "could have",
        "couldve": "could have",
        "couldn't": "could not",
        "couldnt": "could not",
        "couldn't've": "could not have",
        "couldnt've": "could not have",
        "didn't": "did not",
        "didnt": "did not",
        "doesn't": "does not",
        "doesnt": "does not",
        "don't": "do not",
        "dont": "do not",
        "hadn't": "had not",
        "hadnt": "had not",
        "hadn't've": "had not have",
        "hadnt've": "had not have",
        "hasn't": "has not",
        "hasnt": "has not",
        "haven't": "have not",
        "havent": "have not",
        "he'd": "he would",
        "he'd've": "he would have",
        "he'll": "he will",
        "he'll've": "he he will have",
        "he's": "he is",
        "how'd": "how did",
        "howd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how is",
        "hows": "how is",
        "I'd": "I would",
        "Id": "I would",
        "I'd've": "I would have",
        "I'll": "I will",
        "I'll've": "I will have",
        "I'm": "I am",
        "Im": "I am",
        "I've": "I have",
        "i'd": "i would",
        "i'd've": "i would have",
        "i'll": "i will",
        "i'll've": "i will have",
        "i'm": "i am",
        "im": "i am",
        "i've": "i have",
        "isn't": "is not",
        "isnt": "is not",
        "it'd": "it would",
        "it'd've": "it would have",
        "it'll": "it will",
        "it'll've": "it will have",
        "it's": "it is",
        "its": "it is",
        "let's": "let us",
        "lets": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightve": "might have",
        "mightn't": "might not",
        "mightnt": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustve": "must have",
        "mustn't": "must not",
        "mustnt": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "neednt": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtnt": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "shant": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she would",
        "she'd've": "she would have",
        "she'll": "she will",
        "she'll've": "she will have",
        "she's": "she is",
        "shes": "she is",
        "should've": "should have",
        "shouldve": "should have",
        "shouldn't": "should not",
        "shouldnt": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so as",
        "that'd": "that would",
        "thatd": "that would",
        "that'd've": "that would have",
        "that's": "that is",
        "thats": "that is",
        "there'd": "there would",
        "thered": "there would",
        "there'd've": "there would have",
        "there's": "there is",
        "theres": "there is",
        "they'd": "they would",
        "theyd": "they would",
        "they'd've": "they would have",
        "they'll": "they will",
        "theyll": "they will",
        "they'll've": "they will have",
        "they're": "they are",
        "theyre": "they are",
        "they've": "they have",
        "theyve": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "wasnt": "was not",
        "we'd": "we would",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weve": "we have",
        "weren't": "were not",
        "werent": "were not",
        "what'll": "what will",
        "whatll": "what will",
        "what'll've": "what will have",
        "what're": "what are",
        "what's": "what is",
        "whats": "what is",
        "what've": "what have",
        "whatve": "what have",
        "when's": "when is",
        "whens": "when is",
        "when've": "when have",
        "where'd": "where did",
        "whered": "where did",
        "where's": "where is",
        "wheres": "where is",
        "where've": "where have",
        "who'll": "who will",
        "wholl": "who will",
        "who'll've": "who will have",
        "who's": "who is",
        "whos": "who is",
        "who've": "who have",
        "why's": "why is",
        "whys": "why is",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "wont": "will not",
        "won't've": "will not have",
        "wont've": "will not have",
        "would've": "would have",
        "wouldve": "would have",
        "wouldn't": "would not",
        "wouldnt": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "yall": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you would",
        "youd": "you would",
        "you'd've": "you would have",
        "youd've": "you would have",
        "you'll": "you will",
        "youll": "you will",
        "you'll've": "you will have",
        "you're": "you are",
        "youre": "you are",
        "you've": "you have",
        "youve": "you have",
    }

    df[col] = df[col].apply(
        lambda x: " ".join(cont_dict[x] if x in cont_dict else x for x in x.split())
    )
    return df


# 2.2 expanding using contractions library
# https://www.geeksforgeeks.org/nlp-expand-contractions-in-text-processing/

# 2.3 expanding using pycontractions library
# https://pypi.org/project/pycontractions/
