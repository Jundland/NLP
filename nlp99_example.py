from nlp01_preproc import txt_no_punct, txt_single_spaces
from nlp02_contractions import txt_contractions
from nlp03_sentiment import txt_sentiment_tb, txt_sentiment_vd
import pandas as pd

df_raw = [
    ["i was running....something", 1],
    ["i'm a runner", 2],
    ["he, runs", 3],
    ["cool-runnings is a good film", 4],
    ["i saw_runners", 5],
    ["#great!", 6],
    ["great!", 7],
    ["great!!!!", 8],
    ["GREAT!", 9],
    ["this app is great", 10],
    ["this app is great!", 11],
    ["this app is great!!! I love it", 12],
    ["this app is GREAT!!!", 13],
    ["connect connected connection connections connects", 14],
    ["trouble troubled troubles troublesome", 15],
    ["123, 1 2 3", 16],
    ["he must've", 17],
]

df_clean = pd.DataFrame(df_raw, columns=["Comment", "ID"])
df_clean["text_clean"] = df_clean["Comment"]

df_clean = txt_no_punct(df_clean, "text_clean")
df_clean = txt_single_spaces(df_clean, "text_clean")
df_clean


df_clean = txt_sentiment_tb(df_clean, "text_clean")
df_clean = txt_sentiment_vd(df_clean, "text_clean")

df_clean = txt_to_lower(df_clean, "text_clean")
df_clean = txt_contractions(df_clean, "text_clean")

df_clean = txt_replace_char(df_clean, "text_clean", "_", " ")
df_clean = txt_replace_char(df_clean, "text_clean", "#", " ")
df_clean = txt_no_numbers(df_clean, "text_clean")
df_clean = txt_replace_numbers(df_clean, "text_clean")
df_clean = txt_asci_only(df_clean, "text_clean")
df_clean = txt_no_stopwords(df_clean, "text_clean")
exc_set = ["running", "great"]
df_clean = txt_no_customwords(df_clean, "text_clean", exc_set)
df_clean = txt_word_counts(df_clean, "text_clean")
df_clean = txt_stemmer(df_clean, "text_clean")
df_clean = txt_lemmatize(df_clean, "text_clean")
pos_set = ["JJ"]
df_clean = txt_filter_pos(df_clean, "text_clean", pos_set)

df_clean


import nltk

outeracc = nltk.word_tokenize("he must've done, something bad")
outeracc

outeracc = ("he must've done, something bad").split
