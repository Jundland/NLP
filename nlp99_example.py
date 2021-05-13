from nlp01_preproc import txt_to_lower
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
    ["this app is great!!!", 12],
    ["this app is GREAT!!!", 13],
    ["connect connected connection connections connects", 14],
    ["trouble troubled troubles troublesome", 15],
    ["123, 1 2 3", 16],
]

df_clean = pd.DataFrame(df_raw, columns=["Comment", "ID"])
df_clean["text_clean"] = df_clean["Comment"]

df_clean = txt_to_lower(df_clean, "text_clean")
df_clean = txt_no_punct(df_clean, "text_clean")
df_clean = txt_no_underscore(df_clean, "text_clean")
df_clean = txt_no_whitespace(df_clean, "text_clean")
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
