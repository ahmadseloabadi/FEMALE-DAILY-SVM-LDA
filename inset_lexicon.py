import csv
import streamlit as st
import pandas as pd

@st.cache_data
def load_data():
    data = pd.read_csv('data/dataset/Hasil.csv', encoding='latin-1')
    return data

data = load_data()
datas=pd.DataFrame(data)
datas.drop(datas.tail(3).index, inplace=True)

def pelabelan(kalimat):
    lexicon_positive = dict()
    with open('data/kamus/positive.tsv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for row in reader:
            if "word" not in row and "weight" not in row:
                lexicon_positive[row[0]] = int(row[1])

    lexicon_negative = dict()
    with open('data/kamus/negative.tsv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for row in reader:
            if "word" not in row and "weight" not in row:
                lexicon_negative[row[0]] = int(row[1])

    def inset_lexicon(text):
        word_scores = {}
        score= 0
        for word in text:
            if word in lexicon_positive:
                score += lexicon_positive[word]
                word_scores[word] = lexicon_positive[word]
            if word in lexicon_negative:
                score += lexicon_negative[word]
                word_scores[word] =lexicon_negative[word]
            if (word in lexicon_negative) and (word in lexicon_positive):
                word_scores[word] = lexicon_negative[word] + lexicon_positive[word]
        polarity=''
        if (score >= 0):
            polarity = 'positive'
        else:
            polarity = 'negative'
        return word_scores, score, polarity
    results = inset_lexicon(kalimat)
    st.write('Nilai setiap kata :',str(results[0]))
    st.write('Jumlah total :',results[1])
    st.write('Hasil label sentimen :',results[2])


def import_and_display_tsv(name):
    df = pd.read_csv('data/kamus/'+name+'.tsv', sep='\t')
    st.dataframe(df,use_container_width=True)

