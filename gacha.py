import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer ,CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, classification_report, accuracy_score, confusion_matrix,recall_score,precision_score

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
import seaborn as sns

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models.ldamodel import LdaModel

import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

import csv
import joblib
import re
import string

st.title("PENERAPAN METODE SUPPORT VECTOR MACHINE UNTUK ANALISIS SENTIMEN BERBASIS ASPEK PADA ULASAN PRODUK SKINCARE DENGAN PEMODELAN LATENT DIRICHLET ALLOCATION (LDA)")
st.sidebar.title("Analisis Sentimen Berbasis Aspek Pada Ulasan Produk Skincare")


st.set_option('deprecation.showPyplotGlobalUse', False)

# @st.cache(persist=True) deprecated
@st.cache_data
def load_data():
    data = pd.read_csv('data/dataset/Hasil.csv', encoding='latin-1')
    return data

data = load_data()

######################### Initiate SVM #########################
optimal_model = joblib.load('data/model/topic_model.joblib')
bigram_mod = joblib.load('data/model/bigram_model.joblib')
id2word = joblib.load('data/model/id2word_model.joblib')

dataset=pd.read_csv('data/prepro/data_cleaned.csv')
dataset.drop(dataset.tail(3).index, inplace=True) 
tfidf_vectorizer = TfidfVectorizer(max_features=2000, max_df=0.75, min_df=5)  # Tambahkan max_df dan min_df


@st.cache_data
def data_SMOTE(data):
    data['polarity_score'] = data['polarity_score'].apply(lambda x: 1 if x > 0 else 0)
    x_res =tfidf_vectorizer.fit_transform(data['content_preprocessing'])
    y_res =data['polarity_score']
    smote = SMOTE(sampling_strategy='auto')
    X_smote, Y_smote = smote.fit_resample(x_res, y_res)
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_smote, Y_smote, test_size=0.20)
    print('smote',len(yc_test))
    return Xc_train, Xc_test, yc_train, yc_test

@st.cache_data
def dataset_preparation(datase):
    datase['polarity_score'] = datase['polarity_score'].apply(lambda x: 1 if x > 0 else 0)
    X_tfidfi = tfidf_vectorizer.fit_transform(datase['content_preprocessing'])
    X_train, X_test, y_train, y_test = train_test_split(X_tfidfi, datase['polarity_score'], test_size=0.2)
    print('biasa',len(y_test))
    return X_train, X_test, y_train, y_test
    
######################### Start Normalisasi #########################
def cleaningText(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text) # remove mentions
    text = re.sub(r'#[A-Za-z0-9]+', '', text) # remove hashtag
    text = re.sub(r'RT[\s]', '', text) # remove RT
    text = re.sub(r"http\S+", '', text) # remove link
    text = re.sub(r'[0-9]+', '', text) # remove numbers
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Menghapus semua simbol
    emoji_pattern = re.compile("["
                    u"\U0001F600-\U0001F64F"  # emoticons
                    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                    u"\U0001F680-\U0001F6FF"  # transport & map symbols
                    u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                    u"\U0001F700-\U0001F77F"  # alchemical symbols
                    u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                    u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                    u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                    u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                    u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                    u"\U0001F004-\U0001F0CF"  # Additional emoticons
                        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    text = text.replace('\n', ' ')
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip(' ')

    return text

def casefoldingText(text):
    text = text.lower()
    return text

def tokenizingText(text):
    text = word_tokenize(text)
    return text

normalisasi = pd.read_csv('data/kamus/normalisasi.csv', encoding='latin1')
normalisasi_text_dict = {}
for index, row in normalisasi.iterrows():
    if row[0] not in normalisasi_text_dict:
        normalisasi_text_dict[row[0]] = row[1]

def normalisasiText(text):
    return [normalisasi_text_dict[term] if term in normalisasi_text_dict else term for term in text]

def filteringText(text):
    listStopwords = StopWordRemoverFactory().get_stop_words()
    filtered = []
    for txt in text:
        if txt not in listStopwords:
            filtered.append(txt)
    text = filtered
    return text

def stemmingText(text):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    text = [stemmer.stem(word) for word in text]
    return text

X_train, X_test, y_train, y_test=dataset_preparation(dataset)
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)

# Save the model to a file
model_filename = 'svm_sebelum_smote.joblib'
joblib.dump(svm_classifier, model_filename)



y_pred = svm_classifier.predict(X_test)

Xc_train, Xc_test, yc_train, yc_test=data_SMOTE(dataset)
svm = SVC(kernel='linear')
svm.fit(Xc_train, yc_train)
# Save the model to a file
model_filenames = 'svm_sesudah_smote.joblib'
joblib.dump(svm, model_filenames)

yc_pred=svm.predict(Xc_test)

@st.cache_data
def pengujian_nilai_c():
    c_values = [0.5,1,5,10,25,50,100]
    # inisialisasi list untuk menyimpan hasil
    results = []

    # ulangi model SVM dengan setiap nilai C dan gamma pada rentang yang ditentukan
    for c in c_values:

        svm_c = SVC(kernel='linear',C=c, probability=True)
        svm_c.fit(X_train, y_train)
        y_predc = svm_c.predict(X_test)
        # hitung metrik evaluasi model
        accuracy = accuracy_score(y_test, y_predc)*100
        precision = precision_score(y_test, y_predc)*100
        recall = recall_score(y_test, y_predc)*100
        # tambahkan hasil ke dalam list
        results.append({'c': c, 'accuracy': accuracy})
    results=pd.DataFrame(results)
    return results

@st.cache_data(experimental_allow_widgets=True)
def plot_sentiment_label(datas):
    select = st.selectbox('Pilih Tipe Visualisasi', ['Bar plot', 'Pie chart'], key='visualization_type')

    sentiment_count = datas['polarity'].value_counts()
    sentiment_count = pd.DataFrame({'Sentiment':sentiment_count.index, 'Reviews':sentiment_count.values})
    
    st.markdown("### Tampilan diagram jumlah ulasan setiap sentimen")
    if select == 'Bar plot':
        fig = px.bar(sentiment_count, x='Sentiment', y='Reviews', color='Reviews', height=500)
        st.plotly_chart(fig)
    if select == 'Pie chart':
        fig = px.pie(sentiment_count, values='Reviews', names='Sentiment')
        st.plotly_chart(fig)

def tf_idf(dataset):
    kolom_ulasan=dataset['content_preprocessing']
    vectorizer = CountVectorizer()
    word_count_vector = vectorizer.fit_transform(kolom_ulasan)

    # normalize TF vector

    print("tf")
    tf = pd.DataFrame(word_count_vector.toarray(), columns=vectorizer.get_feature_names_out())
   
    st.write('Tampilan TF')
    st.dataframe(tf.transpose())

    tfidf_transformer = TfidfTransformer()
    X = tfidf_transformer.fit_transform(word_count_vector)
    idf = pd.DataFrame({'term':vectorizer.get_feature_names_out(), 'IDF':tfidf_transformer.idf_})
    st.write('Tampilan IDF')
    st.dataframe(idf)

    tf_idf = pd.DataFrame(X.toarray() ,columns=vectorizer.get_feature_names_out())
    st.write('Tampilan Pembobotan TF-IDF')
    st.dataframe(tf_idf.transpose())

    # Transformasi content_preprocessing menggunakan TF-IDF
    tfidf = tfidf_vectorizer.fit_transform(datas['content_preprocessing'])

    # Mendapatkan daftar kata yang digunakan dalam TF-IDF
    feature_names = tfidf_vectorizer.get_feature_names_out()

    # Membuat DataFrame kosong untuk menyimpan nilai TF-IDF
    tfidf_df = pd.DataFrame(columns=['TF-IDF'])

    # Mengisi DataFrame dengan nilai TF-IDF yang tidak nol
    for i, doc in enumerate(datas['content_preprocessing']):
        doc_tfidf = tfidf[i]
        non_zero_indices = doc_tfidf.nonzero()[1]
        tfidf_values = doc_tfidf[0, non_zero_indices].toarray()[0]
        tfidf_dict = {feature_names[idx]: tfidf_values[j] for j, idx in enumerate(non_zero_indices)}
        tfidf_df.loc[i] = [' '.join(f'({feature_name}, {tfidf_dict[feature_name]:.3f})' for feature_name in tfidf_dict)]

    # Menggabungkan DataFrame hasil dengan DataFrame utama
    dataset = pd.concat([dataset, tfidf_df], axis=1)
    st.write('Tampilan Pembobotan TF-IDF Pada Dataset')
    st.dataframe(dataset[['content','content_preprocessing','TF-IDF','sentiment']])

def split_Data(datas):
    ulasan=datas[['content']]
    # Transformasi content_preprocessing menggunakan TF-IDF
    tfidf = tfidf_vectorizer.fit_transform(datas['content_preprocessing'])

    # Mendapatkan daftar kata yang digunakan dalam TF-IDF
    feature_names = tfidf_vectorizer.get_feature_names_out()

    # Membuat DataFrame kosong untuk menyimpan nilai TF-IDF
    tfidf_df = pd.DataFrame(columns=['TF-IDF'])

    # Mengisi DataFrame dengan nilai TF-IDF yang tidak nol
    for i, doc in enumerate(datas['content_preprocessing']):
        doc_tfidf = tfidf[i]
        non_zero_indices = doc_tfidf.nonzero()[1]
        tfidf_values = doc_tfidf[0, non_zero_indices].toarray()[0]
        tfidf_dict = {feature_names[idx]: tfidf_values[j] for j, idx in enumerate(non_zero_indices)}
        tfidf_df.loc[i] = [' '.join(f'({feature_name}, {tfidf_dict[feature_name]:.3f})' for feature_name in tfidf_dict)]

    # Menggabungkan DataFrame hasil dengan DataFrame utama
    ulasan = pd.concat([ulasan, tfidf_df], axis=1)

    x=datas['content_preprocessing']
    y=datas['sentiment']
    Xs_train, Xs_test, ys_train, ys_test = train_test_split(x, y, test_size=0.2)
    
    data_train = pd.DataFrame({'content_preprocessing': Xs_train, 'sentiment': ys_train})
    data_test=pd.DataFrame({'content_preprocessing': Xs_test, 'sentiment': ys_test})
    # Menambahkan kolom index asli ke dataset pertama
    ulasan['index'] = ulasan.index
    # Menambahkan kolom index asli ke dataset pertama
    data_train['index'] = data_train.index
    # Menambahkan kolom index asli ke dataset pertama
    data_test['index'] = data_test.index
    
    train_df = pd.merge(ulasan, data_train, on='index')
    test_df = pd.merge(ulasan, data_test, on='index')
    

    st.write(f'Dari keseluruan dataset yang berjumlah {len(datas)} dilakukan pembagian data train dan data test dengan perbandingan 80:20 sehingga dapat dilihat hasil pemgaian sebagai berikut')
    
    label_train=data_train['sentiment'].value_counts()
    
    st.write(f'Tampilan data training dengan jumlah data sebanyak {len(Xs_train)} dengan jumlah kelas positive sebanyak {label_train[1]} dan kelas negative {label_train[0]}')
    st.dataframe(train_df[['content','content_preprocessing','TF-IDF','sentiment']],use_container_width=True)


    label_test=data_test['sentiment'].value_counts()
    
    st.write(f'Tampilan data test dengan jumlah data sebanyak {len(Xs_test)} dengan jumlah kelas positive sebanyak {label_test[1]} dan kelas negative {label_test[0]}')
    st.dataframe(test_df[['content','content_preprocessing','TF-IDF','sentiment']],use_container_width=True)

@st.cache_data(experimental_allow_widgets=True)
def smote(df):
    
    vectorizer = TfidfVectorizer(max_features=2000, max_df=0.75, min_df=5)  # Tambahkan max_df dan min_df
    df['polarity_score'] = df['polarity_score'].apply(lambda x: 1 if x > 0 else 0)

    x_res =vectorizer.fit_transform(df['content_preprocessing'])
    y_res =df['polarity_score']

    #penerapan smote
    smote = SMOTE(sampling_strategy='auto')
    X_smote, Y_smote = smote.fit_resample(x_res, y_res)
    st.title('SMOTE')
    st.text('SMOTE adalah teknik untuk mengatasi ketidak seimbangan kelas pada dataset')
    
    seb_smote,ses_smote = st.columns(2)
    with seb_smote:
        st.header('sebelum SMOTE')
        st.write('Jumlah dataset:',len(dataset))
        # Hitung jumlah kelas sebelum SMOTE
        st.write("Jumlah kelas:  ")
        Jum_sentimen = dataset['polarity_score'].value_counts()
        spos, sneg = st.columns(2)
        with spos:
            st.markdown("Positive")
            st.markdown(f"<h1 style='text-align: center; color: blue;'>{Jum_sentimen[1]}</h1>", unsafe_allow_html=True)
        with sneg:
            st.markdown("negative")
            st.markdown(f"<h1 style='text-align: center; color: orange;'>{Jum_sentimen[0]}</h1>", unsafe_allow_html=True)
        # menampilkan dalam bentuk plot diagram
        labels = ['negative' , 'positive']
        fig2,ax2=plt.subplots()
        plt.pie(dataset.groupby('polarity_score')['polarity_score'].count(), autopct=" %.1f%% " ,labels=labels)
        ax2.axis('equal')
        st.pyplot(fig2)

    with ses_smote:
        st.header('sesudah SMOTE')
        df_smote = pd.DataFrame(X_smote)
        df_smote.rename(columns={0:'content_preprocessing'}, inplace=True)
        df_smote['sentiment'] = Y_smote
        #melihat banyak dataset
        st.write('Jumlah dataset :',len(Y_smote))
        # melihat jumlah kelas sentimen aetelah SMOTE
        st.write("Jumlah kelas: ")
        Jumlah_sentimen = df_smote['sentiment'].value_counts()
        pos, neg = st.columns(2)
        with pos:
            st.markdown("Positive")
            st.markdown(f"<h1 style='text-align: center; color: blue;'>{Jumlah_sentimen[1]}</h1>", unsafe_allow_html=True)
        with neg:
            st.markdown("negative")
            st.markdown(f"<h1 style='text-align: center; color: orange;'>{Jumlah_sentimen[0]}</h1>", unsafe_allow_html=True)
        #membuat diagram
        labels = ['negative', 'positive']
        fig1,ax1=plt.subplots()
        ax1.pie(df_smote.groupby('sentiment')['sentiment'].count(), autopct=" %.1f%% " ,labels=labels)
        ax1.axis('equal')
        st.pyplot(fig1)
    sintetis = pd.read_csv('data/smote/data_sintetik.csv')
    dsmote = pd.read_csv('data/smote/data_smote.csv')
    st.header('Data sintetis')
    st.write('jumlah data sintetis :',len(sintetis))
    st.write('jumlah penambahan setiap kelas :')
    sinpos,sinneg = st.columns(2)
    with sinpos:
        st.markdown("Positif")
        sel_pos=Jumlah_sentimen[1]-Jum_sentimen[1]
        st.markdown(f"<h1 style='text-align: center; color: blue;'>{sel_pos}</h1>", unsafe_allow_html=True)
    with sinneg:
        st.markdown("Negatif")
        sel_neg=Jumlah_sentimen[0]-Jum_sentimen[0]
        st.markdown(f"<h1 style='text-align: center; color: orange;'>{sel_neg}</h1>", unsafe_allow_html=True)
   
    def sentimen_smote(dataset, Sentimen):
        return dsmote[dsmote['sentimen'].isin(sentiment)]
    
    optiondata = st.selectbox('pilih data',('SMOTE', 'SINTETIS'))

    if(optiondata == 'SMOTE') :
        # Panggil fungsi pencarian
        st.write('menampilkan data sesudah',optiondata)
        sentiment_map = {1: 'positif',0:'negatif'}
        sentiment = st.multiselect('Pilih kelas sentimen', list(sentiment_map.values()),default=list(sentiment_map.values()))
        sentiment = [key for key, value in sentiment_map.items() if value in sentiment]
        filtered_data = sentimen_smote(dsmote, sentiment)
        st.dataframe(filtered_data)
    if(optiondata == 'SINTETIS') :
        st.write('menampilkan data ',optiondata)
        st.dataframe(sintetis)
    with st.expander('---') :
        st.text('cari kalimat ')
        search_query = st.text_input("Masukkan kalimat yang ingin dicari:")
        search_results = dsmote[dsmote['kalimat_asli'].str.contains(search_query)]
        if st.button('cari') :
            st.dataframe(search_results,use_container_width=True)
        # Mencari baris duplikat berdasarkan nilai kalimat asli
        duplicates = dsmote[dsmote.duplicated(subset='kalimat_asli', keep=False)]
        results = {"index": [], "bobot_kalimat": [], "kalimat_asli": [], "sentimen": []}
        # Menampilkan kalimat yang duplikat
        for index, row in duplicates.iterrows():
            results["index"].append(index)
            results["bobot_kalimat"].append(row['bobot_kalimat'])
            results["sentimen"].append(row['sentimen'])
            results["kalimat_asli"].append(row['kalimat_asli'])

        duplikat = pd.DataFrame(results)
        st.dataframe(duplikat)
######################### data visual #########################
st.sidebar.markdown("### Data Visual ")
datas=pd.DataFrame(data)
datas.drop(datas.tail(3).index, inplace=True)
if not st.sidebar.checkbox("Hide", False, key='hide_data_visual'):
    data_option = st.selectbox('Data Option', ['dataset', 'preprocessing','TF-IDF','pembagian data','SMOTE'], key='data_option')

    if data_option =='dataset':
        st.subheader('Dataset')
        st.write(f'Tampilan dataset dengan total jumlah dataset sebanyak {len(datas)}')
        def filter_sentiment(dataset, selected_sentiment):
            return dataset[dataset['sentiment'].isin(selected_sentiment)]

        sentiment_map = {'positive': 'positive', 'negative': 'negative'}
        selected_sentiment = st.multiselect('Pilih kelas sentimen', list(sentiment_map.keys()), default=list(sentiment_map.keys()))
        filtered_data = filter_sentiment(datas, selected_sentiment)
        
        st.dataframe(filtered_data[['content','sentiment']],use_container_width=True)
        
        plot_sentiment_label(datas)
    if data_option =='preprocessing':
        st.subheader('Preprocessing')
        st.write('Hasil Cleansing')
        st.dataframe(datas[['content','content_clean']],use_container_width=True)
        st.write('Hasil Casefolding')
        st.dataframe(datas[['content','content_folding']],use_container_width=True)
        st.write('Hasil Tokenizing')
        st.dataframe(datas[['content','content_tokenizing']],use_container_width=True)
        st.write('Hasil Normalization')
        st.dataframe(datas[['content','content_normalisasi']],use_container_width=True)
        st.write('Hasil Stopword')
        st.dataframe(datas[['content','content_stopword']],use_container_width=True)
    
        datas['content_stemming'] = datas['content_preprocessing'].apply(lambda x: ' '.join(eval(x)))
        st.write('Hasil Stemming')
        st.dataframe(datas[['content','content_stemming']],use_container_width=True)
    if data_option =='TF-IDF':
        st.subheader('TF-IDF')
        
        tf_idf(datas)
    if data_option =='pembagian data':
        st.subheader('Pembagian Data')
        split_Data(datas)
    
    if data_option =='SMOTE':
        st.subheader('pernerapan SMOTE')
        smote(datas)
    

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

######################### inset lexicon #########################
st.sidebar.subheader("Labeling with Inset Lexicon")
if not st.sidebar.checkbox("Hide", True, key='checkbox_labeling'):
    insetlexicon=pd.read_csv('data/label/inset_lexicon.csv')
    insetlexicon.drop(insetlexicon.tail(3).index, inplace=True)
    st.write('Tampilan data pelabelan Inset Lexicon')
    def filter_sentiment(dataset, selected_sentiment):
            return dataset[dataset['sentiment'].isin(selected_sentiment)]

    sentiment_map = {'positive': 'positive', 'negative': 'negative'}
    selected_sentiment = st.multiselect('Pilih kelas sentimen', list(sentiment_map.keys()), default=list(sentiment_map.keys()),key='sentimen_option')
    filtered_data = filter_sentiment(insetlexicon, selected_sentiment)
    st.dataframe(filtered_data,use_container_width=True)
    kalimat=st.text_input('Masukan Kalimat',value='produk jelek')
    if st.button('labeling'):
        clean_text = cleaningText(kalimat)
        case_text = casefoldingText(clean_text)
        token_text = tokenizingText(case_text)
        norm_text = normalisasiText(token_text)
        stop_text = filteringText(norm_text)
        stem_text = stemmingText(stop_text)
        pelabelan(stem_text)

    with st.expander('kamus lexicon'):
        kamus_option = st.multiselect('pilih kamus inset lexicon', ['positive', 'negative'],['positive', 'negative'], key='kamus')
        for file in kamus_option:
            st.write(f"tampilan data kamus {file}:")
            import_and_display_tsv(file)

@st.cache_data
def plot_sentiment(topic_label):
    df = datas[datas['topic_label']==topic_label]
    count = df['polarity'].value_counts()
    count = pd.DataFrame({'Sentiment':count.index, 'Reviews':count.values.flatten()})
    return count
######################### topic modeling #########################
st.sidebar.subheader("Topic modeling with LDA")

riview_topic_data=data[['content','keywords','topic_label','sentiment']]
review_sentiment_count = datas.groupby('topic_label')['polarity'].count().sort_values(ascending=False)
review_sentiment_count = pd.DataFrame({'Topic Label':review_sentiment_count.index, 'Reviews':review_sentiment_count.values.flatten()})
if not st.sidebar.checkbox("Hide", True, key='close_checkbox_2'):
    st.subheader('Tampilan data hasil topik modeling')
    def filter_topik(dataset, selected_topik):
        return dataset[dataset['topic_label'].isin(selected_topik)]
    topik_map = {'Penggunaan Produk': 'Penggunaan Produk', 'Performa Produk': 'Performa Produk', 'Pembelian Produk': 'Pembelian Produk'}
    selected_topik = st.multiselect('Pilih Aspek', list(topik_map.keys()), default=list(topik_map.keys()),key='topik_option')
    filter_data = filter_topik(riview_topic_data, selected_topik)
    st.dataframe(filter_data,use_container_width=True)
    
    each_airline = st.selectbox('Pilih Tipe Visualisasi', ['Bar plot', 'Pie chart'], key='visualization_type_2')
    if each_airline == 'Bar plot':
        st.subheader("Total ulasan pada setiap label")
        fig_1 = px.bar(review_sentiment_count, x='Topic Label', y='Reviews', color='Reviews', height=500)
        st.plotly_chart(fig_1)
    if each_airline == 'Pie chart':
        st.subheader("Total Total ulasan pada setiap label")
        fig_2 = px.pie(review_sentiment_count, values='Reviews', names='Topic Label')
        st.plotly_chart(fig_2)

    ######################### Breakdown Label Topic by Sentiment #########################
    st.subheader("Aspek berdasarkan sentimen")
    choice = st.multiselect('Pilih Aspek', ('Penggunaan Produk', 'Performa Produk', 'Pembelian Produk'), key='pick_topic_label_1')
    if len(choice) > 0:
        
        breakdown_type = st.selectbox('Pilih Tipe Visualisasi', ['Pie chart', 'Bar plot', ], key='breakdown_type')
        fig_3 = make_subplots(rows=1, cols=len(choice), subplot_titles=choice)
        if breakdown_type == 'Bar plot':
            for i in range(1):
                for j in range(len(choice)):
                    fig_3.add_trace(
                        go.Bar(x=plot_sentiment(choice[j]).Sentiment, y=plot_sentiment(choice[j]).Reviews, showlegend=False),
                        row=i+1, col=j+1
                    )
            fig_3.update_layout(height=600, width=800)
            st.plotly_chart(fig_3)
        else:
            fig_3 = make_subplots(rows=1, cols=len(choice), specs=[[{'type':'domain'}]*len(choice)], subplot_titles=choice)
            for i in range(1):
                for j in range(len(choice)):
                    fig_3.add_trace(
                        go.Pie(labels=plot_sentiment(choice[j]).Sentiment, values=plot_sentiment(choice[j]).Reviews, showlegend=True),
                        i+1, j+1
                    )
            fig_3.update_layout(height=600, width=800)
            st.plotly_chart(fig_3)


######################### Predict Input #########################
st.sidebar.header("Predict Sentiment")
if not st.sidebar.checkbox("Hide", True, key='close_checkbox_4'):
    with st.expander ("...."):
        uji_c=pengujian_nilai_c()
        st.dataframe(uji_c,use_container_width=True)
        st.write('penjelasan nilai C')
        st.write('parameter C (juga dikenal sebagai parameter penalti) adalah faktor penting yang mempengaruhi kinerja dan perilaku model SVM. Parameter C mengendalikan antara penalti kesalahan klasifikasi dan lebar margin. Nilai C yang lebih kecil akan menghasilkan margin yang lebih lebar, dan menjadikan model tidak peka terhadap data dan kelasahan klasifikasi(UNDERFITTING). Sebaliknya, nilai C yang lebih besar akan menghasilkan margin yang lebih sempit, menjadikan model lebih peka terhadap data dan tingkat kesalahan klasifikasi.(OVERFITTING)')
    ######################### Predict Topic #########################
    def label_topics(integer):
        if integer == 0:
            return 'Penggunaan Produk'
        elif integer == 1:
            return 'Performa Produk'
        elif integer == 2:
            return 'Pembelian Produk' 
        else:
            return 'Unknown'
    
    def predict_topic(text):
        processed_text = gensim.utils.simple_preprocess(text, deacc=True)
        processed_text = bigram_mod[processed_text]
        bow = id2word.doc2bow(processed_text)
        topics = optimal_model.get_document_topics(bow)
        dominant_topic = sorted(topics, key=lambda x: x[1], reverse=True)[0]
        topic_num = dominant_topic[0]
        return topic_num
    
    def label_solusi(prediction, integer):
        if prediction == 1:
            if integer == 0:
                return "Produk bagus diterapkan, cocok untuk kulit & wajah"
            elif integer == 1:
                return "Efisiensi produk sangat bagus untuk penerapaan kulit & wajah"
            elif integer == 2:
                return "Produk bagus dan user berminat untuk melakukan pembelian kembali"
            else:
                return "Produk bagus"
        else:
            if integer == 0:
                return "Produk tidak cocok pada wajah & kulit user <br> Mungkin Anda dapat menggunakan produk lain seperti : <br> Mild Purifying Toner atau Centella Water Alcohol-Free Toner"
            elif integer == 1:
                return "Efisiensi produk kurang bagus untuk penerapaan kulit & wajah <br> Mungkin Anda dapat menggunakan produk lain seperti : <br> Madagaskar Centella Toning Toner "
            elif integer == 2:
                return "Produk kurang bagus dan user tidak berminat untuk melakukan pembelian kembali <br> Mungkin Anda dapat menggunakan produk lain seperti : <br> True To Skin Mugwort Cica Essence Toner atau Pure Centella Acne Calming Toner"
            else:
                return "Produk kurang bagus"
    
######################### Predict Sentiment #########################
    def predict_sentiment(text):

        clean_text = cleaningText(text)
        case_text = casefoldingText(clean_text)
        token_text = tokenizingText(case_text)
        norm_text = normalisasiText(token_text)
        stop_text = filteringText(norm_text)
        stem_text = stemmingText(stop_text)
        preprocessed_text = ' '.join(stem_text)
        ######################### End Normalisasi #########################

        text_tfidf = tfidf_vectorizer.transform([preprocessed_text])

        # Mendapatkan daftar kata yang digunakan dalam TF-IDF
        feature_names = tfidf_vectorizer.get_feature_names_out()

        # Inisialisasi list untuk menyimpan nilai TF-IDF
        tfidf_list = []

        # Mendapatkan nilai TF-IDF untuk setiap dokumen dan menyimpannya dalam list
        for i in range(1):
            doc_tfidf = text_tfidf[i]
            doc_features = doc_tfidf.nonzero()[1]
            tfidf_values = doc_tfidf[0, doc_features].toarray()[0]
            tfidf_dict = dict(zip(doc_features, tfidf_values))
            tfidf_doc = ['({},{:.3f})'.format(feature_names[idx], tfidf_dict[idx]) for idx in sorted(tfidf_dict.keys())]
            tfidf_list.append(' '.join(tfidf_doc))

        prediction = svm.predict(text_tfidf)[0]
        # probability = svm.predict_proba(text_tfidf)[0][1]

        prediction_topic = predict_topic(preprocessed_text)
        topic = label_topics(prediction_topic)
        solusi = label_solusi(prediction, prediction_topic)

        st.write('hasil cleasing :',clean_text)
        st.write('hasil casefolding :',case_text)
        st.write('hasil tokenizing :',str(token_text))
        st.write('hasil normalization :',str(norm_text))
        st.write('hasil stopword :',str(stop_text))
        st.write('hasil stemming :',str(stem_text))
        st.write('hasil pembobotan TF-IDF :',str(tfidf_list))

        if prediction == 1:
            st.image("data/img/smile.png", width=100)
            st.write('<p style="color:green;">Topic: <b>{}</b></p>'.format(topic), unsafe_allow_html=True)
            st.write('<p style="color:green;">Sentiment: <b>Positive</b></p>', unsafe_allow_html=True)
            st.write('<p style="color:green;">Solusi: <b>{}</b></p>'.format(solusi), unsafe_allow_html=True)
        
        else:
            st.image("data/img/sad.png", width=100)
            st.write('<p style="color:red;">Topic: <b>{}</p>'.format(topic), unsafe_allow_html=True)
            st.write('<p style="color:red;">Sentiment: <b>Negative</b></p>', unsafe_allow_html=True)
            st.write('<div style="color:red; display:flex;" > <p >Solusi: </p> <b>{}</b> </div>'.format(solusi), unsafe_allow_html=True)
            # st.write("Probability:", probability) 
        
    
    text_input = st.text_input("Input Text",value='Jadi aq beli toner ini krn muka aku lagi bruntusan jadinya agak kasar dan merah, pas lihat Npure Toner lgsg tertarik krn ada kandungan centella asiaticanya. Dan bener aja pas nyoba tonernya enak bgt, ringan, cepat menyerap, bruntusan perlahan berkurang, wajahpun jd ga kemerahan, toopp. i had many litle freckles on mc face in 1 week ago, i think its normal bcs after that i got my normal skin again and i think for 1 month my face more bright')
    if st.button("Predict Sentiment"):
        if text_input:
            predict_sentiment(text_input)
        else:
            st.warning("Please enter text.")


def plot_confusion(ys_test,ys_pred):
    ######################### Confusion Matrix #########################
    conf_matrix = confusion_matrix(ys_test, ys_pred)
    f, ax = plt.subplots(figsize=(8,5))
    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_counts = ['{0:0.0f}'.format(value) for value in conf_matrix.flatten()]
    labels = [f'{v1}\n{v2}' for v1, v2 in
            zip(group_names,group_counts)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(conf_matrix, annot=labels, fmt='', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('data prediksi')
    plt.ylabel('data aktual')
    st.pyplot(f)
    classification_rep = classification_report(ys_test, ys_pred, target_names=['Negative', 'Positive'])
    
    st.text("Classification Report:\n" + classification_rep)
    TN, FP, FN, TP = conf_matrix.ravel()
    # Extracting TN, FP, FN, TP
    st.write(f'Pada plot confusion matrix diatas di dapatkan bahwa nilai true negatif sebesar {TN} ,false positif sebesar {FP} ,flase negatif sebesar {FN} , true positif sebesar {TP} dari nilai yang di dapatkan maka untuk menghitung nilai  akurasi ,precision,dan recall')

    with st.expander('Perhitungan manual'):

        # Calculating Accuracy
        accuracy = ((TP + TN) / (TP + TN + FP + FN))* 100

        # Calculating Precision
        precision_pos = (TP / (TP + FP))* 100
        precision_neg = (TN / (TN + FN))* 100
        precision=((precision_pos + precision_neg)/2)
        # Calculating Recall
        recall_pos = (TP / (TP + FN))* 100
        recall_neg = (TN / (TN + FP))* 100
        recall=((recall_pos + recall_neg)/2)

        st.subheader('Perhitungan nilai akurasi')
        st.write(f'Menghitung nilai akurasi dapat menggunakan rumus (true pos+true neg)/(true pos + false neg + false pos + true neg) * 100 maka sehingga ({TP}+{TN})/({TP}+{TN}+{FP}+{FN}) * 100 maka akurasi yang di hasilkan sebesar : {accuracy:.0f}%')
        
        st.subheader('Perhitungan Nilai Precision')
        st.write('Untuk menghitung nilai precision kita harus menghitung nilai precision setiap kelasnya terlebih dahulu sehingga hasilnya akan dijumlahkan dan dibagi banyak kelasnya rumus yang digunakan untuk menghitung nilai presision setiap kelasnya sebagai berikut:')
        st.write('-- Precision Negatif')
        st.write(f'Perhitungan nilai presision untuk kelas negatif menggunakan rumus (true neg)/(true neg + false neg) sehingga ({TN})/({TN}+{FN}) * 100 maka akurasi yang di hasilkan sebesar : {precision_neg:.0f}%')
        st.write('-- Precision Positif')
        st.write(f'Perhitungan nilai presision untuk kelas positif menggunakan rumus (true pos)/(true pos + false pos) sehingga ({TP})/({TP}+{FP}) * 100 maka akurasi yang di hasilkan sebesar : {precision_pos:.0f}%')
        st.write('-- Precision Keseluruhan')
        st.write(f'Menghitung nilai presision keseluruhan dapat menggunakan rumus (precision positif + precision negatif) / 2  maka sehingga ({precision_pos:.0f}+{precision_neg:.0f}) / 2  maka akurasi yang di hasilkan sebesar : {precision:.0f}%')
        
        st.subheader('Perhitungan Nilai Recall')
        st.write('Untuk menghitung nilai recall kita harus menghitung nilai recall setiap kelasnya terlebih dahulu sehingga hasilnya akan dijumlahkan dan dibagi banyak kelasnya rumus yang digunakan untuk menghitung nilai presision setiap kelasnya sebagai berikut:')
        st.write('-- Recall Negatif')
        st.write(f'perhitungan nilai presision untuk kelas negatif menggunakan rumus (true neg)/(true neg + false pos) sehingga ({TN})/({TN}+{FP}) * 100 maka akurasi yang di hasilkan sebesar : {recall_neg:.0f}%')
        st.write('-- Recall Positif')
        st.write(f'perhitungan nilai presision untuk kelas positif menggunakan rumus (true pos)/(true pos + false neg) sehingga ({TP})/({TP}+{FN}) * 100 maka akurasi yang di hasilkan sebesar : {recall_pos:.0f}%')
        st.write('-- Recall Keseluruhan')
        st.write(f'menghitung nilai presision keseluruhan dapat menggunakan rumus (recall positif + recall negatif) / 2  maka sehingga ({recall_pos:.0f}+{recall_neg:.0f}) / 2  maka akurasi yang di hasilkan sebesar : {recall:.0f}%')

def plot_kfold(model,xs_train, ys_train):
    cv_scores = cross_val_score(model, xs_train, ys_train, cv=5)
    colors = ['skyblue', 'lightgreen', 'lightcoral', 'lightsalmon', 'lightblue']

    plt.figure(figsize=(8, 5))
    for i in range(len(cv_scores)):
        plt.bar(i+1, cv_scores[i], color=colors[i % len(colors)])
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.title('Cross-Validation Scores')
    plt.ylim(0, 1)
    st.pyplot()

    cv_scores_df = pd.DataFrame({"K/Fold": range(1, len(cv_scores) + 1), "Accuracy": cv_scores})
    # Mengatur ulang indeks agar dimulai dari 1
    cv_scores_df.index = cv_scores_df.index + 1
    st.dataframe(cv_scores_df,use_container_width=True)
    Kbest=cv_scores_df.loc[(cv_scores_df['Accuracy']==cv_scores_df['Accuracy'].max())]
    Kbad=cv_scores_df.loc[(cv_scores_df['Accuracy']==cv_scores_df['Accuracy'].min())]
    mean=cv_scores.mean() * 100
    st.write("Rata-rata akurasi sebesar: {:.2f}%".format(mean))

    st.write(f"Dari gambar dan tabel diatas kita dapat melihat bahwa kenaikan dan penurunan pada pengujian di setiap nilai K/Fold nya dimana nilai akurasi tertinggi didapatkan pada K/Fold ke {Kbest['K/Fold'].values[0]} dengan nilai akurasi sebesar {round(Kbest['Accuracy'].values[0]*100,2)}% dan nilai akurasi terrendah didapatkan pada K/Fold ke {Kbad['K/Fold'].values[0]} dengan nilai akurasi sebesar {round(Kbad['Accuracy'].values[0]*100,2)}%  dengan rata-rata akurasi yang di dapatkan sebesar {mean:.2f}% dapat disimpulkan bahwa model baik dalam melakukan proses pengklasifikasian terhadap dataset yang digunakan")
    with st.expander('Penjelasan K-fold'):
        st.write('K-Fold Cross Validation merupakan teknik untuk memperkirakan kesalahan prediksi saat mengevaluasi kinerja model. Data dibagi menjadi himpunan bagian dengan k jumlah yang kira-kira sama. Model klasifikasi dilatih dan diuji sebanyak k kali.')
        st.image('data/img/kfold.png')

######################### Dashboard Report #########################
st.sidebar.markdown("### Report")
if not st.sidebar.checkbox("Hide", True, key='hide_checkbox_dashboard'):
    report_option = st.selectbox('Report option', ['Confusion Matrix', 'K-fold'], key='breakdown_type')
    
    if report_option == 'Confusion Matrix':
        st.subheader(f'Tampilan plot {report_option}')
        confusion_option=st.selectbox('dataset option',['sesudah_SMOTE','sebelum_SMOTE'],key='confusion')
        
        if confusion_option == 'sesudah_SMOTE':
            plot_confusion(yc_test,yc_pred)
        if confusion_option == 'sebelum_SMOTE':
            plot_confusion(y_test,y_pred)

    if report_option == 'K-fold':
        st.subheader(f'Tampilan plot {report_option}')
        ######################### Cross Validation #########################
        kfold_option=st.selectbox('dataset option',['sesudah_SMOTE','sebelum_SMOTE'],key='kfold')
        if kfold_option == 'sesudah_SMOTE':
            plot_kfold(svm,Xc_train,yc_train)
        if kfold_option == 'sebelum_SMOTE':
            plot_kfold(svm_classifier,X_train,y_train)

