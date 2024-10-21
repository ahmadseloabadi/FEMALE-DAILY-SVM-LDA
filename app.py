import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import  accuracy_score,recall_score,precision_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

import gensim
import nltk

import joblib

import prepro as pp #text preprocessing
import data_visual as dv #visualisasi data pada data visual 
import inset_lexicon as il #inset lexicon
import report as rt #bagian report
import topik_lda as lda #bagian lda
import predict as pred 

st.title("PENERAPAN METODE SUPPORT VECTOR MACHINE UNTUK ANALISIS SENTIMEN BERBASIS ASPEK PADA ULASAN PRODUK SKINCARE DENGAN PEMODELAN LATENT DIRICHLET ALLOCATION (LDA)")
st.sidebar.title("Analisis Sentimen Berbasis Aspek Pada Ulasan Produk Skincare")

st.set_option('deprecation.showPyplotGlobalUse', False)

# @st.cache(persist=True) deprecated
@st.cache_data
def load_data():
    data = pd.read_csv('data/dataset/Hasil.csv', encoding='latin-1')
    return data
data = load_data()

######################### load model #########################
svm_classifier=joblib.load('data/model/svm_sebelum_smote.joblib')
svm=joblib.load('data/model/svm_sesudah_smote.joblib')

dataset=load_data()
dataset.drop(dataset.tail(3).index, inplace=True) 

tfidf_vectorizer = TfidfVectorizer(max_features=2000, max_df=0.75, min_df=5)  # Tambahkan max_df dan min_df

Encoder = LabelEncoder()


def data_SMOTE(datax):
    datax['polarity']  = Encoder.fit_transform(datax['polarity'] )
    x_res =tfidf_vectorizer.fit_transform(datax['content_preprocessing'])
    y_res =datax['polarity']
    smote = SMOTE(sampling_strategy='auto')
    X_smote, Y_smote = smote.fit_resample(x_res, y_res)
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_smote, Y_smote, test_size=0.20)
    return Xc_train, Xc_test, yc_train, yc_test


def dataset_preparation(datase):
    datase['polarity']  = Encoder.fit_transform(datase['polarity'] )    
    X_tfidfi = tfidf_vectorizer.fit_transform(datase['content_preprocessing'])
    X_train, X_test, y_train, y_test = train_test_split(X_tfidfi, datase['polarity'], test_size=0.2)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test=dataset_preparation(dataset)
Xc_train, Xc_test, yc_train, yc_test=data_SMOTE(dataset)

# yc_pred=svm.predict(Xc_test)
# y_pred = svm_classifier.predict(X_test)


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

######################### data visual #########################
st.sidebar.markdown("### Data Visual ")
datas=pd.DataFrame(data)
datas.drop(datas.tail(3).index, inplace=True)

dfsmote = pd.read_csv('data/smote/data_smote_full.csv')
dfsmote=pd.DataFrame(dfsmote)
if not st.sidebar.checkbox("Hide", False, key='hide_data_visual'):
    data_option = st.selectbox('Data Option', ['dataset', 'preprocessing','TF-IDF','SMOTE','pembagian data'], key='data_option')

    if data_option =='dataset':
        st.subheader('Dataset')
        dataset_options=st.selectbox('pilih dataset',['sebelum_smote','sesudah_smote'],key='dataset data')
        if dataset_options=='sebelum_smote':
            st.write(f'Tampilan dataset sebelum smote dengan total jumlah dataset sebanyak {len(datas)}')
            def filter_sentiment(dataset, selected_sentiment):
                return dataset[dataset['sentiment'].isin(selected_sentiment)]

            sentiment_map = {'positive': 'positive', 'negative': 'negative'}
            selected_sentiment = st.multiselect('Pilih kelas sentimen', list(sentiment_map.keys()), default=list(sentiment_map.keys()))
            filtered_data = filter_sentiment(datas, selected_sentiment)
            
            st.dataframe(filtered_data[['content','sentiment']],use_container_width=True)
            dv.plot_sentiment_label(dataset_options,datas)
        if dataset_options=='sesudah_smote':
            dfsmote['sentimen'] = dfsmote['sentimen'].map({1: 'positive', 0: 'negative'})
            st.write(f'Tampilan dataset sesudah smote dengan total jumlah dataset sebanyak {len(dfsmote)}')
            def filter_sentiment(dataset, selected_sentiment):
                return dataset[dataset['sentimen'].isin(selected_sentiment)]

            sentiment_map = {'positive': 'positive', 'negative': 'negative'}
            selected_sentiment = st.multiselect('Pilih kelas sentimen', list(sentiment_map.keys()), default=list(sentiment_map.keys()))
            filtered_data = filter_sentiment(dfsmote, selected_sentiment)
            
            st.dataframe(filtered_data[['kalimat_asli','sentimen']],use_container_width=True)
        
            dv.plot_sentiment_label(dataset_options,dfsmote)
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
        
        dv.tf_idf(datas)
    if data_option =='pembagian data':
        st.subheader('Pembagian Data')
        split_options=st.selectbox('pilih dataset',['sebelum_smote','sesudah_smote'],key='split data')
        if split_options=='sesudah_smote':
            dv.split_Data(split_options,dfsmote)
        if split_options=='sebelum_smote':
            dv.split_Data(split_options,datas)
    
    if data_option =='SMOTE':
        st.subheader('pernerapan SMOTE')
        dv.smote(datas)
######################### end data visual #########################

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
        clean_text = pp.cleaningText(kalimat)
        case_text = pp.casefoldingText(clean_text)
        token_text = pp.tokenizingText(case_text)
        norm_text = pp.normalisasiText(token_text)
        stop_text = pp.filteringText(norm_text)
        stem_text = pp.stemmingText(stop_text)
        il.pelabelan(stem_text)

    with st.expander('kamus lexicon'):
        kamus_option = st.multiselect('pilih kamus inset lexicon', ['positive', 'negative'],['positive', 'negative'], key='kamus')
        for file in kamus_option:
            st.write(f"tampilan data kamus {file}:")
            il.import_and_display_tsv(file)
######################### end inset lexicon #########################

######################### topic modeling #########################
st.sidebar.subheader("Topic modeling with LDA")

if not st.sidebar.checkbox("Hide", True, key='close_checkbox_2'):
    lda_option=st.selectbox('dataset option',['sesudah_smote','sebelum_smote'],key='lda option')
    if lda_option == 'sesudah_smote':
        lda.plot_data_topic(lda_option)
    if lda_option == 'sebelum_smote':
        lda.plot_data_topic(lda_option)
    
######################### end topic modeling #########################

######################### Predict #########################
st.sidebar.header("Predict Sentiment")
if not st.sidebar.checkbox("Hide", True, key='close_checkbox_4'):
    with st.expander ("...."):
        uji_c=pengujian_nilai_c()
        st.dataframe(uji_c,use_container_width=True)
        st.write('penjelasan nilai C')
        st.write('parameter C (juga dikenal sebagai parameter penalti) adalah faktor penting yang mempengaruhi kinerja dan perilaku model SVM. Parameter C mengendalikan antara penalti kesalahan klasifikasi dan lebar margin. Nilai C yang lebih kecil akan menghasilkan margin yang lebih lebar, dan menjadikan model tidak peka terhadap data dan kelasahan klasifikasi(UNDERFITTING). Sebaliknya, nilai C yang lebih besar akan menghasilkan margin yang lebih sempit, menjadikan model lebih peka terhadap data dan tingkat kesalahan klasifikasi.(OVERFITTING)')
        
    metode_option = st.selectbox('Metode option', ['SVM-SMOTE', 'SVM'], key='metod')
    text_input = st.text_input("Input Text",value='Jadi aq beli toner ini krn muka aku lagi bruntusan jadinya agak kasar dan merah, pas lihat Npure Toner lgsg tertarik krn ada kandungan centella asiaticanya. Dan bener aja pas nyoba tonernya enak bgt, ringan, cepat menyerap, bruntusan perlahan berkurang, wajahpun jd ga kemerahan, toopp. i had many litle freckles on mc face in 1 week ago, i think its normal bcs after that i got my normal skin again and i think for 1 month my face more bright')
    if metode_option=='SVM-SMOTE':
        if st.button("Predict Sentiment"):
            if text_input:
                pred.predik_sentiment(metode_option,text_input)
            else:
                st.warning("Please enter text.")
    if metode_option=='SVM':
        if st.button("Predict Sentiment"):
            if text_input:
                pred.predik_sentiment(metode_option,text_input)
            else:
                st.warning("Please enter text.")
            
######################### end Predict #########################

######################### Dashboard Report #########################
st.sidebar.markdown("### Report")
if not st.sidebar.checkbox("Hide", True, key='hide_checkbox_dashboard'):
    report_option = st.selectbox('Report option', ['Confusion Matrix', 'K-fold'], key='breakdown_type')
    
    if report_option == 'Confusion Matrix':
        st.subheader(f'Tampilan plot {report_option}')
        confusion_option=st.selectbox('dataset option',['sesudah_smote','sebelum_smote'],key='confusion')

        if confusion_option == 'sesudah_smote':
            rt.plot_confusion(confusion_option,119,19,11,95)
        if confusion_option == 'sebelum_smote':
            rt.plot_confusion(confusion_option,124,10,26,40)

    if report_option == 'K-fold':
        st.subheader(f'Tampilan plot {report_option}')
        ######################### Cross Validation #########################
        kfold_option=st.selectbox('dataset option',['sesudah_smote','sebelum_smote'],key='kfold')
        if kfold_option == 'sesudah_smote':
            rt.plot_kfold(kfold_option)
        if kfold_option == 'sebelum_smote':
            rt.plot_kfold(kfold_option)

######################### Dashboard Report #########################

