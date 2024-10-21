import streamlit as st
import joblib
import gensim
import prepro as pp #text preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

######################### load model #########################
optimal_model = joblib.load('data/model/topic_model.joblib')
bigram_mod = joblib.load('data/model/bigram_model.joblib')
id2word = joblib.load('data/model/id2word_model.joblib')
svm_classifier=joblib.load('data/model/svm_sebelum_smote.joblib')
svm=joblib.load('data/model/svm_sesudah_smote.joblib')

@st.cache_data
def load_data():
    data = pd.read_csv('data/dataset/Hasil.csv', encoding='latin-1')
    return data
data = load_data()
data.drop(data.tail(3).index, inplace=True) 

tfidf_vectorizer = TfidfVectorizer(max_features=2000, max_df=0.75, min_df=5)  # Tambahkan max_df dan min_df
tfidf_vectorizer.fit_transform(data['content_preprocessing'])

######################### Predict Topic #########################
def label_topics(integer):
    if integer == 0:
        return 'Penggunaan Produk'
    elif integer == 1:
        return 'kualitas Produk'
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
def predik_sentiment(option,text):
    clean_text = pp.cleaningText(text)
    case_text = pp.casefoldingText(clean_text)
    token_text = pp.tokenizingText(case_text)
    norm_text = pp.normalisasiText(token_text)
    stop_text = pp.filteringText(norm_text)
    stem_text = pp.stemmingText(stop_text)
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

    if option=='SVM-SMOTE':
        prediction = svm.predict(text_tfidf)[0]
        prediction_topic = predict_topic(preprocessed_text)
        topic = label_topics(prediction_topic)
        solusi = label_solusi(prediction, prediction_topic)

        st.write(f'prediksi menggunakan metode {option}')
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

    if option=='SVM':
        prediction = svm_classifier.predict(text_tfidf)[0]
        prediction_topic = predict_topic(preprocessed_text)
        topic = label_topics(prediction_topic)
        solusi = label_solusi(prediction, prediction_topic)
        st.write(f'prediksi menggunakan metode {option}')
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

