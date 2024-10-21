import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer ,CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

@st.cache_data
def load_data():
    data = pd.read_csv('data/dataset/Hasil.csv', encoding='latin-1')
    return data
Encoder = LabelEncoder()

tfidf_vectorizer = TfidfVectorizer(max_features=2000, max_df=0.75, min_df=5)  # Tambahkan max_df dan min_df

dataset=load_data()
dataset.drop(dataset.tail(3).index, inplace=True) 

ulasan=dataset[['content']]
# Transformasi content_preprocessing menggunakan TF-IDF
tfidf = tfidf_vectorizer.fit_transform(dataset['content_preprocessing'])

# Mendapatkan daftar kata yang digunakan dalam TF-IDF
feature_names = tfidf_vectorizer.get_feature_names_out()

# Membuat DataFrame kosong untuk menyimpan nilai TF-IDF
tfidf_df = pd.DataFrame(columns=['TF-IDF'])

# Mengisi DataFrame dengan nilai TF-IDF yang tidak nol
for i, doc in enumerate(dataset['content_preprocessing']):
    doc_tfidf = tfidf[i]
    non_zero_indices = doc_tfidf.nonzero()[1]
    tfidf_values = doc_tfidf[0, non_zero_indices].toarray()[0]
    tfidf_dict = {feature_names[idx]: tfidf_values[j] for j, idx in enumerate(non_zero_indices)}
    tfidf_df.loc[i] = [' '.join(f'({feature_name}, {tfidf_dict[feature_name]:.3f})' for feature_name in tfidf_dict)]

# Menggabungkan DataFrame hasil dengan DataFrame utama
ulasan = pd.concat([ulasan, tfidf_df], axis=1)

x=dataset['content_preprocessing']
y=dataset['sentiment']
Xs_train, Xs_test, ys_train, ys_test = train_test_split(x, y, test_size=0.2,random_state=42)

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
@st.cache_data(experimental_allow_widgets=True)

def plot_sentiment_label(option,datasss):
    select = st.selectbox('Pilih Tipe Visualisasi', ['Bar plot', 'Pie chart'], key='visualization_type')
    if option =='sebelum_smote':
        sentiment_count = datasss['polarity'].value_counts()
    if option =='sesudah_smote':
        sentiment_count = datasss['sentimen'].value_counts()

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
    tfidf = tfidf_vectorizer.fit_transform(dataset['content_preprocessing'])

    # Mendapatkan daftar kata yang digunakan dalam TF-IDF
    feature_names = tfidf_vectorizer.get_feature_names_out()

    # Membuat DataFrame kosong untuk menyimpan nilai TF-IDF
    tfidf_df = pd.DataFrame(columns=['TF-IDF'])

    # Mengisi DataFrame dengan nilai TF-IDF yang tidak nol
    for i, doc in enumerate(dataset['content_preprocessing']):
        doc_tfidf = tfidf[i]
        non_zero_indices = doc_tfidf.nonzero()[1]
        tfidf_values = doc_tfidf[0, non_zero_indices].toarray()[0]
        tfidf_dict = {feature_names[idx]: tfidf_values[j] for j, idx in enumerate(non_zero_indices)}
        tfidf_df.loc[i] = [' '.join(f'({feature_name}, {tfidf_dict[feature_name]:.3f})' for feature_name in tfidf_dict)]

    # Menggabungkan DataFrame hasil dengan DataFrame utama
    dataset = pd.concat([dataset, tfidf_df], axis=1)
    st.write('Tampilan Pembobotan TF-IDF Pada Dataset')
    st.dataframe(dataset[['content','content_preprocessing','TF-IDF','sentiment']])

def split_Data(options,datas):
    if options == 'sebelum_smote':
        
        

        st.write(f'Dari keseluruan dataset yang berjumlah {len(datas)} dilakukan pembagian data train dan data test dengan perbandingan 80:20 sehingga dapat dilihat hasil pemgaian sebagai berikut')
        
        label_train=data_train['sentiment'].value_counts()
        
        st.write(f'Tampilan data training dengan jumlah data sebanyak {len(Xs_train)} dengan jumlah kelas positive sebanyak {label_train[1]} dan kelas negative {label_train[0]}')
        st.dataframe(train_df[['content','content_preprocessing','TF-IDF','sentiment']],use_container_width=True)


        label_test=data_test['sentiment'].value_counts()
        
        st.write(f'Tampilan data test dengan jumlah data sebanyak {len(Xs_test)} dengan jumlah kelas positive sebanyak {label_test[1]} dan kelas negative {label_test[0]}')
        st.dataframe(test_df[['content','content_preprocessing','TF-IDF','sentiment']],use_container_width=True)
    
    if options == 'sesudah_smote':
        smote_data=pd.read_csv('data/smote/data_smote.csv',sep=',')
        x=smote_data['kalimat_asli']
        y=smote_data['sentimen']
        data_train_smote = pd.DataFrame({'content_preprocessing': x,'TF-IDF':smote_data['TF-IDF'], 'sentiment': y})
        

        st.write(f'Dari keseluruan dataset yang berjumlah {len(smote_data)+len(Xs_test)} dilakukan pembagian data train dan data test dengan perbandingan 80:20 sehingga dapat dilihat hasil pemgaian sebagai berikut')
        
        label_train=data_train_smote['sentiment'].value_counts()
        
        st.write(f'Tampilan data training dengan jumlah data sebanyak {len(smote_data)} dengan jumlah kelas positive sebanyak {label_train[1]} dan kelas negative {label_train[0]}')
        st.dataframe(data_train_smote[['content_preprocessing','TF-IDF','sentiment']],use_container_width=True)
        label_test=data_test['sentiment'].value_counts()
        st.write(f'Tampilan data test dengan jumlah data sebanyak {len(Xs_test)} dengan jumlah kelas positive sebanyak {label_test[1]} dan kelas negative {label_test[0]}')
        st.dataframe(test_df[['content_preprocessing','TF-IDF','sentiment']],use_container_width=True)

@st.cache_data(experimental_allow_widgets=True)
def smote(df):
    vectorizer = TfidfVectorizer(max_features=2000, max_df=0.75, min_df=5)  # Tambahkan max_df dan min_df
    df['polatiry'] = Encoder.fit_transform(df['polarity'] )

    x_res =vectorizer.fit_transform(df['content_preprocessing'])
    y_res =df['polatiry']
    X_train, X_test, y_train, y_test = train_test_split(x_res, y_res, test_size=0.2,random_state=42)

    #penerapan smote
    smote = SMOTE(sampling_strategy='auto')
    X_smote, Y_smote = smote.fit_resample(X_train, y_train)
    st.title('SMOTE')
    st.text('SMOTE adalah teknik untuk mengatasi ketidak seimbangan kelas pada dataset')
    
    seb_smote,ses_smote = st.columns(2)
    with seb_smote:
        st.header('sebelum SMOTE')
        st.write('Jumlah dataset:',len(y_train))
        # Hitung jumlah kelas sebelum SMOTE
        st.write("Jumlah kelas:  ")
        Jum_sentimen = y_train.value_counts()
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
        plt.pie(dataset.groupby('sentiment')['sentiment'].count(), autopct=" %.1f%% " ,labels=labels)
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
        st.write(f'menampilkan data sesudah {optiondata} dengan jumlah {len(dsmote)}')
        sentiment_map = {1: 'positif',0:'negatif'}
        sentiment = st.multiselect('Pilih kelas sentimen', list(sentiment_map.values()),default=list(sentiment_map.values()))
        sentiment = [key for key, value in sentiment_map.items() if value in sentiment]
        filtered_data = sentimen_smote(dsmote, sentiment)
        st.dataframe(filtered_data)
    if(optiondata == 'SINTETIS') :
        st.write(f'menampilkan data sesudah {optiondata} dengan jumlah {len(sintetis)}')
        st.dataframe(sintetis)
    with st.expander('---') :
        st.text('cari kalimat ')
        search_query = st.text_input("Masukkan kalimat yang ingin dicari:")
        search_results = dsmote[dsmote['kalimat_asli'].str.contains(search_query)]
        if st.button('cari') :
            st.dataframe(search_results,use_container_width=True)
        # Mencari baris duplikat berdasarkan nilai kalimat asli
        duplicates = dsmote[dsmote.duplicated(subset='kalimat_asli', keep=False)]
        results = {"index": [], "TF-IDF": [], "kalimat_asli": [], "sentimen": []}
        # Menampilkan kalimat yang duplikat
        for index, row in duplicates.iterrows():
            results["index"].append(index)
            results["TF-IDF"].append(row['TF-IDF'])
            results["sentimen"].append(row['sentimen'])
            results["kalimat_asli"].append(row['kalimat_asli'])

        duplikat = pd.DataFrame(results)
        st.dataframe(duplikat)