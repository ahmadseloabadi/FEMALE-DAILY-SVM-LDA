import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def get_Data(data_name):
    direc=str(data_name)
    if data_name=='sebelum_smote':
        data_topik=pd.read_csv('data/topik/hasil_TOPIC & SENTIMEN '+direc+'.csv')
        data_topik.drop(data_topik.tail(3).index, inplace=True) 
    else:
        data_topik=pd.read_csv('data/topik/hasil_TOPIC & SENTIMEN '+direc+'.csv')
    return data_topik


def plot_sentiment(nama_Data,topic_label):
    datas=get_Data(nama_Data)
    df = datas[datas['topic_label']==topic_label]
    count = df['sentimen'].value_counts()
    count = pd.DataFrame({'Sentimen':count.index, 'Reviews':count.values.flatten()})
    return count


def data_topic(nama_Data):
    data_topik=get_Data(nama_Data)
    riview_topic_data=data_topik[['Text','Keywords','topic_label','sentimen']]
    review_sentiment_count = data_topik.groupby('topic_label')['sentimen'].count().sort_values(ascending=False)
    review_sentiment_count = pd.DataFrame({'Topic Label':review_sentiment_count.index, 'Reviews':review_sentiment_count.values.flatten()})
    return review_sentiment_count ,riview_topic_data

def plot_data_topic(nama_Data):
    review_sentiment_count ,riview_topic_data=data_topic(nama_Data)
    st.subheader('Tampilan data hasil topik modeling')
    def filter_topik(dataset, selected_topik):
        return dataset[dataset['topic_label'].isin(selected_topik)]
    topik_map = {'Penggunaan Produk': 'Penggunaan Produk', 'Kualitas Produk': 'Kualitas Produk', 'Pembelian Produk': 'Pembelian Produk'}
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
    choice = st.multiselect('Pilih Aspek', ('Penggunaan Produk', 'Kualitas Produk', 'Pembelian Produk'), key='pick_topic_label_1')
    if len(choice) > 0:
        
        breakdown_type = st.selectbox('Pilih Tipe Visualisasi', ['Pie chart', 'Bar plot', ], key='breakdown_type')
        fig_3 = make_subplots(rows=1, cols=len(choice), subplot_titles=choice)
        if breakdown_type == 'Bar plot':
            for i in range(1):
                for j in range(len(choice)):
                    fig_3.add_trace(
                        go.Bar(x=plot_sentiment(nama_Data,choice[j]).Sentimen, y=plot_sentiment(nama_Data,choice[j]).Reviews, showlegend=False),
                        row=i+1, col=j+1
                    )
            fig_3.update_layout(height=600, width=800)
            st.plotly_chart(fig_3)
        else:
            fig_3 = make_subplots(rows=1, cols=len(choice), specs=[[{'type':'domain'}]*len(choice)], subplot_titles=choice)
            for i in range(1):
                for j in range(len(choice)):
                    fig_3.add_trace(
                        go.Pie(labels=plot_sentiment(nama_Data,choice[j]).Sentimen, values=plot_sentiment(nama_Data,choice[j]).Reviews, showlegend=True),
                        i+1, j+1
                    )
            fig_3.update_layout(height=600, width=800)
            st.plotly_chart(fig_3)
