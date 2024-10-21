import streamlit as st
import pandas as pd

def plot_confusion(options,TN, FP, FN, TP):
    ######################### Confusion Matrix #########################
    direc=str(options)
    st.image('data/eval/cf_'+direc+'.png')
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

def plot_kfold(fileloc):
    direct=str(fileloc)
    st.image('data/eval/kfold_'+direct+'.png')

    cv_scores_df = pd.read_csv('data/eval/kfold_'+direct+'.csv')
    cv_scores_df=pd.DataFrame(cv_scores_df)
    # Mengatur ulang indeks agar dimulai dari 1
    cv_scores_df.index = cv_scores_df.index + 1
    st.dataframe(cv_scores_df,use_container_width=True)
    Kbest=cv_scores_df.loc[(cv_scores_df['Accuracy']==cv_scores_df['Accuracy'].max())]
    Kbad=cv_scores_df.loc[(cv_scores_df['Accuracy']==cv_scores_df['Accuracy'].min())]
    mean=cv_scores_df['Accuracy'].mean() * 100
    st.write("Rata-rata akurasi sebesar: {:.2f}%".format(mean))

    st.write(f"Dari gambar dan tabel diatas kita dapat melihat bahwa kenaikan dan penurunan pada pengujian di setiap nilai K/Fold nya dimana nilai akurasi tertinggi didapatkan pada K/Fold ke {Kbest['K/Fold'].values[0]} dengan nilai akurasi sebesar {round(Kbest['Accuracy'].values[0]*100,2)}% dan nilai akurasi terrendah didapatkan pada K/Fold ke {Kbad['K/Fold'].values[0]} dengan nilai akurasi sebesar {round(Kbad['Accuracy'].values[0]*100,2)}%  dengan rata-rata akurasi yang di dapatkan sebesar {mean:.2f}% dapat disimpulkan bahwa model baik dalam melakukan proses pengklasifikasian terhadap dataset yang digunakan")
    with st.expander('Penjelasan K-fold'):
        st.write('K-Fold Cross Validation merupakan teknik untuk memperkirakan kesalahan prediksi saat mengevaluasi kinerja model. Data dibagi menjadi himpunan bagian dengan k jumlah yang kira-kira sama. Model klasifikasi dilatih dan diuji sebanyak k kali.')
        st.image('data/img/kfold.png')

def plot_confusion_topik(options,TN, FP, FN, TP):
    ######################### Confusion Matrix #########################
    direc=str(options)
    st.image('data/eval/cf_'+direc+'.png')
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

def plot_kfold_topik(fileloc):
    direct=str(fileloc)
    st.image('data/eval/kfold_'+direct+'.png')

    cv_scores_df = pd.read_csv('data/eval/kfold_'+direct+'.csv')
    cv_scores_df=pd.DataFrame(cv_scores_df)
    # Mengatur ulang indeks agar dimulai dari 1
    cv_scores_df.index = cv_scores_df.index + 1
    st.dataframe(cv_scores_df,use_container_width=True)
    Kbest=cv_scores_df.loc[(cv_scores_df['Accuracy']==cv_scores_df['Accuracy'].max())]
    Kbad=cv_scores_df.loc[(cv_scores_df['Accuracy']==cv_scores_df['Accuracy'].min())]
    mean=cv_scores_df['Accuracy'].mean() * 100
    st.write("Rata-rata akurasi sebesar: {:.2f}%".format(mean))

    st.write(f"Dari gambar dan tabel diatas kita dapat melihat bahwa kenaikan dan penurunan pada pengujian di setiap nilai K/Fold nya dimana nilai akurasi tertinggi didapatkan pada K/Fold ke {Kbest['K/Fold'].values[0]} dengan nilai akurasi sebesar {round(Kbest['Accuracy'].values[0]*100,2)}% dan nilai akurasi terrendah didapatkan pada K/Fold ke {Kbad['K/Fold'].values[0]} dengan nilai akurasi sebesar {round(Kbad['Accuracy'].values[0]*100,2)}%  dengan rata-rata akurasi yang di dapatkan sebesar {mean:.2f}% dapat disimpulkan bahwa model baik dalam melakukan proses pengklasifikasian terhadap dataset yang digunakan")
    with st.expander('Penjelasan K-fold'):
        st.write('K-Fold Cross Validation merupakan teknik untuk memperkirakan kesalahan prediksi saat mengevaluasi kinerja model. Data dibagi menjadi himpunan bagian dengan k jumlah yang kira-kira sama. Model klasifikasi dilatih dan diuji sebanyak k kali.')
        st.image('data/img/kfold.png')