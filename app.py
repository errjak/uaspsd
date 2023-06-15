import streamlit as st
import pandas as pd
import numpy as np

from sklearn.utils.validation import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import make_scorer, accuracy_score, precision_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import model_selection
import pickle
from pickle import dump
from collections import OrderedDict

st.sidebar.title("Selamat Datang!")
st.sidebar.write(
    "Di Website Prediksi Saham Perusahaan Bukalapak.")
page1, page2, page3, page4 = st.tabs(
    ["Data", "Preprocessing", "Modelling", "Prediksi"])

with page1:
    st.title("Dataset Saham Bukalapak")
    st.write(
        "Dataset Yang digunakan adalah **Saham Bukalapak** dari [Yahoo Finance](https://finance.yahoo.com/quote/BUKA.JK/history?p=BUKA.JK)")
    st.write("Deskripsi Data")
    st.write("""
    Dataset yang digunakan adalah dataset tentang Saham Bukalapak untuk memprediksi secara diagnostik apakah saham perusahaan Bukalapak akan naik atau turun,
    berdasarkan pengukuran diagnostik tertentu yang disertakan dalam kumpulan data. Data yang digunakan adalah data harian dari tanggal 16 Juni 2022 sampai 14 Juni 2023.
    Yang memiliki kolom close dengan nilai yang berubah-ubah setiap harinya.
    Dalam dataset ini terdapat 7 fitur yaitu : 'Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', dan 'Volume'.
    Dimana fitur 'Close' yang akan diprediksi.
    Karena 'Close' merupakan harga penutupan saham Bukalapak setiap harinya.
    """)
    st.write("Berikut adalah 5 data teratas dari dataset yang digunakan")
    df = pd.read_csv(
        'https://raw.githubusercontent.com/errjak/dataset/main/BUKA.JK.csv')
    st.dataframe(df.head())
