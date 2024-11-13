import streamlit as st
import pandas as pd
import re
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn import preprocessing
import pickle
from selenium import webdriver
from selenium.webdriver.edge.options import Options
from selenium.webdriver.edge.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.microsoft import EdgeChromiumDriverManager

# Inisialisasi library NLTK
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Fungsi untuk inisialisasi WebDriver
def web_driver():
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    service = Service(EdgeChromiumDriverManager().install())
    driver = webdriver.Edge(service=service, options=options)
    return driver

# Fungsi untuk scraping konten artikel
def get_element_text(driver, xpath):
    try:
        return WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, xpath))
        ).text.strip()
    except Exception as e:
        st.write(f"Error finding element with XPath {xpath}: {e}")
        return ""

def extract_article_content(driver, article_url):
    driver.get(article_url)
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, '//h1'))
    )
    title = get_element_text(driver, './/h1')
    date = get_element_text(driver, './/p[@class="pt-20 date"]')
    content_elements = driver.find_elements(By.XPATH, './/div[@class="news-text"]/p')
    content = " ".join(p.text for p in content_elements)
    kategori = get_element_text(driver, './/div[@class="breadcrumb-content"]/p')

    return {
        "Title": title,
        "Date": date,
        "Content": content,
        "Category": kategori
    }

# Fungsi untuk preprocessing
def clean_lower(text):
    if isinstance(text, str):
        return text.lower()
    return text

def clean_punct(text):
    if isinstance(text, str):
        clean_patterns = re.compile(r'[0-9]|[/(){}\[\]\|@,;_]|[^a-z ]')
        text = clean_patterns.sub(' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return text

def _normalize_whitespace(text):
    if isinstance(text, str):
        corrected = re.sub(r'\s+', ' ', text)
        return corrected.strip()
    return text

def clean_stopwords(text):
    if isinstance(text, str):
        stopword = set(stopwords.words('indonesian'))
        text = ' '.join(word for word in text.split() if word not in stopword)
        return text.strip()
    return text

def sastrawistemmer(text):
    factory = StemmerFactory()
    st = factory.create_stemmer()
    text = ' '.join(st.stem(word) for word in tqdm(text.split()) if word in text)
    return text

# Fungsi utama untuk Streamlit
def main():
    st.title("News Article Classification")

    # Meminta input URL artikel dari pengguna
    url_input = st.text_input("Masukkan URL artikel:")

    if st.button("Scrape and Predict"):
        driver = web_driver()
        article_data = extract_article_content(driver, url_input)
        driver.quit()
        
        # Tampilkan hasil scraping
        st.write("**Title:**", article_data["Title"])
        st.write("**Date:**", article_data["Date"])
        st.write("**Content:**", article_data["Content"])
        st.write("**Category (Scraped):**", article_data["Category"])
        
        # Preprocessing data
        df = pd.DataFrame([article_data])
        df['lower case'] = df['Content'].apply(clean_lower)
        df['tanda baca'] = df['lower case'].apply(clean_punct)
        df['spasi'] = df['tanda baca'].apply(_normalize_whitespace)
        df['stopwords'] = df['spasi'].apply(clean_stopwords)
        df['stemming'] = df['stopwords'].apply(sastrawistemmer)

        # Load TF-IDF dan terapkan pada data baru
        filename_tfidf = 'tfidf_vectorizer.sav'
        tfidf_vectorizer = pickle.load(open(filename_tfidf, 'rb'))
        corpus = df['stemming'].tolist()
        x_tfidf = tfidf_vectorizer.transform(corpus)

        # Load model SVD (LSA)
        filename_svd = 'svd_model.sav'
        svd = pickle.load(open(filename_svd, 'rb'))
        x_new_lsa = svd.transform(x_tfidf)
        train_lsa_df = pd.DataFrame(x_new_lsa, columns=[f'Component_{i+1}' for i in range(x_new_lsa.shape[1])])
        train_lsa_df['Category'] = df["Category"].values

        # Label encoding kategori
        label_encoder = preprocessing.LabelEncoder()
        train_lsa_df['Category'] = label_encoder.fit_transform(train_lsa_df['Category'])

        # Load model Logistic Regression
        filename = 'lr_model.sav'
        lr_model = pickle.load(open(filename, 'rb'))

        # Prediksi kategori
        x_test = train_lsa_df.drop(['Category'], axis=1)
        y_pred = lr_model.predict(x_test)
        
        # Ubah hasil prediksi ke label kategori
        predicted_category = "Ekonomi" if y_pred[0] == 0 else "Olahraga"
        st.write("**Predicted Category:**", predicted_category)

if __name__ == "__main__":
    main()
