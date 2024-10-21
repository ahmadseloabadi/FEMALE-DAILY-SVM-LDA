import re
import string
from nltk.tokenize import word_tokenize
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

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