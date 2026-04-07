import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Ensure necessary NLTK data is downloaded
def download_nltk_resources():
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

download_nltk_resources()

lemma = WordNetLemmatizer()
stop = set(stopwords.words('english'))

def clean_text(txt):
    """
    Cleans the input text by:
    1. Converting to lowercase
    2. Tokenizing
    3. Lemmatizing words and removing stopwords
    4. Removing non-alphabetic characters
    """
    txt = txt.lower()
    words = word_tokenize(txt)
    words = [lemma.lemmatize(word) for word in words if word not in stop]
    clean_txt = " ".join(words)
    clean_txt = re.sub('[^a-z]', ' ', clean_txt)
    # Remove extra whitespace
    clean_txt = ' '.join(clean_txt.split())
    return clean_txt
