#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model
model = joblib.load('rf.pkl')

# Define custom stopwords
custom_stopwords = [
    'according', 'administration', 'also', 'america', 'american', 'americans', 'another', 'back', 'bill', 'black',
    'called', 'campaign', 'clinton', 'could', 'country', 'day', 'department', 'donald', 'election', 'even', 'every',
    'fact', 'first', 'former', 'fox', 'get', 'go', 'going', 'good', 'government', 'group', 'hillary', 'house',
    'Image', 'it', 'know', 'last', 'law', 'like', 'made', 'make', 'may', 'media', 'much', 'national', 'never', 'new',
    'news', 'man', 'many', 'obama', 'office', 'one', 'party', 'people', 'police', 'political', 'president',
    'presidential', 'public', 'really', 'republican', 'republicans', 'right', 'said', 'say', 'says', 'see', 'show',
    'since', 'state', 'states', 'still', 'support', 'take', 'think', 'time', 'told', 'trump', 'two', 'united', 'us',
    'via', 'video', 'vote', 'white', 'women', 'world', 'would', 'year', 'years', 'want'
]

# Initialize NLTK components
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words('english')

# Function to preprocess text
def preprocess_text(text):
    text = re.sub(r'[\x1d\x1c\x1b]', '', text)
    text = re.sub(r'\x19', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub('http\S+\s*', ' ', text)
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub('\s+', ' ', text)
    text = word_tokenize(text.lower())
    text = [word for word in text if word.lower() not in stop_words]
    text = [lemmatizer.lemmatize(word) for word in text]
    text = ' '.join(text)
    return text

# Function to predict the class
def predict_class(text):
    preprocessed_text = preprocess_text(text)
    tfidf_text = vectorizer.transform([preprocessed_text])
    prediction = model.predict(tfidf_text)[0]
    return prediction

# Load the TfidfVectorizer
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Streamlit app
def main():
    st.title('News Classifier')
    st.write('Enter a news text to classify it as real or fake:')
    text_input = st.text_area('Input Text', height=200)
    if st.button('Classify'):
        if text_input:
            prediction = predict_class(text_input)
            if prediction == 1:
                st.write('The news is classified as **real**.')
            else:
                st.write('The news is classified as **fake**.')
        else:
            st.write('Please enter a news text.')

if __name__ == '__main__':
    main()


# In[ ]:




