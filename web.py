import streamlit as st
import pickle
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import re

# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')

ps = PorterStemmer()
lm = WordNetLemmatizer()

st.title('Hate Speech Detection')

st.write('This is a simple web app to predict hate speech in text')

text = st.text_area('Text')
# text = pd.DataFrame([text])

def preprocess(text):

    #remove non alphabetic characters
    text = re.sub('[^A-Za-z]', ' ', text)

    #lowercase
    text = text.lower()

    #tokenization
    words = nltk.word_tokenize(text)

    #punctuation mark removal
    words = [word for word in words if word.isalnum()]

    #stopwords removal
    words_stop = []
    for word in words:
        if word not in stopwords.words('english'):
            words_stop.append(word)

    #stemming
    words_stem = []
    for word in words_stop:
        words_stem.append(ps.stem(word))

    #lemmatization
    words_lemmatized = []
    for word in words_stem:
        words_lemmatized.append(lm.lemmatize(word))

    #join words
    text = ' '.join(words_lemmatized)

    return text

text = preprocess(text)

tfidf = pickle.load(open('./models/tf_idf.pkl', 'rb'))
text = tfidf.transform([text])


# use drop down to select model
model = st.selectbox('Model', ['SVM', 'KNN', 'Naive Bayes', 'Random Forest', 'Cost Sensitive SVM'])

# use button to trigger prediction

if st.button('Predict'):

    # load model
    if model == 'SVM':
        loaded_model = pickle.load(open('./models/svm_classifier.pkl', 'rb'))
    elif model == 'KNN':
        loaded_model = pickle.load(open('./models/knn_classifier.pkl', 'rb'))
    elif model == 'Naive Bayes':
        loaded_model = pickle.load(open('./models/nb_classifier.pkl', 'rb'))
    elif model == 'Random Forest':
        loaded_model = pickle.load(open('./models/rf_classifier.pkl', 'rb'))
    elif model == 'Cost Sensitive SVM':
        loaded_model = pickle.load(open('./models/cost_svm_classifier.pkl', 'rb'))

    # predict
    
    prediction = loaded_model.predict(text)

    # display
    if prediction == 0:
        st.metric('Result','Hate Speech')
    elif prediction == 1:
        st.metric('Result','Offensive Language')
    else:
        st.metric('Result','Neutral')
