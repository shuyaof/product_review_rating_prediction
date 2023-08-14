## YOUR SCORE.PY CODE HERE ##

import json, re, pickle
from azureml.core.model import Model

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.svm import LinearSVC

nltk.download("stopwords", quiet = True)
nltk.download("wordnet", quiet = True)
nltk.download("punkt", quiet = True)

def init():
    # load model and vectorizer and save them as global variables
    global final_model, tfidf_vectorizer

    model_dir = Model.get_model_path('final_linearsvc')
    vec_dir = Model.get_model_path('final_vectorizer')
    
    with open(vec_dir,'rb') as f:
        tfidf_vectorizer = pickle.load(f)

    with open(model_dir,'rb') as f:
        final_model = pickle.load(f)

def process_text(text):
    # Removing HTML tags
    clean_text = re.sub(r'<.*?>', '', text)
    
    # Removing punctuations and special characters
    clean_text = re.sub(r'[^\w\s]', '', clean_text)
    clean_text = clean_text.lower()
    
    # Tokenizing
    tokens = word_tokenize(clean_text)
    
    # Stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    
    # Lemmatizing
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in stemmed_tokens]
    
    # Removing stopwords
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in lemmatized_tokens if word not in stop_words]
    
    return ' '.join(filtered_words)

def run(input_data):
    input_json = json.loads(input_data)
    reviews = []
    for n in range(input_json['n_reviews']): 
        review = input_json['reviews'][n]['review_body']
        reviews.append(review)

    processed_reviews = [process_text(review) for review in reviews]
    matrix = tfidf_vectorizer.transform(processed_reviews)
    predictions = final_model.predict(matrix).tolist()
    response_content = {"predictions":predictions}
    return response_content

