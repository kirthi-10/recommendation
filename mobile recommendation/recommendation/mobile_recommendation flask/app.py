import numpy as np
import pickle
import pandas as pd
import os
from flask import Flask, request, render_template
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download the 'stopwords' resource
nltk.download('stopwords')

app = Flask(__name__)

model = pickle.load(open('src_model_dataframe.pkl', 'rb'))
vectorizer = pickle.load(open('src_model_similarity.pkl', 'rb'))

mobile_df = pd.read_csv('mobile_data.csv') 

vectorizer = CountVectorizer(stop_words='english')
count_matrix = vectorizer.fit_transform(mobile_df['name'])

ps = PorterStemmer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/input')
def input():
    return render_template('input.html')

@app.route('/submit', methods=["POST", "GET"])
def submit():  # Reading the inputs given by the user

    mobile_name = request.form['userInput']

    # Check if the mobile name exists in the DataFrame
    if mobile_name not in mobile_df['name'].values:
        return render_template("output.html", result="Mobile name not found in the database.")

    # Get the index of the mobile in the DataFrame
    idx = mobile_df[mobile_df['name'] == mobile_name].index[0]

    # Calculate cosine similarity between the given mobile and all others
    cosine_sim = cosine_similarity(count_matrix[idx], count_matrix)

    sim_scores = list(enumerate(cosine_sim[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:6]  # Get top 5 similar phones

    # Get the indices of the similar phones
    phone_indices = [i[0] for i in sim_scores]

    # Get the names of the similar phones
    recommendations = mobile_df['name'].iloc[phone_indices].tolist()

    return render_template("output.html", recommendations=recommendations)
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 1222))
    app.run(port=port, debug=True)

