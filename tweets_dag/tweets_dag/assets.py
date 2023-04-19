from dagster import asset
from nltk import word_tokenize, FreqDist
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords
import pandas as pd
import os
from zipfile import ZipFile


@asset(
    description="Download tweets from github repo and extract nouns"
)
def download_russian_troll_tweets():
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download("stopwords")
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    files = os.listdir('russian-troll-tweets')
    for file in files:
        if file.endswith('.csv'):
            df = pd.read_csv('russian-troll-tweets/' + file)
            en = df[df['language'] == 'English']
            try:
                phrase = "\n".join(list(en['content']))
            except:
                pass
    words = word_tokenize(phrase)
    lem_words = [lemmatizer.lemmatize(word) for word in words]
    good_words = [word for word in lem_words if word.casefold() not in stop_words]
    tags = nltk.pos_tag(good_words)
    noun_words = [t[0] for t in tags if t[1] in ('NN','NE') and len(t[0]) > 2]
    distribution = FreqDist(noun_words)
    return distribution


@asset(
    description="Extract politifact tweets from zip file downloaded from kaggle and extract nouns"
)
def extract_political_fact_tweets():
    with ZipFile('political_facts.zip', 'r') as zip_ref:
        zip_ref.extractall('.')
    df = pd.read_json('politifact_factcheck_data.json', lines=True)
    fact_tweets = df['statement']
    fact_tweets = [str(tweet) for tweet in fact_tweets]
    fact_words = word_tokenize(" ".join(fact_tweets))
    lemmatizer = WordNetLemmatizer()
    lem_fact_words = [lemmatizer.lemmatize(word) for word in fact_words]
    stop_words = set(stopwords.words("english"))
    good_fact_words = [word for word in lem_fact_words if word.casefold() not in stop_words]
    fact_tags = nltk.pos_tag(good_fact_words)
    fact_noun_words = [t[0] for t in fact_tags if t[1] in ('NN','NE') and len(t[0]) > 2]
    fact_distribution = FreqDist(fact_noun_words)
    return fact_distribution


@asset(
    description="Merge the distributions of troll tweets and fact tweets"
)
def merge_distributions(download_russian_troll_tweets, extract_political_fact_tweets):
    troll_tweets_distribution = download_russian_troll_tweets()
    fact_tweets_distribution = extract_political_fact_tweets()
    all_words = list(troll_tweets_distribution.keys()) + list(fact_tweets_distribution.keys())
    all_distribution = FreqDist(all_words)
    return all_distribution


@asset(
    description="Process the tweets(merge, extract, download) and return a dataframe of the top 20 topics with their verdicts"
)
def process_tweets(merge_distributions, download_russian_troll_tweets, extract_political_fact_tweets):
    troll_tweets_distribution = download_russian_troll_tweets()
    fact_tweets_distribution = extract_political_fact_tweets
    all_distribution = merge_distributions(troll_tweets_distribution, fact_tweets_distribution)
    return all_distribution

@asset(
    description="Plot a heatmap of the distributions of the top 20 topics with their verdicts"
)
def plot_heatmap(process_tweets):
    import seaborn as sns
    import matplotlib.pyplot as plt
    verdict_topic_df = process_tweets()
    plt.figure(figsize=(20,20))
    sns.heatmap(verdict_topic_df.corr(), annot=True, fmt=".2f")
    plt.show()

