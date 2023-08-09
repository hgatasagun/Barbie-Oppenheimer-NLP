# Import libraries
###############################
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
from textblob import TextBlob
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import random
from gensim import corpora
from gensim.models.ldamodel import LdaModel

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


# Scraping user reviews from an IMDb movie page
###############################################
def scrape_reviews(url):
    driver = webdriver.Safari()
    driver.get(url)

    while True:
        try:
            load_more_button = driver.find_element(By.CLASS_NAME, "ipl-load-more__button")
            load_more_button.click()
            time.sleep(2)
        except:
            break

    review_elements = driver.find_elements(By.CLASS_NAME, "text.show-more__control")
    reviews = [review.text for review in review_elements]

    try:
        next_button = driver.find_element(By.CLASS_NAME, "ipl-pagination__next")
        next_button.click()
        time.sleep(2)
        new_review_elements = driver.find_elements(By.CLASS_NAME, "text.show-more__control")
        new_reviews = [review.text for review in new_review_elements]
        reviews.extend(new_reviews)
    except:
        pass

    driver.quit()

    return reviews


if __name__ == "__main__":
    imdb_url = "https://www.imdb.com/title/tt1517268/reviews/?ref_=tt_ql_2"
    all_reviews = scrape_reviews(imdb_url)

    data = {"Reviews": all_reviews}
    df1 = pd.DataFrame(data, index=range(1, len(all_reviews) + 1))

    imdb_url_2 = "https://www.imdb.com/title/tt15398776/reviews/?ref_=tt_ql_2"
    all_reviews_2 = scrape_reviews(imdb_url_2)

    data_2 = {"Reviews": all_reviews_2}
    df2 = pd.DataFrame(data_2, index=range(1, len(all_reviews_2) + 1))

df1.to_csv("film1_reviews.csv", index=False)
df2.to_csv("film2_reviews.csv", index=False)


# Text preprocessing and lemmatization on text data
###################################################
data = pd.read_csv("film1_reviews.csv")
data2 = pd.read_csv("film2_reviews.csv")

lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    filtered_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and lemmatizer.lemmatize(word).isalnum()]
    return ' '.join(filtered_words)

data['cleaned_lemmatized_reviews'] = data['Reviews'].apply(preprocess_text)
data2['cleaned_lemmatized_reviews'] = data2['Reviews'].apply(preprocess_text)


# Sentiment analysis
#####################
def analyze_sentiment(comment):
    blob = TextBlob(comment)
    sentiment = blob.sentiment.polarity
    if sentiment > 0:
        return "Positive"
    elif sentiment < 0:
        return "Negative"
    else:
        return "Neutral"

data['sentiment'] = data['cleaned_lemmatized_reviews'].apply(analyze_sentiment)
data2['sentiment'] = data2['cleaned_lemmatized_reviews'].apply(analyze_sentiment)

sentiment_counts_data = data['sentiment'].value_counts()
sentiment_counts_data2 = data2['sentiment'].value_counts()

total_comments_data = sentiment_counts_data.sum()
total_comments_data2 = sentiment_counts_data2.sum()

def calculate_sentiment_ratio(sentiment_counts, total_comments):
    positive_ratio = sentiment_counts.get('Positive', 0) / total_comments if total_comments != 0 else 0
    negative_ratio = sentiment_counts.get('Negative', 0) / total_comments if total_comments != 0 else 0
    neutral_ratio = sentiment_counts.get('Neutral', 0) / total_comments if total_comments != 0 else 0
    return positive_ratio, negative_ratio, neutral_ratio

positive_ratio_barbie, negative_ratio_barbie, neutral_ratio_barbie = (
    calculate_sentiment_ratio(sentiment_counts_data, total_comments_data))

positive_ratio_oppenheimer, negative_ratio_oppenheimer, neutral_ratio_oppenheimer = (
    calculate_sentiment_ratio(sentiment_counts_data2, total_comments_data2))

# Graph
labels = ['Positive', 'Negative']
data_ratios = [positive_ratio_barbie, negative_ratio_barbie]
data2_ratios = [positive_ratio_oppenheimer, negative_ratio_oppenheimer]

fig, ax = plt.subplots(figsize=(10, 6))

x = range(len(labels))
width = 0.35

bars1 = ax.bar(x, data_ratios, width, label='Barbie', color='pink')
bars2 = ax.bar([pos + width for pos in x], data2_ratios, width, label='Oppenheimer', color='darkorange')

ax.set_ylabel('Ratio', fontsize=14)  # Metin boyutunu büyütme
ax.set_title('Sentiment Ratios Comparison', fontsize=16, fontweight='bold', color='black')
ax.set_xticks([pos + width / 2 for pos in x])
ax.set_xticklabels(labels, fontsize=12)
ax.legend(fontsize=12)

def autolabel(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2%}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=12)  # Yüzde değeri metin boyutunu büyütme

autolabel(bars1)
autolabel(bars2)

plt.tight_layout()
plt.show()


# Identifying the frequently occurring adjectives from the reviews and visualizing them using word clouds
##########################################################################################################
def find_most_common_adjectives(data, num_adjectives=40, unwanted_words=[]):
    adjectives = [word for review in data['cleaned_lemmatized_reviews']
                  for word, pos in TextBlob(review).tags if pos == 'JJ']

    most_common_adjectives = Counter(adjectives).most_common(num_adjectives)
    return most_common_adjectives

most_common_adjectives_data = find_most_common_adjectives(data)
most_common_adjectives_data2 = find_most_common_adjectives(data2)

unwanted_words = ["nolan", "oppenheimer", "cillian", 'barbie', 'much', 'many', 'u',
                  'robbie', 'ryan', 'ken', 'overall', 'girl', 'second', 'last',
                  'whole', 'human', 'sure', 'robert']
def plot_word_cloud(adjectives_data, title, color_func, unwanted_words):
    filtered_adjectives = [(adj, count) for adj, count in adjectives_data if adj
                           not in unwanted_words]
    adjectives_text = ' '.join([adj for adj, _ in filtered_adjectives])
    wordcloud = WordCloud(width=800, height=400, background_color='white',
                          color_func=color_func).generate(adjectives_text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()

def color_func_barbie(word, font_size, position, orientation, random_state=None, **kwargs):
    return "hsl({}, 100%, 70%)".format(random.randint(340, 360))

def color_func_oppenheimer(word, font_size, position, orientation, random_state=None, **kwargs):
    return "hsl({}, 100%, 40%)".format(random.randint(20, 40))


plot_word_cloud(most_common_adjectives_data, 'Most Common Adjectives for Barbie Film',
                color_func_barbie, unwanted_words)
plot_word_cloud(most_common_adjectives_data2, 'Most Common Adjectives for Oppenheimer Film',
                color_func_oppenheimer, unwanted_words)

# Identifying the most significant themes and their corresponding key words using LDA modeling technique
########################################################################################################

def perform_lda(data):
    texts = [review.split() for review in data['cleaned_lemmatized_reviews']]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    num_topics = 5
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)

    topic_keywords = []
    for topic_id in range(num_topics):
        topic_words = lda_model.show_topic(topic_id, topn=50)  # Top 50 keywords
        keywords = [word for word, _ in topic_words]
        topic_name = ", ".join(keywords)
        topic_keywords.append(topic_name)

    return topic_keywords

topic_keywords_data = perform_lda(data)
topic_keywords_data2 = perform_lda(data2)

for idx, topic_name in enumerate(topic_keywords_data):
    print(f"Topic {idx} for 'data': {topic_name}")

print("\n")

for idx, topic_name in enumerate(topic_keywords_data2):
    print(f"Topic {idx} for 'data2': {topic_name}")