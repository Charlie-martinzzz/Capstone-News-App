## Firstly, import neccesary libraries

import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd
import psycopg2 as psql
from collections import Counter
import re
from wordcloud import WordCloud
import nltk
nltk.download(["vader_lexicon"])
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download(["stopwords"])
from nltk.corpus import stopwords
from datetime import datetime, timedelta


## set up the page

st.set_page_config(
   page_title="UK News Sentiment",
   page_icon="📰",
   layout="centered",
   initial_sidebar_state="auto",
)

def add_bg_from_local():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://github.com/Charlie-martinzzz/Capstone-News-App/blob/main/news_backgound2.jpg?raw=true");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local()




## get secrets from streamlit secrets - must be in the gitignore for security

user = st.secrets['SQL_USER']
password = st.secrets['SQL_PASSWORD']
my_host = st.secrets['HOST']

## access the database

conn = psql.connect(database = 'pagila',
                    user = user,
                    password = password,
                    host = my_host,
                    port = 5432
                    )

cur = conn.cursor()

## Execute the SQL query and fetch the data into a DataFrame

df = pd.read_sql_query("SELECT * FROM student.capstone_charlie", conn)

## Close the cursor and connection

cur.close()
conn.close()



## create title 
st.title('Whats in the UK news?')


## these blank st.write's create whitespace for dashboard readability
st.write(' ')
st.write(' ')


## This section creates a word cloud for top 50 words in titles

st.header('Word Cloud Generator')
st.write('Choose a start and end date to see the top 50 words')

## Firstly, create the function to preprocess text
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\b\w{1,2}\b', '', text)  # Remove short words
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)


## Convert 'date' column to datetime
df['date'] = pd.to_datetime(df['date'])


col1, col2 = st.columns([1,1])

## Date input for start and end date

with col1:
    start_date = st.date_input('Start date', min_value=datetime(2024, 6, 29), value=datetime(2024, 6, 29))
with col2:
    end_date = st.date_input('End date', min_value=start_date, value=datetime.now().date())

st.write(' ')

## Filter DataFrame for the selected date range
df_filtered = df[(df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))]

## Combine all titles into one large string
combined_text = ' '.join(df_filtered['title'].apply(preprocess_text))

## Get the top 50 words
word_counts = Counter(combined_text.split())
top_50_words = dict(word_counts.most_common(50))

## Generate word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(top_50_words)

## Convert word cloud to image
wordcloud_image = wordcloud.to_image()

## Display the word cloud 
st.image(wordcloud_image, use_column_width=True)

st.write(' ')
st.write(' ')

st.header('Top news sources')

## Top 10 sources and recent story link

col3, col4 = st.columns(2)

## select the 10 most occuring sources
top_sources = df['source'].value_counts().head(10).index

## Dropdown to select a news source
with col3:
    selected_source = st.selectbox('Choose a News Source To See Their Most Recent Story', top_sources)


## Filter DataFrame to get the most recent story for the selected source
recent_story = df[df['source'] == selected_source].nlargest(1, 'id')

recent_story_link = recent_story.iloc[0]['link']
recent_story_title = recent_story.iloc[0]['title']

with col4:
    st.write(' ')
    st.markdown(f'<a href="{recent_story_link}" target="_blank" rel="noopener noreferrer" style="font-size: 20px;">{recent_story_title}</a>', unsafe_allow_html=True)


st.write(' ')

## Display the image corresponding to the selected source
selected_icon = df.loc[df['source'] == selected_source, 'icon'].iloc[0]

st.image(selected_icon, width = 500)

st.write(' ')
st.write(' ')


## this section covers the sentiment analysis exploration

## create title
st.header('News Sentiment')

st.write(' ')

## Initialize VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

## Function to calculate sentiment scores
def calculate_sentiment(title):
    scores = sid.polarity_scores(title)
    return scores['compound']

## Add a new column 'sentiment_score' to the DataFrame
df['sentiment_score'] = df['title'].apply(calculate_sentiment)

## Dropdown box for selecting which figure to display
figure_choice = st.selectbox('Select Figure', ['Sentiment Score Distribution', 'Average sentiment per source (Top 5 sources)', 'Average sentiment score per day'])

# Display selected figure based on user choice
if figure_choice == 'Sentiment Score Distribution':
    st.subheader('Sentiment Score Distribution')
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(df['sentiment_score'], bins=20, color='skyblue', edgecolor='black')
    ax.set_xlabel('Sentiment Score')
    ax.set_ylabel('Frequency')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    st.pyplot(fig)

elif figure_choice == 'Average sentiment per source (Top 5 sources)':
    st.subheader('Average sentiment per source (Top 5 sources)')
    top_5_sources = df['source'].value_counts().nlargest(5).index
    filtered_df = df[df['source'].isin(top_5_sources)]
    avg_sentiment = filtered_df.groupby('source')['sentiment_score'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(avg_sentiment['source'], avg_sentiment['sentiment_score'], color='skyblue', edgecolor='black')
    ax.set_xlabel('News Source')
    ax.set_ylabel('Average Sentiment Score')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xticks(rotation=45)
    st.pyplot(fig)

elif figure_choice == 'Average sentiment score per day':
    st.subheader('Average sentiment score per day')
    start_filter_date = pd.Timestamp('2024-06-29')
    filtered_df2 = df[df['date'] >= start_filter_date]
    avg_sentiment_by_date = filtered_df2.groupby('date')['sentiment_score'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=avg_sentiment_by_date, x='date', y='sentiment_score', marker='o', ax=ax)
    ax.set_ylabel('Average Sentiment Score')
    start_date = avg_sentiment_by_date['date'].min()
    end_date = avg_sentiment_by_date['date'].max()
    ax.set_xticks([start_date, end_date])
    ax.set_xticklabels([start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    st.pyplot(fig)

st.write(' ')
st.write(' ')

# Identify the most negative and most positive stories
most_negative_story = df.loc[df['sentiment_score'].idxmin()]
most_positive_story = df.loc[df['sentiment_score'].idxmax()]

# Display the titles and links of the most negative and most positive stories with larger font size
st.subheader('Most Negative Story')

st.markdown(f'<a href="{most_negative_story["link"]}" style="font-size: 20px; color: red;">{most_negative_story["title"]}</a>', 
    unsafe_allow_html=True)

st.subheader('Most Positive Story')

st.markdown(f'<a href="{most_positive_story["link"]}" style="font-size: 20px; color: green;">{most_positive_story["title"]}</a>', 
    unsafe_allow_html=True)

