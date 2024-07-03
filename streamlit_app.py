## import neccesary libraries

import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd
import psycopg2 as psql
from collections import Counter
import re
from wordcloud import WordCloud
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download(["vader_lexicon"])
nltk.download(["stopwords"])
from nltk.corpus import stopwords
from datetime import datetime, timedelta


## set up page

st.set_page_config(
   page_title="UK News Sentiment",
   page_icon="📰",
   layout="centered",
   initial_sidebar_state="collapsed",
)

## get secrets

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




## columns for title section
col1, col2 = st.columns([3,1])

## create title and image
with col1:
    st.title('Whats in the UK news?')

with col2:
    st.image('https://github.com/Charlie-martinzzz/Capstone/blob/main/News.jpg?raw=true')


## Word cloud for top 50 words in titles

st.header('Words of the week')

st.write(' ')
st.write(' ')

## Convert 'date' column to datetime
df['date'] = pd.to_datetime(df['date'])


## Function to preprocess text

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\b\w{1,2}\b', '', text)  # Remove short words
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

## Get the current date and the start of the week 
current_date = datetime.now()
start_of_week = current_date - timedelta(days=current_date.weekday())

## Filter DataFrame for rows from the current week
df_current_week = df[df['date'] >= start_of_week]

## Combine all titles into one large string
combined_text = ' '.join(df_current_week['title'].apply(preprocess_text))

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



## select the 10 most occuring sources
top_sources = df['source'].value_counts().head(10).index

## Dropdown to select a news source
selected_source = st.selectbox('Choose a News Source To See Their Most Recent Story', top_sources)

st.write(' ')

## Display the image corresponding to the selected source
selected_icon = df.loc[df['source'] == selected_source, 'icon'].iloc[0]

st.image(selected_icon, width = 500)

st.write(' ')

# Filter DataFrame to get the most recent story for the selected source
recent_story = df[df['source'] == selected_source].nlargest(1, 'id')

recent_story_link = recent_story.iloc[0]['link']
recent_story_title = recent_story.iloc[0]['title']

st.markdown(f'<a href="{recent_story_link}" target="_blank" rel="noopener noreferrer" style="font-size: 20px;">{recent_story_title}</a>', unsafe_allow_html=True)

st.write(' ')




st.title('News Sentiment')

st.subheader('Sentiment Score Distribution')

st.write(' ')

# Initialize VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Function to calculate sentiment scores
def calculate_sentiment(title):
    scores = sid.polarity_scores(title)
    return scores['compound']

# Add a new column 'sentiment_score' to the DataFrame
df['sentiment_score'] = df['title'].apply(calculate_sentiment)

# Display histogram of sentiment scores
fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(df['sentiment_score'], bins=20, color='skyblue', edgecolor='black')
ax.set_xlabel('Sentiment Score')
ax.set_ylabel('Frequency')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Display the plot in Streamlit
st.pyplot(fig)

st.write(' ')

st.subheader('Average sentiment per source (Top 5 sources)')

st.write(' ')

# Get the top 10 most frequently occurring sources
top_5_sources = df['source'].value_counts().nlargest(5).index

# Filter the DataFrame to include only the top 10 sources
filtered_df = df[df['source'].isin(top_5_sources)]

# Calculate average sentiment score for each of the top 10 news sources
avg_sentiment = filtered_df.groupby('source')['sentiment_score'].mean().reset_index()

# Create a bar chart for average sentiment score by the top 10 news sources
fig2, ax = plt.subplots(figsize=(10, 6))
ax.bar(avg_sentiment['source'], avg_sentiment['sentiment_score'], color='skyblue', edgecolor='black')
ax.set_xlabel('News Source')
ax.set_ylabel('Average Sentiment Score')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xticks(rotation=45)

# Display the bar chart in Streamlit
st.pyplot(fig2)



# Identify the most negative and most positive stories
most_negative_story = df.loc[df['sentiment_score'].idxmin()]
most_positive_story = df.loc[df['sentiment_score'].idxmax()]

# Display the titles and links of the most negative and most positive stories with larger font size
st.subheader('Most Negative Story')

st.markdown(f'<a href="{most_negative_story["link"]}" style="font-size: 20px;">{most_negative_story["title"]}</a>', 
    unsafe_allow_html=True)

st.subheader('Most Positive Story')

st.markdown(f'<a href="{most_positive_story["link"]}" style="font-size: 20px;">{most_positive_story["title"]}</a>', 
    unsafe_allow_html=True)



