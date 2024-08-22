# 📰 UK News Sentiment App

A simple Streamlit app showcasing Uk news trends, exploring both popular stories and simple sentiment analysis.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://capstone-news-app-kefspgzdrip.streamlit.app/)

capstone.py contains the python script, running on a CRON job that accesses the API and stores the data is a postgreSQL database.

streamlit_app.py contains the code for the streamlit app.


### PLEASE NOTE:
    
This app succesfully scraped data from 29/06/2024 once per hour up until 08/07/2024 before running out of API searches. The searches are reset each month, thus selecting dates after 08/07/2024 will result in missing data and skewed graphs.

### It is recommended to view this app in light mode. This app has 3 main sections:

1. Word Cloud

   This section allows the user to select a start and end date and view the top 50 words from titles in a WordCloud. The WordCloud takes data from all sources and titles within the timeframe.


2. Top Sources and Recent News

   This section allows the user to select one of any of the top 10 sources (determined by number of unique stories from source in database). Selecting a source will display the sources logo, alongside a link to their most recent story.


3. Sentiment Analysis

   The sentiment analysis of this project was done using vader lexicon, a pre trained sentiment analysis model. This model takes a string and returns a score, 1 being the most positive, and -1 being the most negative.

   This section allows the user to explore sentiment analysis in 3 different visualisations:
      
      1. Overall Sentiment Score Distribution

         A simple plot displaying the distribution of sentiment scores from all titles from all sources.

      2. Avergae Sentiment Score Per Source

         A simple plot displaying the avergae sentiment score of the top 5 sources. This was limited to the top 5 sources for readability, alongside avoiding bias. Since other sources have significantly less entries in the database, they tended to show more extreme scores due to one high or low score having an exagerated effect. 

      3. Average Sentiment Score Per Day

         A simple plot showing the average sentiment score per day. This plot used data from all stories and all sources to define an average score for the day.

