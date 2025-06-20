# sentiment_and_topics.py
# This script performs sentiment analysis and topic modelling on a dataset of reviews.


# Importing libraries

# Standard library imports
import os
import re
import logging
from datetime import datetime

# Third party imports
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import LatentDirichletAllocation as LDA
import pyLDAvis
import pyLDAvis.lda_model
from wordcloud import WordCloud


# Create a logger
logger = logging.getLogger(__name__)


# Function to read CSV file
def read_csv_file(file_path):
    """
    Reads a CSV file and returns a pandas DataFrame.
    
    :param file_path: Path to the CSV file.
    :return: DataFrame containing the data from the CSV file.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except pd.errors.EmptyDataError:
        print(f"No data in file: {file_path}")
        return None
    except pd.errors.ParserError:
        print(f"Error parsing file: {file_path}")
        return None




# Function to perform sentiment analysis on the reviews text
def analyze_sentiment(text):
    """
    Analyzes the sentiment of the given text using TextBlob.
    
    :param text: The text to analyze.
    :return: A tuple containing polarity and subjectivity.
    """
    analysis = TextBlob(text)
    return analysis.sentiment.polarity, analysis.sentiment.subjectivity



# Function to create the outputs directory if it doesn't exist
def create_outputs_directory():
    """
    Creates the outputs directory if it doesn't exist.
    """
    try:
        # Check if the directory already exists
        if os.path.exists('outputs'):
            logger.info("Output directory already exists.")
        else:
            logger.info("Creating output directory...")
            # Create the output directory
            os.makedirs('outputs', exist_ok=True)
            logger.info("Output directory created successfully.")
    except Exception as e:
        logger.error(f"Error creating output directory: {e}")


# Function that will generate metrics for each source type
def generate_metrics(df, source_type):
    """
    Generates metrics for the given source type.
    
    :param df: DataFrame containing the data.
    :param source_type: The source type to filter by.
    :return: A DataFrame containing the metrics.
    """
    logger.debug(f"Generating metrics for source type: {source_type}")
    
    # Check if the source type is valid
    if source_type not in df['source'].unique() and source_type != 'all':
        logger.error(f"Invalid source type: {source_type}. Must be one of {df['source'].unique()} or 'all'.")
        
    if source_type == 'all':
        # If 'all' is selected, use the entire DataFrame
        filtered_df = df
    else:
        # Filter the DataFrame by the source type
        filtered_df = df[df['source'] == source_type]

    # Count the number of reviews
    review_count = filtered_df.shape[0]

    #Count the mean and mode average ratings
    mean_rating = filtered_df['rating'].mean()
    mode_rating = filtered_df['rating'].mode()[0] if not filtered_df['rating'].mode().empty else None

    # Calculate average polarity and subjectivity
    avg_polarity = filtered_df['polarity'].mean()
    avg_subjectivity = filtered_df['subjectivity'].mean()
    
    # Count the number of positive and negative reviews
    positive_count = (filtered_df['polarity'] > 0).sum()
    negative_count = (filtered_df['polarity'] < 0).sum()

    # Calculate the percentage of positive and negative reviews avoiding division by zero
    positive_pct = (positive_count / review_count) * 100 if review_count > 0 else 0
    negative_pct = (negative_count / review_count) * 100 if review_count > 0 else 0
    
    return pd.DataFrame({
        'Source Type': [source_type],
        'Review Count': [review_count],
        'Mean Rating': [round(mean_rating, 1)], # to one decimal places
        'Mode Rating': [mode_rating],
        'Average Polarity': [round(avg_polarity, 2)], # to two decimal places
        'Average Subjectivity': [round(avg_subjectivity, 2)], # to two decimal places
        'Positive Reviews': [positive_count],
        'Positive Reviews (%)': [round(positive_pct, 1)], # to one decimal place 
        'Negative Reviews': [negative_count],
        'Negative Reviews (%)': [round(negative_pct, 1)], # to one decimal place 
    })


# Function to generate metrics for all source types
def generate_metrics_for_all_sources(df):
    """
    Generates metrics for all source types in the DataFrame.
    
    :param df: DataFrame containing the data.
    :return: A DataFrame containing the metrics for all source types.
    """
    source_types = df['source'].unique()
    metrics_list = []
    
    # generate metrics for each source type 
    for source_type in source_types:
        metrics = generate_metrics(df, source_type)
        metrics_list.append(metrics)
    
    # generate overal metrics
    overall_metrics = generate_metrics(df, 'all')
    metrics_list.append(overall_metrics)

    return pd.concat(metrics_list, ignore_index=True)


# Function to generate a line chart of the average rating over time
def plot_average_rating_over_time(df):
    """
    Plots the average rating over time.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'pub_date' and 'rating' columns.

    Returns:
    None
    """
    try:
        # Convert the 'pub_date' column to datetime format
        df['pub_date'] = pd.to_datetime(df['pub_date'], errors='coerce')
        
        # Remove timestamps from the pub_date column
        df['pub_date'] = df['pub_date'].dt.date
        
        # Drop rows with invalid dates
        df = df.dropna(subset=['pub_date'])
        
        # Group by date and calculate the mean rating for each date
        grouped_df = df.groupby('pub_date')['rating'].mean().reset_index()
        
        # Sort the DataFrame by date
        grouped_df = grouped_df.sort_values('pub_date')

        # Create a line chart of the average rating over time
        plt.figure(figsize=(12, 6))
        plt.plot(grouped_df['pub_date'], grouped_df['rating'], marker='o', linestyle='-')
        plt.title(f"Average Rating Over Time. Overall rating is {round(df['rating'].mean(), 1)}")
        plt.xlabel('Review Date')
        plt.ylabel('Average Rating')
        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))  # Show every 7 days
        plt.gcf().autofmt_xdate()  # Auto format date labels
        plt.ylim(1, 5) # Set y-axis limits to rating scale (1-5)
        plt.grid(False)
        plt.tight_layout()
        plt.savefig('outputs/average_rating_over_time.png')

        logger.info("Line chart of average rating over time saved successfully.")
        print("Line chart of average rating over time saved to outputs/average_rating_over_time.png")

    except Exception as e:
        logger.error(f"Error in plot_average_rating_over_time: {e}")



# Function to remove custom stop words
def remove_custom_stopwords(text, custom_stopwords):
    """
    Removes custom stop words from the text.
    
    :param text: The text to clean.
    :param custom_stopwords: A list of custom stop words to remove.
    :return: Cleaned text without custom stop words.
    """
    try:
        for word in custom_stopwords:
            text = re.sub(r'\b' + re.escape(word) + r'\b', '', text, flags=re.IGNORECASE)
        return text
    except Exception as e:
        logger.error(f"Error in remove_custom_stopwords: {e}")
        return text


# Function to clean text
def clean_text(text):
    """
    Cleans the input text by removing non-alphabetic characters, extra spaces, leading/trailing spaces, and converting to lowercase.

    Parameters:
    text (str): Input text to be cleaned.

    Returns:
    str: Cleaned text.
    """
    try:
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
        text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
        text = text.strip()  # Remove leading and trailing spaces
        text = text.lower()  # Convert to lowercase
        return text
    except Exception as e:
        logger.error(f"Error in clean_text: {e}")
        return text


# Function to vectorise text
def vectorise_text(text_data):
    """
    Vectorises the text data using CountVectorizer.
    
    :param text_data: The text data to vectorise.
    :return: The vectorised text data.
    """
    try:
        custom_stopwords = ['aa', 'car']
        # Add custom stop words to the default English stop words
        combined_stop_words = list(ENGLISH_STOP_WORDS) + custom_stopwords

        vectoriser = CountVectorizer(stop_words=combined_stop_words)
        transformed_text = vectoriser.fit_transform(text_data)
        return transformed_text, vectoriser
    except Exception as e:
        logger.error(f"Error in vectorise_text: {e}")
        return None, None


# Function to perform topic modelling
def perform_topic_modelling(vectorised_data, n_topics=5):
    """
    Performs topic modelling using LDA.
    
    :param vectorised_data: The vectorised text data.
    :param n_topics: The number of topics to extract.
    :return: The fitted LDA model.
    """
    # Check if the number of topics is greater than the number of reviews
    if n_topics > vectorised_data.shape[0]:
        logger.warning(f"Number of topics ({n_topics}) is greater than the number of reviews ({vectorised_data.shape[0]}). Setting n_topics to {vectorised_data.shape[0]}.")
        n_topics = vectorised_data.shape[0]

    try:
        # Create and fit the LDA model
        lda = LDA(n_components=n_topics, random_state=1234)
        lda.fit(vectorised_data)
        return lda
    except Exception as e:
        logger.error(f"Error in perform_topic_modelling: {e}")
        return None
    

# Function to print the top words for each topic
def print_top_words(model, feature_names, n_top_words):
    """
    Prints the top words for each topic in the LDA model.
    
    :param model: The fitted LDA model.
    :param feature_names: The names of the features (words).
    :param n_top_words: The number of top words to print for each topic.
    """
    try:
        if model is None or feature_names is None:
            logger.error("Model or feature names are None. Cannot print top words.")
            return

        print("\n") # for readability
        print("Top words for each topic:")
        print("===================================")
        # Print the top words for each topic
        for topic_num, topic in enumerate(model.components_):
            print(f"Topic {topic_num}:")
            print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
        
        print("\n") # for readability
    except Exception as e:
        logger.error(f"Error in print_top_words: {e}")


# Function to visualise the topics using pyLDAvis
def visualise_topics(lda_model, vectorised_data, vectoriser):
    """
    Visualises the topics using pyLDAvis.
    
    :param lda_model: The fitted LDA model.
    :param vectorised_data: The vectorised text data.
    :param vectoriser: The CountVectorizer used for vectorisation.
    """
    try:
        # Prepare the visualisation
        pyLDAvis.enable_notebook()
        vis = pyLDAvis.lda_model.prepare(lda_model, vectorised_data, vectoriser)
        
        # Display the visualisation in a Jupyter notebook
        pyLDAvis.display(vis)
        
        # Save the visualisation to an HTML file
        pyLDAvis.save_html(vis, 'outputs/lda_visualisation.html')
        
        logger.info("LDA visualisation saved successfully.")
        print("Interactive LDA visualisation saved to outputs/lda_visualisation.html")

    except Exception as e:
        logger.error(f"Error in visualise_topics: {e}")
    

    
# Function to create a word cloud
def create_wordcloud(text_data, title):
    """
    Creates a word cloud from the given text data.
    
    :param text_data: The text data to create the word cloud from.
    :param title: The title for the word cloud.
    """
    try:
    
        # Join all the text data into a single string
        text = " ".join(text_data)
        
        # Create a word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        
        # Display the word cloud
        #plt.figure(figsize=(10, 5))
        #plt.imshow(wordcloud, interpolation='bilinear')
        #plt.axis('off')
        #plt.title(title)
        #plt.show()
    
        # Save the word cloud to a file
        wordcloud.to_file('outputs/Negative Reviews Wordcloud.png')
        logger.info("Word cloud saved successfully.")
        print("Negative reviews word cloud saved to outputs/Negative Reviews Wordcloud.png")

    except Exception as e:
        logger.error(f"Error in create_wordcloud: {e}")
        



# Main script
def sentiment_and_topics():
    """
    Main function to execute the script.
    """

    # Read the csv file into a pandas dataframe
    df = read_csv_file('csv files/reviews.csv')

    if df is not None:
        # Apply the sentiment analysis function to the 'text' column
        df[['polarity', 'subjectivity']] = df['text'].apply(lambda x: pd.Series(analyze_sentiment(x)))

        # Save the updated DataFrame to a new CSV file
        csv_file_path = 'csv files/reviews_with_sentiment.csv'
        df.to_csv(csv_file_path, index=False)

        #generate the metrics for each source type
        source_metrics_df = generate_metrics_for_all_sources(df)
        logger.info("Metrics for all sources generated successfully.")

        # create the outputs directory if it doesn't exist
        create_outputs_directory()

        # Save the metrics DataFrame to a CSV file
        output_metrics_filepath = 'outputs/source_metrics.csv'
        source_metrics_df.to_csv(output_metrics_filepath, index=False)
        logger.info(f"Metrics for all sources saved to {output_metrics_filepath}")
        print("Source metrics generated and saved to outputs/source_metrics.csv")

        #create a dataframe of the negative reviews
        negative_reviews = df[(df['polarity'] < 0) & (df['rating'] < 3)]

        # save dataframe to a csv file
        negative_reviews.to_csv('outputs/negative_reviews.csv', index=False)
        logger.info("Negative reviews saved to outputs/negative_reviews.csv")

        # Define custom stop words for the company
        custom_stop_words = ['aa', 'car']

        # Apply the remove_custom_stopwords function using .loc to avoid SettingWithCopyWarning
        negative_reviews.loc[:, 'text'] = negative_reviews['text'].apply(remove_custom_stopwords, custom_stopwords=custom_stop_words)

        # Apply the cleam text function using .loc to avoid SettingWithCopyWarning
        negative_reviews.loc[:, 'text'] = negative_reviews['text'].apply(clean_text)

        # Create word cloud for negative reviews
        create_wordcloud(negative_reviews['text'], "Word Cloud of Negative Reviews")

        # Vectorise the cleaned text data
        transformed_text, vectoriser = vectorise_text(negative_reviews['text'])

        if transformed_text is not None and vectoriser is not None:
            # Perform topic modelling
            lda_model = perform_topic_modelling(transformed_text, n_topics=5)

            if lda_model is not None:
                logger.info("Topic modelling completed successfully.")

                # Print the top words for each topic
                print_top_words(lda_model, vectoriser.get_feature_names_out(), n_top_words=5)

                # Visualise the topics using pyLDAvis
                visualise_topics(lda_model, transformed_text, vectoriser)
            else:
                logger.error("Error in topic modelling. Exiting.")
                exit(1)
        else:
            logger.error("Error in vectorising text data. Exiting.")
            exit(1)
        
        # generate line chart for rating over time
        plot_average_rating_over_time(df)

    else:
        logger.error("Error reading the CSV file. Exiting.")
        exit(1)
    







