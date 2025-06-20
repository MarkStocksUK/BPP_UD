# Structure.py


from bs4 import BeautifulSoup
from pathlib import Path
import json
import pandas as pd
import logging
import os


# Create a logger
logger = logging.getLogger(__name__)

# Function to create the csv files directory if it doesn't exist
def create_csv_files_directory():
    """
    Creates the csv files directory if it doesn't exist.
    """
    try:
        # Check if the directory already exists
        if os.path.exists('csv files'):
            logger.info("csv files directory already exists.")
        else:
            logger.info("Creating csv files directory...")
            # Create the csv files directory
            os.makedirs('csv files', exist_ok=True)
            logger.info("csv files directory created successfully.")
    except Exception as e:
        logger.error(f"Error creating csv files directory: {e}")


def get_overall_rating(page_content: BeautifulSoup) -> str:
    """
    Extract the overall rating from the page content by searching for <p class="typography_body-l__v5JLj typography_appearance-subtle__PYOVM" in the PageContent
    
    Parameters:
    page_content (BeautifulSoup): The parsed page content.
    
    Returns:
    str: The overall rating.
    """
    try:
        overall_rating = page_content.find("p", class_="typography_body-l__v5JLj typography_appearance-subtle__PYOVM").text

        # Assert statement for testing overall rating extraction
        assert overall_rating is not None, "Expected non-None overall rating"

        return overall_rating
    except AttributeError as e:
        logger.error(f"Error extracting overall rating: {e}")
        return None


def get_reviews_data(page_content: BeautifulSoup) -> list:
    """
    Extract the reviews data from the page content.
    
    Parameters:
    page_content (BeautifulSoup): The parsed page content.
    
    Returns:
    list: A list of reviews.
    """
    try:
        script_tags = page_content.find("script", {"id": "__NEXT_DATA__"})  # Find the script tags
        json_data = json.loads(script_tags.string)  # Load the JSON data from the first script tag
        reviews = json_data['props']['pageProps']['reviews']  # Get the reviews from the JSON data
        return reviews
    except (IndexError, KeyError, json.JSONDecodeError) as e:
        logger.error(f"Error extracting reviews data: {e}")
        return []


# Define a function to read in a page and extract the reviews
def read_page_and_extract_reviews(page_file_path: Path) -> list:
    """
    Read a page file and extract the reviews.
    
    Parameters:
    page_file_path (Path): The path to the page file.
    
    Returns:
    list: A list of reviews.
    """
    try:
        with open(page_file_path, 'r', encoding='utf-8') as file:
            page_content = file.read()
            soup = BeautifulSoup(page_content, 'html.parser')
            reviews = get_reviews_data(soup)

            
            # Assert statement for testing reviews extraction
            assert isinstance(reviews, list), "Expected reviews to be a list"
            
            return reviews
    except FileNotFoundError as e:
        logger.error(f"Error reading page file: {e}")
        return []
    

# Loop through the files in the pages directory, extract the reviews and append them to a pandas dataframe
def extract_reviews_from_directory(directory_path: Path) -> list:
    """
    Extract reviews from all HTML files in a directory.
    
    Parameters:
    directory_path (Path): The path to the directory containing the HTML files.
    
    Returns:
    list: A list of all reviews extracted from the HTML files.
    """
    all_reviews = []
    logger.debug(f"Extracting reviews from directory: {directory_path}")
    for page_file in directory_path.glob('*.html'):
        logger.debug(f"Processing file: {page_file.name}")
        reviews = read_page_and_extract_reviews(page_file)

        # Assert statement for testing reviews extraction from each file
        assert isinstance(reviews, list), f"Expected reviews from file {page_file.name} to be a list"

        all_reviews.extend(reviews)
    return all_reviews


def collate_reviews_into_dataframe() -> pd.DataFrame:
    """
    Collate the reviews into a pandas DataFrame.
    
    Returns:
    pd.DataFrame: A DataFrame containing all the reviews.
    """
    all_reviews = extract_reviews_from_directory(Path('pages'))
    
    # Assert statement for testing all_reviews type
    assert isinstance(all_reviews, list), "Expected all_reviews to be a list"

    df = pd.DataFrame(all_reviews)

    # Assert statement for testing DataFrame creation
    assert isinstance(df, pd.DataFrame), "Expected df to be a pandas DataFrame"

    return df


def extract_exp_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the experience date from the reviews DataFrame.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the reviews.
    
    Returns:
    pd.DataFrame: The DataFrame with an additional 'exp_date' column.
    """

    # Assert statement for testing input DataFrame type
    assert isinstance(df, pd.DataFrame), "Expected df to be a pandas DataFrame"

    df['exp_date'] = df['dates'].apply(lambda x: x['experiencedDate'] if 'experiencedDate' in x else None)
    # Convert the exp_date column to datetime format
    df['exp_date'] = pd.to_datetime(df['exp_date'], errors='coerce')
    return df


def extract_pub_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the publication date from the reviews DataFrame.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the reviews.
    
    Returns:
    pd.DataFrame: The DataFrame with an additional 'pub_date' column.
    """

    # Assert statement for testing input DataFrame type
    assert isinstance(df, pd.DataFrame), "Expected df to be a pandas DataFrame"

    df['pub_date'] = df['dates'].apply(lambda x: x['publishedDate'] if 'publishedDate' in x else None)
    # Convert the pub_date column to datetime format
    df['pub_date'] = pd.to_datetime(df['pub_date'], errors='coerce')
    return df

# Function to replace the source column values with a more readable version
def replace_source_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace the source column values with a more readable version.

    InvitationLinkApi = invited by company to leave a review
    Organic = left a review without being invited by the company
    BasicLink = old invite link method
    DomainLink = redirected from the company website

    Parameters:
    df (pd.DataFrame): The DataFrame containing the reviews.
    
    Returns:
    pd.DataFrame: The DataFrame with the source column values replaced.
    """

    # Assert statement for testing input DataFrame type
    assert isinstance(df, pd.DataFrame), "Expected df to be a pandas DataFrame"

    df['source'] = df['source'].replace({
        'InvitationLinkApi': 'Invited',
        'Organic': 'Organic',
        'BasicLink': 'Invited',
        'DomainLink': 'From Website'
    })
    return df

def structure():
    """
    Main function to execute the script.
    """
    # create the csv files directory if it doesn't exist
    logger.info("Starting the script...")
    create_csv_files_directory()

    # collate the reviews from all pages in the pages directory into a pandas dataframe
    logger.info("Collating reviews into a DataFrame...")
    df_reviews = collate_reviews_into_dataframe()

    # Extract the experienced date and published date from the reviews DataFrame
    logger.info("Extracting experience and publication dates...")
    df_reviews = extract_exp_date(df_reviews)
    df_reviews = extract_pub_date(df_reviews)

    # Dropping any unnecessary columns
    df_reviews = df_reviews[['id', 'text', 'rating', 'source', 'exp_date', 'pub_date']]

    # Renaming the source values to something more meaningful
    logger.info("Replacing source values with more readable versions...")
    df_reviews = replace_source_values(df_reviews)

    # Save the dataframe to a csv file
    logger.info("Saving the DataFrame to a CSV file...")
    df_reviews.to_csv('csv files/reviews.csv', index=False, encoding='utf-8')
    logger.info("Reviews saved to csv_files/reviews.csv")
