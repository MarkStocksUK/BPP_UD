# scrape.py


from urllib.robotparser import RobotFileParser
from urllib.error import URLError
from bs4 import BeautifulSoup
from pathlib import Path
import requests
import time
import random
import json
import logging


# Create a logger
logger = logging.getLogger(__name__)

# Check if we are able to scrape Trustpilot according to their Robots.txt file
def can_we_scrape(url: str, user_agent: str) -> bool:
    """
    Check if the website allows scraping based on its robots.txt file.
    
    Parameters:
    url (str): The URL of the website.
    user_agent (str): The user agent string.
    
    Returns:
    bool: True if scraping is allowed, False otherwise.
    """
    parser = RobotFileParser()
    try:
        parser.set_url(url + "/robots.txt")
        parser.read()
        return parser.can_fetch(user_agent, url)
    except URLError as e:
        logger.error(f"Error fetching robots.txt: {e}")
        return False

assert can_we_scrape("https://www.bbc.com", "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)") == True, "BBC should be scrapeable"
assert can_we_scrape("https://www.google.com", "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)") == False, "Google should not be scrapeable"


# Function to build URLs
def build_page_url(url_to_scrape: str, page_number: int, time_span: str) -> str:
    """
    Build the URL for the specific page and time span.
    
    Parameters:
    url_to_scrape (str): The base URL to scrape.
    page_number (int): The page number.
    time_span (str): The time span for the reviews.
    
    Returns:
    str: The constructed URL.
    """
    if page_number == 1:
        return f"{url_to_scrape}?date={time_span}"
    else:
        return f"{url_to_scrape}?date={time_span}&page={page_number}"

assert build_page_url("https://uk.trustpilot.com/review/www.theaa.com", 1, "week") == "https://uk.trustpilot.com/review/www.theaa.com?date=week", "Expected URL for page 1"
assert build_page_url("https://uk.trustpilot.com/review/www.theaa.com", 2, "week") == "https://uk.trustpilot.com/review/www.theaa.com?date=week&page=2", "Expected URL for page 2"


# Function to get the page contents
def get_page_content(session: requests.Session, url: str, user_agent: str) -> BeautifulSoup:
    """
    Get the content of the specified page.
    
    Parameters:
    session (requests.Session): The session object for making requests.
    url (str): The URL of the page to scrape.
    user_agent (str): The user agent string.
    
    Returns:
    BeautifulSoup: The parsed page content.
    """
    session.headers.update({"User-Agent": user_agent})
    response = session.get(url)
    if response.status_code == 200:
        return BeautifulSoup(response.content, 'html.parser')
    else:
        logger.warning(f"Failed to retrieve page: {url} with status code {response.status_code}")
        return None


# Function to find the overall score
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
    

# Function to find the reviews on that page
def get_reviews_data(page_content: BeautifulSoup) -> list:
    """
    Extract the reviews data from the page content.
    
    Parameters:
    page_content (BeautifulSoup): The parsed page content.
    
    Returns:
    list: A list of reviews.
    """
    try:
        script_tag = page_content.find("script", {"id": "__NEXT_DATA__"})  # Find the script tag
        
        # Assert statement for testing script tag extraction
        assert script_tag is not None, "Expected non-None script tag"

        json_data = json.loads(script_tag.string)  # Load the JSON data from the script tag
        reviews = json_data['props']['pageProps']['reviews']  # Get the reviews from the JSON data
        return reviews
    except (AttributeError, KeyError, json.JSONDecodeError) as e:
        logger.error(f"Error extracting reviews data: {e}")
        return []


# Function to scrape trustpilot for reviews
def scrape():
    """
    Main function to execute the script.
    """
    # Set the variables
    url_to_scrape = "https://uk.trustpilot.com/review/www.bpp.com"
    user_agent = "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)"
    page_number = 1
    time_span = "last30days"
    
    # Create a folder for pages if it doesn't exist
    page_folder_path = Path("pages")
    page_folder_path.mkdir(parents=True, exist_ok=True)
    
    # Assert statement for testing folder existence after folder check
    assert page_folder_path.exists(), "Expected pages folder to exist after check"
    
    # Create a session
    with requests.Session() as session:
        # Check if we can scrape the website
        if not can_we_scrape(url_to_scrape, user_agent):
            logger.error("We can't scrape the website")
            return
        else:
            logger.debug("We can scrape the website")
        
        while True:
            try:
                scrape_url = build_page_url(url_to_scrape, page_number, time_span)
                logger.debug(f"URL to scrape is {scrape_url}")
                
                page_content = get_page_content(session, scrape_url, user_agent)
                logger.debug(f"Page content retrieved for page {page_number}")
                
                if page_content is None:
                    logger.warning("Error getting the page - stopping fetching pages")
                    break
                
                if "We have received an unusually large amount of requests from your IP so you have been rate limited" in page_content.text:
                    logger.error("We have been rate limited - stopping fetching pages")
                    break
                
                logger.debug("We have page content and we have not been rate limited")
                if page_number == 1:
                    overall_rating = get_overall_rating(page_content)
                    logger.debug(f"Overall Rating: {overall_rating}")
                
                page_file_path = page_folder_path / f"page{page_number}.html"
                logger.debug(f"Saving page content to {page_file_path}")
                page_file_path.write_text(str(page_content), encoding="utf-8")
                logger.debug(f"Page {page_number} saved successfully")
                
                page_reviews = get_reviews_data(page_content)
                logger.debug(f"Page {page_number} - Number of reviews: {len(page_reviews)}")
                
                page_number += 1
                time_delay = random.randint(2, 5) # Random delay between 2 and 5 seconds
                logger.debug(f"Sleeping for {time_delay} seconds")
                time.sleep(time_delay)
                
            except Exception as e:
                logger.error(f"Error: {e}")
                break
        
        logger.debug("Finished fetching pages")
        