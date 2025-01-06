from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import pandas as pd

def scrape_youtube_comments(video_link, num_comments=100):
    chrome_options = webdriver.ChromeOptions()
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=chrome_options)
    driver.get(video_link)

    # Use implicit wait to wait for elements to be present
    driver.implicitly_wait(10)  # Adjust the timeout based on your needs

    # Scroll to load more comments
    scrolls = 0
    while scrolls < num_comments // 20:  # Each scroll loads approximately 20 comments
        driver.execute_script(f'window.scrollTo(1, {3000 * scrolls});')
        time.sleep(2)  # Adjust the sleep time based on your needs
        scrolls += 1

    # Explicitly wait for the comments to be loaded
    try:
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.XPATH, '//*[@class="style-scope ytd-comment-renderer"]//*[@id="author-text"]'))
        )
    except Exception as e:
        print(f"Error waiting for comments to load: {e}")

    # Extract usernames and comments
    username_elems = driver.find_elements(By.XPATH, '//*[@class="style-scope ytd-comment-renderer"]//*[@id="author-text"]')
    comment_elems = driver.find_elements(By.XPATH, '//*[@class="style-scope ytd-comment-renderer"]//*[@id="content-text"]')

    items = [{'Author': username.text, 'Comment': comment.text} for username, comment in zip(username_elems, comment_elems)]

    driver.quit()
    scraped_comments = items[:num_comments]
    df = pd.DataFrame(scraped_comments)
    df.to_csv("run.csv")
    return df
