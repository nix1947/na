# You may need to install the required libraries first.
# Open your terminal or command prompt and run:
# pip install selenium webdriver-manager

import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

def initiate_scraper():
    # --- Configuration ---
    # Set to False to run in GUI mode (a browser window will open).
    # Set to True to run in headless mode (no browser window will appear).
    HEADLESS_MODE = False
    EXPECTED_FILENAME = "today_price.csv"
    # --- End Configuration ---

    # --- Set up Download Path ---
    # Save the file in a 'downloads' folder in the same directory as the script.
    download_dir = os.path.abspath("./downloads")
    os.makedirs(download_dir, exist_ok=True) # Create the directory if it doesn't exist
    downloaded_file_path = os.path.join(download_dir, EXPECTED_FILENAME)

    # --- Set up Chrome Browser Options ---
    chrome_options = Options()

    # Configure preferences to automatically save files to the specified download directory.
    # This prevents the "Save As" dialog from appearing.
    prefs = {
        "download.default_directory": download_dir,
        "download.prompt_for_download": False, # Disable prompt
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True # Enable safe browsing
    }
    chrome_options.add_experimental_option("prefs", prefs)

    if HEADLESS_MODE:
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--window-size=1920,1080") # Specify window size for headless

    # --- Main Script ---
    driver = None  # Initialize driver to None
    try:
        # Use webdriver_manager to automatically download and set up the correct chromedriver.
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)

        print("WebDriver initialized successfully.")
        print(f"Files will be downloaded to: {download_dir}")
        
        # If the file already exists, remove it to ensure we're getting a fresh copy.
        if os.path.exists(downloaded_file_path):
            os.remove(downloaded_file_path)
            print(f"Removed existing file: {downloaded_file_path}")

        # 1. Open the target page
        driver.get("https://nepalstock.com.np/today-price")
        print("Navigated to the NEPSE today's price page.")

        # 2. Wait for the download button to be ready and click it
        download_button_xpath = "//a[i[contains(@class, 'fa-download')]]"
        
        print("Waiting for the download button to become clickable...")
        
        wait = WebDriverWait(driver, 20) # Wait for a maximum of 20 seconds
        download_btn = wait.until(EC.element_to_be_clickable((By.XPATH, download_button_xpath)))

        # 3. Click the button to start the download
        print("Download button found. Clicking to start download...")
        driver.execute_script("arguments[0].click();", download_btn)

        # 4. Wait for the download to complete
        # This is a simple wait. For very large files, a more complex check
        # would be needed to see if the file has finished downloading.
        print("Waiting 10 seconds for the download to complete...")
        time.sleep(10)

        if os.path.exists(downloaded_file_path):
            print(f"\n✅ Download complete!")
            print(f"The file '{EXPECTED_FILENAME}' has been saved in your downloads folder: {download_dir}")
        else:
            print(f"\n❌ Download failed. File not found at {downloaded_file_path}")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please ensure you have a stable internet connection and Chrome is installed.")

    finally:
        # 5. Clean up and close the browser
        if driver:
            driver.quit()
            print("\nWebDriver has been closed.")

# This block ensures the script runs when executed directly
if __name__ == "__main__":
    initiate_scraper()

