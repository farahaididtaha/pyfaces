import os
from pathlib import Path
from selenium import webdriver

DRIVER_PATH = os.path.join(
    Path(__file__).parent.resolve(), "chromedriver_linux64/chromedriver"
)

# TODO: Implement Scraper

class ImageScrapper:
    def __init__(self):
        self.driver = None

    def __enter__(self):
        if self.driver is None:
            self.driver = webdriver.Chrome(executable_path=DRIVER_PATH)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.driver:
            self.driver.close()
