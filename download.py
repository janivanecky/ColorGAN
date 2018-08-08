import argparse
import datetime
import json
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# Constants
SCROLL_PAUSE_TIME = 0.5
CHROMEDRIVER_BINARY_PATH = './chromedriver'
ADOBE_COLORS_URL = "https://color.adobe.com/explore/?filter=most-popular&time=all"


def download(output_file_path):
    # Create "Chrome" browser instance through selenium.
    options = Options()
    options.add_argument("--headless")     # So no Chrome window pops up.
    options.add_argument("--log-level=3")  # So logs don't flood stdout.
    browser = webdriver.Chrome(CHROMEDRIVER_BINARY_PATH, chrome_options=options)

    # Load the initial webpage.
    browser.get(ADOBE_COLORS_URL)
    time.sleep(SCROLL_PAUSE_TIME)

    # We're going to scroll all the way down on the "browse most popular colors" page.
    # This needs to be done repeatedly, as webpage doesn't load all the colors at once,
    # but loads them continuously as we're scrolling.
    # Notice that we're waiting after each scroll, to make sure that page is loaded.
    last_height = browser.execute_script("return document.body.scrollHeight")
    while True:
        browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(SCROLL_PAUSE_TIME)
        new_height = browser.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height
    time.sleep(SCROLL_PAUSE_TIME)

    # Get webpage source.
    source = browser.page_source
    browser.quit()

    color_palettes = []
    while True:
        # Each color palette is inside div with class "frame ctooltip",
        # individiual colors are in children divs:
        # 
        # <div class="frame ctooltip">
        #     <div style="background: #849696"></div>
        #     <div style="background: #FEFFFB"></div>
        #     <div style="background: #232D33"></div>
        #     <div style="background: #17384D"></div>
        #     <div style="background: #FF972C"></div>
        # </div>
        source_index = source.find('frame ctooltip')
        if source_index == -1:
            break

        # Move "the pointer" to the current color palette div.
        source = source[source_index:]

        # Get current palette (5 colors in HEX).
        current_palette = []
        for _ in range(5):
            source_index = source.find('background:') + 13
            color = source[source_index:source_index + 6]
            current_palette.append(color)

            # Move "the pointer" so the next color can be parsed.
            source = source[source_index + 6:]
        color_palettes.append(current_palette)

    # Get current timestamp so we know when the dataset was exported.
    timestamp = datetime.datetime.utcnow()
    print("Downloaded {} color palettes.".format(len(color_palettes)))
    print("Timestamp:", timestamp)

    # Create and save output dictionary.
    output = {
        "data": color_palettes,
        "time": str(timestamp),
    }
    with open(output_file_path, 'w') as f:
        json.dump(output, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("output", help="Where to store the downloaded dataset.")
    args = parser.parse_args()

    download(args.output)
