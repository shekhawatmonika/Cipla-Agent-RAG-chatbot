from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time, json

options = Options()
options.add_argument("--headless")
driver = webdriver.Chrome(options=options)

# Starting URLs
to_visit = [
    "https://www.cipla.com/about-us",
    "https://www.cipla.com/careers",
    "https://www.cipla.com/products",
    "https://www.cipla.com/investors",
    "https://www.cipla.com/research-development",
    "https://www.cipla.com/sustainability",
    "https://www.cipla.com/media",
    "https://www.cipla.com/contact-us",
]

visited = set()
all_data = []

while to_visit:
    url = to_visit.pop(0)
    if url in visited:
        continue
    visited.add(url)
    print(f"opening: {url}")
    try:
        if "/talentcommunity/apply/" in url or "/apply/" in url:
            print(f"  -> Skipping job application page: {url}")
            continue
        driver.get(url)
        time.sleep(3)

        text = driver.find_element(By.TAG_NAME, "body").text
        # do not include "About Us" and "Contact Us" sections
        if len(text) > 200:
            all_data.append({"url": url, "text": text})

        # Tab / Button click karo
        tabs = driver.find_elements(By.CSS_SELECTOR, "button, [role='tab']")
        for tab in tabs:
            try:
                tab_text = tab.text.strip()
                if tab_text:
                    tab.click()
                    time.sleep(2)
                    new_text = driver.find_element(By.TAG_NAME, "body").text
                    all_data.append({"url": url + f" | TAB: {tab_text}", "text": new_text})
                    print(f"  -> Tab: {tab_text}")
            except:
                pass

        # fetch all links on the page
        links = driver.find_elements(By.TAG_NAME, "a")
        for l in links:
            href = l.get_attribute("href")
            # only add links that are part of the Cipla domain and not PDFs or anchors
            if href and "cipla.com" in href and ".pdf" not in href and "#" not in href:
                if href not in visited and href not in to_visit:
                    to_visit.append(href)
    except Exception as e:
        print(f"Fail: {url}")

driver.quit()

with open("data/cipla_full_site.json", "w", encoding="utf-8") as f:
    json.dump(all_data, f, ensure_ascii=False, indent=2)

print(f"All done! Total {len(all_data)} pages/tabs data")