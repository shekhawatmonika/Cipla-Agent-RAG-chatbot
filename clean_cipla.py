import json
import re

# 1. File load karo - jo Selenium se banayi thi
with open("data/cipla_full_site.json", encoding="utf-8") as f:
    data = json.load(f)

print(f"Before cleaning: {len(data)} pages")

# 2. Ye sab lines kachra hai - inko hatana hai
JUNK_LINES = {
    "Skip to main content", "Search", "Breadcrumb", "Apply",
    "Corporate Information", "Our Presence", "Our Offerings",
    "Quick Links - Footer", "Privacy Policy", "Disclaimer",
    "Terms of Use", "Contact Us", "Sitemap", "Related Links",
    "Accept All Cookies", "Reject Cookies", "More info",
    "© Copyright 2019 Cipla. All rights reserved",
    "Digital Partner : Indus Net Technologies",
    "Chatbot API", "Terms & Condition"
}

JUNK_STARTS = [
    "We use cookies", "By continuing to browse",
    "Beware of Recruitment Fraud"
]

def clean_text(text):
    lines = text.split("\n")
    cleaned = []
    seen = set()

    for line in lines:
        line = line.strip()
        # 1. Khali line, bahut chhoti line, ya junk line hata do
        if not line or len(line) < 3:
            continue
        if line in JUNK_LINES:
            continue
        if any(line.startswith(j) for j in JUNK_STARTS):
            continue
        # 2. Sirf menu jaisi line - jaise "- Any -" ya sirf 1 word ka country
        if line == "- Any -":
            continue
        # 3. Ek hi line baar-baar aaye to ek baar rakho
        if line in seen:
            continue
        seen.add(line)
        cleaned.append(line)

    # 4. Sabko jodo aur extra space saaf karo
    full = " ".join(cleaned)
    full = re.sub(r'\s+', ' ', full).strip()
    return full

clean_data = []
for item in data:
    text = clean_text(item.get("text", ""))
    # 200 se chhota matlab us page me kaam ka kuch tha hi nahi
    if len(text) > 200:
        clean_data.append({
            "url": item["url"],
            "text": text,
            "chars": len(text)
        })

# 3. Saaf file save karo
with open("cipla_clean.json", "w", encoding="utf-8") as f:
    json.dump(clean_data, f, ensure_ascii=False, indent=2)

print(f"After cleaning: {len(clean_data)} pages saved")
print(f"Example - Pehla page ab {clean_data[0]['chars']} chars ka hai")
print(clean_data[0]['text'][:500])