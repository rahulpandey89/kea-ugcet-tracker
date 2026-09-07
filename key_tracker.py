import os
import re
import json
import hashlib
import requests
from dotenv import load_dotenv

from bs4 import BeautifulSoup
from urllib.parse import urljoin
from openai import OpenAI

load_dotenv()
# ============================================================
# CONFIG
# ============================================================

KEA_URL = "https://cetonline.karnataka.gov.in/kea/ugcet2026.aspx"


HF_TOKEN = os.getenv("HF_TOKEN")



if not HF_TOKEN:
    raise ValueError("HF_TOKEN .env mein nahi mila")

MODEL = "openai/gpt-oss-120b:cheapest"

SEEN_FILE = "seen_updates.json"


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/150.0.0.0 Safari/537.36"
    )
}


# ============================================================
# HUGGING FACE CLIENT
# ============================================================

if not HF_TOKEN:
    raise ValueError(
        "HF_TOKEN environment variable nahi mila."
    )


client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN
)


# ============================================================
# 1. SCRAPE KEA
# ============================================================

def scrape_kea():
    html_file = os.getenv("KEA_HTML_FILE", "kea.html")

    print(f"📄 Reading KEA HTML from: {html_file}")

    with open(html_file, "r", encoding="utf-8") as f:
        html = f.read()

    print(f"✅ HTML loaded: {len(html)} characters")

    soup = BeautifulSoup(html, "html.parser")

    accordion = soup.select_one(
        "#ContentPlaceHolder1_req_accordion"
    )

    if not accordion:
        raise Exception("KEA notification section nahi mila")

    updates = []

    for card in accordion.select(".card"):
        link = card.select_one(
            ".card-header a[id^='lnk']"
        )

        if not link:
            continue

        title = link.get_text(" ", strip=True)

        href = link.get("href")

        if href:
            href = urljoin(
                "https://cetonline.karnataka.gov.in/kea/ugcet2026.aspx",
                href
            )

        updates.append({
            "id": link.get("id", "").replace("lnk", ""),
            "title": title,
            "url": href
        })

    print(f"📢 Total updates found: {len(updates)}")

    return updates


# ============================================================
# 2. KEYWORD FILTER
# ============================================================

IMPORTANT_KEYWORDS = [

    # English
    "result",
    "rank",
    "answer key",
    "seat allotment",
    "choice entry",
    "counselling",
    "counseling",
    "cutoff",
    "document verification",
    "admission",
    "fee",
    "deadline",
    "exam",
    "allotment",
    "round",

    # Kannada
    "ಫಲಿತಾಂಶ",
    "ಸೀಟು ಹಂಚಿಕೆ",
    "ಆಯ್ಕೆ",
    "ಕೌನ್ಸೆಲಿಂಗ್",
    "ದಾಖಲೆ",
    "ಪ್ರವೇಶ",
]


def looks_important(title):

    title_lower = title.lower()

    for keyword in IMPORTANT_KEYWORDS:

        if keyword.lower() in title_lower:
            return True

    return False


# ============================================================
# 3. LOAD SEEN UPDATES
# ============================================================

def load_seen_updates():

    if not os.path.exists(SEEN_FILE):

        return set()

    try:

        with open(
            SEEN_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        return set(data)

    except Exception:

        return set()


# ============================================================
# 4. SAVE SEEN UPDATES
# ============================================================

def save_seen_updates(seen):

    with open(
        SEEN_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            list(seen),
            file,
            ensure_ascii=False,
            indent=2
        )


# ============================================================
# 5. CREATE UNIQUE UPDATE ID
# ============================================================

def get_update_key(update):

    if update.get("id"):

        return update["id"]

    raw = (
        update["title"]
        + str(update["url"])
    )

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()


# ============================================================
# 6. AI ANALYSIS
# ============================================================

def analyze_update(update):

    title = update["title"]
    url = update["url"]

    prompt = f"""
Analyze this KEA/KCET 2026 notification.

TITLE:
{title}

URL:
{url}

Determine whether this is important for a KCET/UGCET 2026 student.

Important categories:
- Result
- Rank
- Answer Key
- Seat Allotment
- Choice Entry
- Counselling
- Cutoff
- Document Verification
- Admission
- Fee
- Important Deadline
- Exam
- Round-wise allotment

The notification can be in English, Kannada, or mixed language.

Rules:
- Understand Kannada and English.
- Never invent a date.
- Never invent a time.
- If date is not explicitly mentioned, use null.
- If time is not explicitly mentioned, use null.
- Keep message short.
"""


    try:

        response = client.chat.completions.create(

            model=MODEL,

            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a KEA notification classifier. "
                        "Return the requested structured JSON."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0,

            max_tokens=300,

            # IMPORTANT:
            # Force valid JSON from the model
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "kea_notification",
                    "strict": True,
                    "schema": {
                        "type": "object",

                        "properties": {

                            "important": {
                                "type": "boolean"
                            },

                            "type": {
                                "type": [
                                    "string",
                                    "null"
                                ]
                            },

                            "message": {
                                "type": [
                                    "string",
                                    "null"
                                ]
                            },

                            "date": {
                                "type": [
                                    "string",
                                    "null"
                                ]
                            },

                            "time": {
                                "type": [
                                    "string",
                                    "null"
                                ]
                            }
                        },

                        "required": [
                            "important",
                            "type",
                            "message",
                            "date",
                            "time"
                        ],

                        "additionalProperties": False
                    }
                }
            }
        )

        content = response.choices[0].message.content

        print("\n🔎 RAW MODEL RESPONSE:")
        print(repr(content))

        if not content:

            raise ValueError(
                "Model ne empty response return kiya."
            )

        result = json.loads(content)

        return result


    except Exception as error:

        print(
            f"\n❌ AI analysis failed: {error}"
        )

        return {
            "important": False,
            "type": None,
            "message": None,
            "date": None,
            "time": None
        }


# ============================================================
# 7. PRINT RESULT
# ============================================================

def print_notification(update, result):

    date = result.get("date")

    time = result.get("time")

    if not date:
        date = "Not mentioned by KEA"

    if not time:
        time = "Not mentioned by KEA"

    print("\n" + "=" * 65)

    print("🚨 KEA UGCET 2026 UPDATE")

    print("=" * 65)

    print(
        f"\n📌 {result.get('message')}"
    )

    print(
        f"📂 Type: {result.get('type')}"
    )

    print(
        f"📅 Date: {date}"
    )

    print(
        f"⏰ Time: {time}"
    )

    print(
        f"🔗 {update.get('url')}"
    )

    print("=" * 65)


# ============================================================
# 8. MAIN TRACKER
# ============================================================

def run_tracker():

    print("\n")
    print("=" * 65)
    print("🚀 KEA UGCET 2026 TRACKER")
    print("=" * 65)

    # --------------------------------------------------------
    # Scrape website
    # --------------------------------------------------------

    updates = scrape_kea()

    # --------------------------------------------------------
    # Load previously processed updates
    # --------------------------------------------------------

    seen = load_seen_updates()

    print(
        f"📦 Previously seen: {len(seen)}"
    )

    new_count = 0
    important_count = 0

    # --------------------------------------------------------
    # Process notifications
    # --------------------------------------------------------

    for update in updates:

        update_key = get_update_key(update)

        # Already processed
        if update_key in seen:
            continue

        new_count += 1

        print("\n" + "-" * 65)

        print(
            f"🆕 NEW UPDATE"
        )

        print(
            f"📌 {update['title']}"
        )

        print(
            f"🔗 {update['url']}"
        )

        # ----------------------------------------------------
        # Mark as seen
        # ----------------------------------------------------

        seen.add(update_key)

        # ----------------------------------------------------
        # Keyword filter
        # ----------------------------------------------------

        if not looks_important(
            update["title"]
        ):

            print(
                "ℹ️ Keyword filter: Not important"
            )

            continue

        # ----------------------------------------------------
        # AI
        # ----------------------------------------------------

        print(
            "🤖 Analyzing with Hugging Face..."
        )

        result = analyze_update(
            update
        )

        print(
            "\nAI RESULT:"
        )

        print(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2
            )
        )

        # ----------------------------------------------------
        # Important notification
        # ----------------------------------------------------

        if result.get("important"):

            important_count += 1

            print_notification(
                update,
                result
            )

        else:

            print(
                "ℹ️ AI says: Not important"
            )

    # --------------------------------------------------------
    # Save processed updates
    # --------------------------------------------------------

    save_seen_updates(
        seen
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 65)
    print("📊 SUMMARY")
    print("=" * 65)

    print(
        f"Total KEA updates : {len(updates)}"
    )

    print(
        f"New updates       : {new_count}"
    )

    print(
        f"Important updates : {important_count}"
    )

    print("=" * 65)

    print(
        "\n✅ Tracker completed successfully."
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    run_tracker()