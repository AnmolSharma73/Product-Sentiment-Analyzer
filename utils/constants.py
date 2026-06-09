import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Model Configuration ---
USE_BERT = os.getenv("USE_BERT", "false").lower() == "true"
BERT_MODEL_NAME = "yangheng/deberta-v3-base-absa-v1.1"

# --- Analysis Configuration ---
SENTIMENT_LABELS = ["negative", "neutral", "positive"]

# Aspect definition mapping each aspect to a list of keywords
ASPECT_KEYWORDS = {
    "battery": ["battery", "charge", "charging", "battery life", "power", "mah"],
    "camera": ["camera", "photo", "picture", "image", "lens", "zoom", "selfie", "megapixel"],
    "display": ["display", "screen", "resolution", "brightness", "oled", "amoled", "lcd"],
    "performance": ["performance", "speed", "fast", "slow", "lag", "smooth", "processor", "ram", "heating", "thermal"],
    "price": ["price", "value", "cost", "worth", "expensive", "cheap", "affordable", "overpriced"],
    "build": ["build", "design", "quality", "material", "plastic", "metal", "premium", "durability", "weight"],
    "software": ["software", "ui", "ux", "interface", "update", "bug", "os", "android", "ios", "feature"],
    "delivery": ["delivery", "shipping", "packaging", "box", "arrived", "days", "courier"]
}

# Keywords to identify the review text column in tabular data
TEXT_COLUMN_KEYWORDS = [
    "review", "text", "comment", "feedback", 
    "description", "body", "content", "opinion"
]

# Negation words to keep during preprocessing
NEGATION_WORDS = {"not", "never", "no", "without", "hardly", "none", "neither", "nor", "cannot", "n't", "barely", "scarcely"}

if __name__ == "__main__":
    print("Constants Loaded:")
    print(f"USE_BERT: {USE_BERT}")
    print(f"Aspects tracked: {list(ASPECT_KEYWORDS.keys())}")
