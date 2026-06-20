import os

# Load .env similarly to the backend's loader
base = os.path.dirname(__file__)
dotenv = os.path.join(base, ".env")
if os.path.exists(dotenv):
    with open(dotenv, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

# Report masked GOOGLE_API_KEY presence and length
key = os.environ.get("GOOGLE_API_KEY")
if key:
    try:
        length = len(key)
    except Exception:
        length = "?"
    print("GOOGLE_API_KEY: SET")
    print("GOOGLE_API_KEY_LENGTH:", length)
else:
    print("GOOGLE_API_KEY: NOT SET")

# Report GEMINI_CHAT_MODEL value
print("GEMINI_CHAT_MODEL:", os.environ.get("GEMINI_CHAT_MODEL", "(not set)"))
