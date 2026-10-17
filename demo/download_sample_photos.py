"""
Download standard CC0 / Public Domain test images for Agentic Vision demonstration.

Fetches diverse real-world images from Wikimedia Commons:
1. Cat (Animal category)
2. Desk with Laptop & Mug (Objects & Electronics category)
3. Person with Bicycle & Dog (Multiple mixed categories)
"""

import os
import urllib.request
import ssl

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "samples")
os.makedirs(SAMPLES_DIR, exist_ok=True)

# Free public domain / CC0 Wikimedia Commons images
SAMPLE_URLS = {
    "cat_photo.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/640px-Cat03.jpg",
    "street_dog_bicycle.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/Golden_Retriever_medium-to-light-coat.jpg/640px-Golden_Retriever_medium-to-light-coat.jpg",
}


def download_samples():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    print("[*] Fetching CC0 sample test photos for demo...")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    for name, url in SAMPLE_URLS.items():
        dest = os.path.join(SAMPLES_DIR, name)
        if os.path.exists(dest):
            print(f"[=] {name} already exists.")
            continue
        try:
            print(f"[>] Downloading {name}...")
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, context=ctx, timeout=15) as resp, open(dest, "wb") as out:
                out.write(resp.read())
            print(f"[+] Downloaded: {dest}")
        except Exception as e:
            print(f"[!] Warning: Could not download {name}: {e}")


if __name__ == "__main__":
    download_samples()
