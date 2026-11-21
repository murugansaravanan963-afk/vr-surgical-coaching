"""Print "<file name> <download url>" for every .zip in the given figshare articles."""
import json
import sys
import urllib.request

for article in sys.argv[1:]:
    with urllib.request.urlopen(f"https://api.figshare.com/v2/articles/{article}") as r:
        for f in json.load(r)["files"]:
            if f["name"].endswith(".zip"):
                print(f["name"], f["download_url"])
