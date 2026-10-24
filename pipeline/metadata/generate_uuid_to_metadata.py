# make_uuid_metadata.py
import json

with open("slim_releases_with_covers.json") as f:
    covers = json.load(f)

uuid_to_meta = {}

for c in covers:
    uuid_to_meta[c["id"]] = c

with open("uuid_to_metadata.json", "w") as f:
    json.dump(uuid_to_meta, f, indent=2)
