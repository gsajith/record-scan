# Run from the repo root: python -m pipeline.metadata.generate_uuid_to_metadata
import json

from pipeline import config

with open(config.META_DIR / "slim_releases_with_covers.json") as f:
    covers = json.load(f)

uuid_to_meta = {}

for c in covers:
    uuid_to_meta[c["id"]] = c

with open(config.META_DIR / "uuid_to_metadata.json", "w") as f:
    json.dump(uuid_to_meta, f, indent=2)
