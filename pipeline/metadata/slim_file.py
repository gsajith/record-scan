#!/usr/bin/env python3
import json
import sys
import time
from pathlib import Path
from tqdm import tqdm  # pip install tqdm

def get_artist_ids(obj):
    ac = obj.get("artist-credit", [])
    if not ac:
        return None

    ids = []
    for credit in ac:
        artist = credit.get("artist")
        if artist and "id" in artist:
            ids.append(artist["id"])
    return ids

def extract_artist_info(obj):
    result = {}
    ac = obj.get("artist-credit", [])
    if not ac:
        return result

    for entry in ac:
        artist = entry.get("artist")
        if not artist:
            continue

        artist_id = artist.get("id")
        if not artist_id:
            continue

        aliases = [
            alias["name"] for alias in artist.get("aliases", [])
            if alias.get("name")
        ]

        result[artist_id] = {
            "name": artist.get("name", ""),
            "aliases": aliases,
            "disambiguation": artist.get("disambiguation", "")
        }

    return result

def get_aliases(obj):
    aliases = obj.get("aliases", [])
    if not aliases:
        return None
    return [alias["name"] for alias in aliases]

def filter_dict(obj):
    return {key: value for key, value in obj.items() if value is not None and value != ""}

def clean_dict(obj):
    # Clean the dict
    cleaned = {}
    for artist_id, info in obj.items():
        new_info = {k: v for k, v in info.items()
                    if not (k == "disambiguation" and v == "")
                    and not (k == "aliases" and v == [])}
        cleaned[artist_id] = new_info
    return cleaned


def slim_record(obj):
    art_info = obj.get("cover-art-archive", {})
    if not art_info.get("front") and not art_info.get("artwork"):
        return None

    media_list = obj.get("media", [])
    media_format = media_list[0].get("format") if media_list else None

    release = {
        "title": obj.get("title"),
        "id": obj.get("id"),
        "artist_ids": get_artist_ids(obj),
        "asin": obj.get("asin"),
        "disambiguation": obj.get("disambiguation"),
        "format": media_format,
        "aliases": get_aliases(obj)
    }
    artists = extract_artist_info(obj)

    return {
        "release": filter_dict(release),
        "artists": clean_dict(artists)
    }

def merge_and_check_conflicts(dict1, dict2):
    """
    Merge dict2 into dict1 in place.
    Prints conflicts if the same key exists with different values.
    Returns dict1 (mutated).
    """
    for key, value2 in dict2.items():
        if key in dict1:
            value1 = dict1[key]
            if value1 != value2:
                print(f"Conflict detected for key '{key}': "
                      f"'{value1}' (dict1) vs '{value2}' (dict2)")
        # Always overwrite with dict2’s value (can change strategy if needed)
        dict1[key] = value2
    return dict1

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {Path(sys.argv[0]).name} <input.jsonl> <num lines>")
        sys.exit(1)

    infile = Path(sys.argv[1])
    num_lines = sys.argv[2]
    outfile = infile.with_name(infile.stem + "_slim_" + num_lines + ".json")
    artists_outfile = infile.with_name(infile.stem + "_artists_slim_" + num_lines + ".json")

    slimmed_list = []
    artists = {}

    start_time = time.time()  # start timing

    # Count total lines for tqdm if you want full progress bar (optional)
    # total_lines = sum(1 for _ in open(infile, 'r', encoding='utf-8'))

    with infile.open("r", encoding="utf-8") as fin:
        for line in tqdm(fin, unit=" lines", desc="Processing lines"):
            if len(slimmed_list) >= int(num_lines):
                break;
            try:
                obj = json.loads(line)
                slim = slim_record(obj)
                if slim is not None:
                    slimmed_list.append(slim["release"])
                    artists = merge_and_check_conflicts(artists,slim["artists"])

            except Exception as e:
                print(f"⚠️ Skipping a line due to error: {e}")

    elapsed = time.time() - start_time

    with outfile.open("w", encoding="utf-8") as f:
        json.dump(slimmed_list, f, ensure_ascii=False, indent=2)

    with artists_outfile.open("w", encoding="utf-8") as f:
        json.dump(artists, f, ensure_ascii=False, indent=2)

    print(f"\n⏱ Time elapsed: {elapsed:.2f} seconds")
    print(f"✅ Wrote {len(slimmed_list)} slimmed records to {outfile}")

if __name__ == "__main__":
    main()
