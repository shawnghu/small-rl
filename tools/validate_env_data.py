"""Validate data files for RL environments.

Checks schema, uniqueness, and distribution balance.
"""

import json
import os
import sys
from collections import Counter


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

VALID_COLORS = {"red", "orange", "yellow", "green", "blue", "purple", "pink",
                "brown", "black", "white", "gray", "silver", "gold"}
VALID_CATEGORIES = {"animal", "plant", "tool", "food", "vehicle", "furniture",
                    "clothing", "instrument", "toy", "electronics", "building", "nature"}
VALID_CONTINENTS = {"Europe", "Asia", "Africa", "Americas", "Oceania"}


def validate_objects():
    """Validate data/objects.jsonl."""
    path = os.path.join(DATA_DIR, "objects.jsonl")
    if not os.path.exists(path):
        print(f"SKIP: {path} does not exist")
        return False

    with open(path) as f:
        objects = [json.loads(line) for line in f if line.strip()]

    errors = []
    names = set()
    colors = Counter()
    categories = Counter()

    for i, obj in enumerate(objects):
        # Required fields
        for field in ["name", "color", "category", "size_cm", "found_in_nature"]:
            if field not in obj:
                errors.append(f"Line {i+1}: missing field '{field}'")

        name = obj.get("name", "")
        if " " in name or "-" in name:
            errors.append(f"Line {i+1}: name '{name}' contains spaces or hyphens")
        if name in names:
            errors.append(f"Line {i+1}: duplicate name '{name}'")
        names.add(name)

        color = obj.get("color", "")
        if color not in VALID_COLORS:
            errors.append(f"Line {i+1}: invalid color '{color}'")
        colors[color] += 1

        category = obj.get("category", "")
        if category not in VALID_CATEGORIES:
            errors.append(f"Line {i+1}: invalid category '{category}'")
        categories[category] += 1

        if not isinstance(obj.get("size_cm"), (int, float)):
            errors.append(f"Line {i+1}: size_cm must be numeric")
        if not isinstance(obj.get("found_in_nature"), bool):
            errors.append(f"Line {i+1}: found_in_nature must be boolean")

    print(f"objects.jsonl: {len(objects)} entries, {len(names)} unique names")
    print(f"  Colors: {dict(colors.most_common())}")
    print(f"  Categories: {dict(categories.most_common())}")

    if errors:
        for e in errors[:20]:
            print(f"  ERROR: {e}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more errors")
        return False
    print("  OK")
    return True


def validate_cities():
    """Validate data/cities.jsonl."""
    path = os.path.join(DATA_DIR, "cities.jsonl")
    if not os.path.exists(path):
        print(f"SKIP: {path} does not exist")
        return False

    with open(path) as f:
        cities = [json.loads(line) for line in f if line.strip()]

    errors = []
    city_names = set()
    continents = Counter()
    countries = Counter()

    for i, city in enumerate(cities):
        for field in ["city", "country", "continent"]:
            if field not in city:
                errors.append(f"Line {i+1}: missing field '{field}'")

        name = city.get("city", "")
        if " " in name:
            errors.append(f"Line {i+1}: city '{name}' contains spaces")
        if name in city_names:
            errors.append(f"Line {i+1}: duplicate city '{name}'")
        city_names.add(name)

        country = city.get("country", "")
        if " " in country:
            errors.append(f"Line {i+1}: country '{country}' contains spaces")
        countries[country] += 1

        continent = city.get("continent", "")
        if continent not in VALID_CONTINENTS:
            errors.append(f"Line {i+1}: invalid continent '{continent}'")
        continents[continent] += 1

    print(f"cities.jsonl: {len(cities)} entries, {len(city_names)} unique cities")
    print(f"  Continents: {dict(continents.most_common())}")
    print(f"  Top countries: {dict(countries.most_common(10))}")

    if errors:
        for e in errors[:20]:
            print(f"  ERROR: {e}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more errors")
        return False
    print("  OK")
    return True


def validate_nouns():
    """Validate data/nouns.txt."""
    path = os.path.join(DATA_DIR, "nouns.txt")
    if not os.path.exists(path):
        print(f"SKIP: {path} does not exist")
        return False

    with open(path) as f:
        nouns = [line.strip().lower() for line in f if line.strip()]

    errors = []
    seen = set()
    for i, noun in enumerate(nouns):
        if " " in noun or "-" in noun:
            errors.append(f"Line {i+1}: '{noun}' contains spaces or hyphens")
        if noun in seen:
            errors.append(f"Line {i+1}: duplicate '{noun}'")
        seen.add(noun)

    print(f"nouns.txt: {len(nouns)} entries, {len(seen)} unique")
    if errors:
        for e in errors[:20]:
            print(f"  ERROR: {e}")
        return False
    print("  OK")
    return True


def validate_translations():
    """Validate data/en_es_translations.jsonl."""
    path = os.path.join(DATA_DIR, "en_es_translations.jsonl")
    if not os.path.exists(path):
        print(f"SKIP: {path} does not exist")
        return False

    with open(path) as f:
        translations = [json.loads(line) for line in f if line.strip()]

    errors = []
    english_words = set()
    freq_classes = Counter()

    for i, t in enumerate(translations):
        for field in ["english", "spanish", "frequency_class"]:
            if field not in t:
                errors.append(f"Line {i+1}: missing field '{field}'")

        english = t.get("english", "")
        if " " in english:
            errors.append(f"Line {i+1}: english '{english}' contains spaces")
        if english in english_words:
            errors.append(f"Line {i+1}: duplicate english word '{english}'")
        english_words.add(english)

        spanish = t.get("spanish", "")
        if " " in spanish:
            errors.append(f"Line {i+1}: spanish '{spanish}' contains spaces")

        fc = t.get("frequency_class", "")
        if fc not in ("common", "rare"):
            errors.append(f"Line {i+1}: invalid frequency_class '{fc}'")
        freq_classes[fc] += 1

        alt = t.get("alt_spanish", [])
        if not isinstance(alt, list):
            errors.append(f"Line {i+1}: alt_spanish must be a list")

    print(f"en_es_translations.jsonl: {len(translations)} entries, {len(english_words)} unique")
    print(f"  Frequency classes: {dict(freq_classes)}")

    if errors:
        for e in errors[:20]:
            print(f"  ERROR: {e}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more errors")
        return False
    print("  OK")
    return True


if __name__ == "__main__":
    all_ok = True
    all_ok &= validate_objects()
    all_ok &= validate_cities()
    all_ok &= validate_nouns()
    all_ok &= validate_translations()

    if not all_ok:
        print("\nSome validations failed!")
        sys.exit(1)
    else:
        print("\nAll validations passed!")
