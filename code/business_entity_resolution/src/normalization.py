"""
High-performance normalization utilities for business names and addresses.
Amazon ML Challenge 2026: Business Entity Resolution.
"""

import re
from typing import Dict, List, Optional, Set, Tuple


# Fast character translation table replacing common punctuation with spaces
PUNCT_CHARS = ",.-/()[]{}:;\"'!?@#$%^*_+=~`|\\<>"
PUNCT_TABLE = str.maketrans({ch: " " for ch in PUNCT_CHARS})


NULL_SET = {"null", "nan", "none", "na", "unknown"}

# Business Legal Suffixes across US, India, and France
LEGAL_SINGLES = {
    "llc", "inc", "corp", "corporation", "ltd", "limited", "pvt", "private",
    "co", "company", "llp", "gmbh", "sarl", "sas", "sa", "sci", "eurl",
    "incorporated", "enterprises", "holdings"
}

LEGAL_PAIRS = {
    ("private", "limited"),
    ("pvt", "ltd"),
    ("pvt", "limited"),
    ("private", "ltd"),
    ("limited", "liability"),
    ("public", "limited"),
}

# Honorific prefixes (especially common in India)
HONORIFIC_PREFIXES = {"smt", "shri", "shree", "mr", "mrs", "ms", "dr", "messrs"}

# Common address abbreviations expansion dictionary
ADDR_ABBREVIATIONS: Dict[str, str] = {
    # Road types
    "st": "street",
    "str": "street",
    "rd": "road",
    "ave": "avenue",
    "av": "avenue",
    "dr": "drive",
    "blvd": "boulevard",
    "bld": "boulevard",
    "bd": "boulevard",
    "ln": "lane",
    "ct": "court",
    "pl": "place",
    "cir": "circle",
    "ter": "terrace",
    "terr": "terrace",
    "pkwy": "parkway",
    "hwy": "highway",
    "sq": "square",
    "expwy": "expressway",
    "fwy": "freeway",
    "pk": "park",
    "pkg": "park",
    # French road types
    "r": "rue",
    "imp": "impasse",
    "all": "allee",
    # Unit / Sub-address designators
    "apt": "apartment",
    "ste": "suite",
    "bldg": "building",
    "fl": "floor",
    "rm": "room",
    "dept": "department",
    "no": "number",
    # Landmark / Positional (common in India)
    "opp": "opposite",
    "nr": "near",
    "beside": "near",
    "behind": "behind",
    "bazar": "bazaar",
}

# Postal code regexes
RE_PIN_CODE = re.compile(r"\b([1-9]\d{5})\b")
RE_ZIP_CODE = re.compile(r"\b(\d{5})(?:-\d{4})?\b")
RE_FRENCH_POSTAL = re.compile(r"\b(0[1-9]|[1-8]\d|9[0-8])\d{3}\b")


def clean_text(text: Optional[str]) -> str:
    """Fast lowercasing, ampersand normalization, and whitespace cleanup."""
    if text is None or not isinstance(text, str):
        return ""
    text = text.lower().replace("&", " and ")
    tokens = text.translate(PUNCT_TABLE).split()
    tokens = [t for t in tokens if t not in NULL_SET]
    return " ".join(tokens)


def normalize_business_name(name: Optional[str], strip_legal: bool = True) -> str:
    """Normalize business name by removing punctuation, honorific prefixes, and legal suffixes."""
    if name is None or not isinstance(name, str):
        return ""
    cleaned = clean_text(name)
    if not cleaned:
        return ""

    tokens = cleaned.split()
    
    # Strip honorific prefixes
    if len(tokens) >= 2 and tokens[0] in HONORIFIC_PREFIXES:
        tokens = tokens[1:]

    # Strip legal suffixes
    if strip_legal:
        if len(tokens) >= 2 and (tokens[-2], tokens[-1]) in LEGAL_PAIRS:
            tokens = tokens[:-2]
        elif len(tokens) >= 1 and tokens[-1] in LEGAL_SINGLES:
            tokens = tokens[:-1]
            # Check once more if another suffix precedes (e.g. "Services LLC")
            if len(tokens) >= 1 and tokens[-1] in {"services", "co", "corp", "group", "holdings"}:
                tokens = tokens[:-1]

    return " ".join(tokens)


def normalize_address(address: Optional[str], expand_abbr: bool = True) -> str:
    """Normalize address string with lowercase, expanded abbreviations, and cleaned whitespace."""
    if address is None or not isinstance(address, str):
        return ""
    cleaned = clean_text(address)
    if not cleaned:
        return ""

    tokens = cleaned.split()
    if expand_abbr:
        tokens = [ADDR_ABBREVIATIONS.get(t, t) for t in tokens]
    return " ".join(tokens)


def extract_postal_code(address: Optional[str], country: Optional[str] = None) -> Optional[str]:
    """Extract postal code (PIN / ZIP / French Code) based on country or general pattern."""
    if not address or not isinstance(address, str):
        return None

    if country == "India":
        m = RE_PIN_CODE.search(address)
        return m.group(1) if m else None
    elif country == "US":
        m = RE_ZIP_CODE.search(address)
        return m.group(1) if m else None
    elif country == "France":
        m = RE_FRENCH_POSTAL.search(address)
        return m.group(0) if m else None

    # Open set fallback:
    m_pin = RE_PIN_CODE.search(address)
    if m_pin:
        return m_pin.group(1)
    m_zip = RE_ZIP_CODE.search(address)
    if m_zip:
        return m_zip.group(1)
    return None


def extract_street_numbers(address: Optional[str]) -> List[str]:
    """Extract street/building numerical identifiers from an address."""
    if not address or not isinstance(address, str):
        return []
    cleaned = address.lower().translate(PUNCT_TABLE)
    numbers = []
    for tok in cleaned.split():
        if tok.isdigit():
            numbers.append(tok)
        elif len(tok) > 1 and tok[:-1].isdigit() and tok[-1].isalpha():
            numbers.append(tok)
    return numbers


def compute_soundex(word: str) -> str:
    """Compute standard American Soundex for an English word.
    Returns a 4-character code (letter + 3 digits).
    """
    if not word or not isinstance(word, str):
        return ""
    word = re.sub(r"[^a-zA-Z]", "", word).upper()
    if not word:
        return ""

    first_letter = word[0]
    char_map = {
        "B": "1", "F": "1", "P": "1", "V": "1",
        "C": "2", "G": "2", "J": "2", "K": "2", "Q": "2", "S": "2", "X": "2", "Z": "2",
        "D": "3", "T": "3",
        "L": "4",
        "M": "5", "N": "5",
        "R": "6",
        "A": "", "E": "", "I": "", "O": "", "U": "", "Y": "", "H": "", "W": "",
    }

    encoded = []
    prev_code = char_map.get(first_letter, "")
    for ch in word[1:]:
        code = char_map.get(ch, "")
        if code != "" and code != prev_code:
            encoded.append(code)
            prev_code = code
        elif code == "":
            prev_code = ""

    return (first_letter + "".join(encoded) + "000")[:4]


def extract_name_blocking_keys(name: Optional[str], min_token_len: int = 3) -> Dict[str, any]:
    norm = normalize_business_name(name, strip_legal=True)
    tokens = [t for t in norm.split() if len(t) >= min_token_len]
    first_token = tokens[0] if tokens else ""
    soundex_first = compute_soundex(first_token) if first_token else ""
    clean_no_space = norm.replace(" ", "")
    prefix_4 = clean_no_space[:4] if len(clean_no_space) >= 4 else clean_no_space

    return {
        "norm_name": norm,
        "first_token": first_token,
        "tokens": set(tokens),
        "soundex_first": soundex_first,
        "prefix_4": prefix_4,
    }


def extract_address_blocking_keys(address: Optional[str], country: Optional[str] = None) -> Dict[str, any]:
    norm = normalize_address(address, expand_abbr=True)
    postal = extract_postal_code(address, country)
    numbers = extract_street_numbers(address)
    tokens = {t for t in norm.split() if len(t) >= 3 and not t.isdigit()}

    return {
        "norm_address": norm,
        "postal_code": postal,
        "street_numbers": numbers,
        "addr_tokens": tokens,
    }
