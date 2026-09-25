"""
Unit tests for the normalization module.
"""

import pytest
from src.normalization import (
    clean_text,
    normalize_business_name,
    normalize_address,
    extract_postal_code,
    compute_soundex,
    extract_name_blocking_keys,
    extract_address_blocking_keys,
)


def test_clean_text():
    assert clean_text("  A & B   Corp.  ") == "a and b corp"
    assert clean_text("None") == ""
    assert clean_text("123, High St. (Near SBI)") == "123 high st near sbi"


def test_normalize_business_name():
    # Legal suffix removal
    assert normalize_business_name("Infosys Technologies Pvt Ltd") == "infosys technologies"
    assert normalize_business_name("Infosys Technologies Private Limited") == "infosys technologies"
    assert normalize_business_name("Apple Inc.") == "apple"
    assert normalize_business_name("Amazon Services LLC") == "amazon"
    assert normalize_business_name("TotalEnergies SE SARL") == "totalenergies se"
    assert normalize_business_name("Dahlia Power Reliable Scientific LLC") == "dahlia power reliable scientific"


def test_normalize_address():
    # Abbreviation expansion
    assert normalize_address("123 Main St, Suite 4") == "123 main street suite 4"
    assert normalize_address("456 Park Ave., Apt 2B") == "456 park avenue apartment 2b"
    assert normalize_address("Plot 12, MG Rd, Opp. State Bank") == "plot 12 mg road opposite state bank"
    assert normalize_address("1795 Westchester Drive, High Point, NC") == "1795 westchester drive high point nc"


def test_extract_postal_code():
    # India 6-digit PIN
    assert extract_postal_code("Lake Town Block A, Kolkata 700089", country="India") == "700089"
    # US 5-digit ZIP
    assert extract_postal_code("1000 Fifth Avenue, New York, NY 10028-0198", country="US") == "10028"
    # France 5-digit postal code
    assert extract_postal_code("10 Rue de la Paix, 75002 Paris", country="France") == "75002"


def test_soundex():
    # Classic Soundex equivalences
    assert compute_soundex("Robert") == "R163"
    assert compute_soundex("Rupert") == "R163"
    assert compute_soundex("Smith") == "S530"
    assert compute_soundex("Smythe") == "S530"


def test_extract_blocking_keys():
    name_keys = extract_name_blocking_keys("Dahlia Power Reliable Scientific LLC")
    assert name_keys["first_token"] == "dahlia"
    assert name_keys["soundex_first"] == compute_soundex("dahlia")
    assert "dahlia" in name_keys["tokens"]
    assert "power" in name_keys["tokens"]
    assert name_keys["prefix_4"] == "dahl"

    addr_keys = extract_address_blocking_keys("630 45th Terrace, Kansas City, MO 64111", country="US")
    assert addr_keys["postal_code"] == "64111"
    assert "630" in addr_keys["street_numbers"]
    assert "terrace" in addr_keys["norm_address"]
