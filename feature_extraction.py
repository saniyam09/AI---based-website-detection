"""Utility functions for extracting URL-based phishing indicators.

This module is intentionally shared between model training and live prediction.
Every feature must be computed the same way in both places to keep the ML model
consistent and reliable.
"""

import re
from urllib.parse import urlparse

FEATURE_COLUMNS = [
    "url_length",
    "domain_length",
    "num_dots",
    "num_hyphens",
    "num_special_chars",
    "has_at",
    "has_ip",
    "https",
    "num_digits",
    "num_subdomains",
    "contains_login",
    "contains_verify",
    "contains_account",
    "contains_secure",
    "contains_update",
    "contains_bank",
    "contains_password",
    "contains_confirm",
]

SUSPICIOUS_KEYWORDS = [
    "login",
    "verify",
    "account",
    "secure",
    "update",
    "bank",
    "password",
    "confirm",
]


def _extract_domain(url: str) -> str:
    """Return a clean domain name without scheme or path."""
    parsed = urlparse(url)
    if parsed.netloc:
        domain = parsed.netloc.lower().split(":")[0]
        return domain
    return url.lower().strip("/").split("/")[0]


def _is_ip_address(value: str) -> bool:
    """Check if the domain is actually an IPv4 address."""
    pattern = r"^(?:\d{1,3}\.){3}\d{1,3}$"
    return bool(re.fullmatch(pattern, value))


def extract_features(url: str) -> dict:
    """Extract phishing-related features from a single URL."""
    cleaned_url = str(url).strip()
    if not cleaned_url:
        raise ValueError("URL is empty.")

    if not cleaned_url.lower().startswith(("http://", "https://")):
        cleaned_url = "https://" + cleaned_url if "://" not in cleaned_url else cleaned_url

    parsed = urlparse(cleaned_url)
    full_url = cleaned_url.lower()
    domain = _extract_domain(cleaned_url)
    domain_without_port = domain.split(":")[0]

    num_dots = full_url.count(".")
    num_hyphens = full_url.count("-")
    num_special_chars = sum(
        1 for char in full_url if char in "?_&=%#+/;:~!$,'*[](){}|^\\"
    )
    has_at = 1 if "@" in full_url else 0
    has_ip = 1 if _is_ip_address(domain_without_port) else 0
    https = 1 if full_url.startswith("https://") else 0
    num_digits = sum(1 for ch in full_url if ch.isdigit())

    num_subdomains = 0
    if "." in domain_without_port:
        base_parts = domain_without_port.split(".")
        num_subdomains = max(0, len(base_parts) - 2)

    feature_map = {
        "url_length": len(cleaned_url),
        "domain_length": len(domain_without_port),
        "num_dots": num_dots,
        "num_hyphens": num_hyphens,
        "num_special_chars": num_special_chars,
        "has_at": has_at,
        "has_ip": has_ip,
        "https": https,
        "num_digits": num_digits,
        "num_subdomains": num_subdomains,
    }

    for keyword in SUSPICIOUS_KEYWORDS:
        feature_map[f"contains_{keyword}"] = 1 if keyword in full_url else 0

    return feature_map


def features_as_vector(url: str) -> dict:
    """Return the feature dict in the exact order expected by the models."""
    raw_features = extract_features(url)
    ordered = {}
    for column in FEATURE_COLUMNS:
        ordered[column] = raw_features.get(column, 0)
    return ordered
