"""License normalization and classification.

Different sources express licenses differently:
  - iNaturalist: short codes  e.g. "cc-by-nc", "cc0", None (=all rights reserved)
  - GBIF media:  full URLs    e.g. "http://creativecommons.org/licenses/by-nc/4.0/"
                 or SPDX-ish  e.g. "CC_BY_NC_4_0", "CC0_1_0"

We normalize everything to a canonical tag so filtering and reporting are uniform.
Classification is used for *reporting* (the provenance statement on the dataset
card) — it is no longer a hard gate, because datasets are private and feed a model
rather than being redistributed. Opt-in license filtering still uses these tags.
"""
from __future__ import annotations

import re

# Canonical tags
CC0 = "CC0"
CC_BY = "CC-BY"
CC_BY_SA = "CC-BY-SA"
CC_BY_NC = "CC-BY-NC"
CC_BY_NC_SA = "CC-BY-NC-SA"
CC_BY_ND = "CC-BY-ND"
CC_BY_NC_ND = "CC-BY-NC-ND"
PUBLIC_DOMAIN = "PUBLIC-DOMAIN"
ALL_RIGHTS_RESERVED = "ALL-RIGHTS-RESERVED"
UNKNOWN = "UNKNOWN"

_URLS = {
    CC0: "https://creativecommons.org/publicdomain/zero/1.0/",
    PUBLIC_DOMAIN: "https://creativecommons.org/publicdomain/mark/1.0/",
    CC_BY: "https://creativecommons.org/licenses/by/4.0/",
    CC_BY_SA: "https://creativecommons.org/licenses/by-sa/4.0/",
    CC_BY_NC: "https://creativecommons.org/licenses/by-nc/4.0/",
    CC_BY_NC_SA: "https://creativecommons.org/licenses/by-nc-sa/4.0/",
    CC_BY_ND: "https://creativecommons.org/licenses/by-nd/4.0/",
    CC_BY_NC_ND: "https://creativecommons.org/licenses/by-nc-nd/4.0/",
}

# Order matters: check the most specific (longest) variants first.
_PATTERNS = [
    (CC_BY_NC_SA, r"by[-_ ]?nc[-_ ]?sa"),
    (CC_BY_NC_ND, r"by[-_ ]?nc[-_ ]?nd"),
    (CC_BY_NC, r"by[-_ ]?nc"),
    (CC_BY_SA, r"by[-_ ]?sa"),
    (CC_BY_ND, r"by[-_ ]?nd"),
    (CC_BY, r"\bby\b|licenses/by"),
    (CC0, r"cc0|zero|publicdomain/zero"),
    (PUBLIC_DOMAIN, r"publicdomain|pdm|public[-_ ]?domain"),
]

# Tags whose images may be redistributed publicly (informational only now).
REDISTRIBUTABLE = {CC0, PUBLIC_DOMAIN, CC_BY, CC_BY_SA}
# Tags that permit commercial use.
COMMERCIAL_OK = {CC0, PUBLIC_DOMAIN, CC_BY, CC_BY_SA, CC_BY_ND}


def normalize(raw: str | None) -> str:
    """Map any source license string to a canonical tag."""
    if raw is None or str(raw).strip() == "":
        return ALL_RIGHTS_RESERVED  # iNat: a missing code means "no license" = all rights reserved
    s = str(raw).strip().lower()
    if s in {"none", "null", "all rights reserved", "c"}:
        return ALL_RIGHTS_RESERVED
    for tag, pat in _PATTERNS:
        if re.search(pat, s):
            return tag
    return UNKNOWN


def license_url(tag: str) -> str | None:
    return _URLS.get(tag)


def is_redistributable(tag: str) -> bool:
    return tag in REDISTRIBUTABLE


def is_commercial_ok(tag: str) -> bool:
    return tag in COMMERCIAL_OK


def requires_attribution(tag: str) -> bool:
    return tag not in {CC0, PUBLIC_DOMAIN}
