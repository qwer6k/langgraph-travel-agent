from decimal import Decimal, InvalidOperation
import re
import json
import urllib.request
from typing import Optional, Dict

# Fallback static rates: 1 USD = X unit of currency
FALLBACK_RATES: Dict[str, Decimal] = {
    "USD": Decimal("1"),
    "CNY": Decimal("7.0"),
    "EUR": Decimal("0.93"),
    "JPY": Decimal("145.0"),
    "GBP": Decimal("0.81"),
    "AUD": Decimal("1.55"),
}


def _fetch_rates_base_usd() -> Dict[str, Decimal]:
    """
    Try to fetch live FX rates with base=USD from exchangerate.host.
    Return mapping currency -> rate (units per 1 USD). On failure return FALLBACK_RATES.
    """
    url = "https://api.exchangerate.host/latest?base=USD"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.load(resp)
            rates = data.get("rates", {})
            out: Dict[str, Decimal] = {}
            for k, v in rates.items():
                try:
                    out[k.upper()] = Decimal(str(v))
                except (InvalidOperation, TypeError):
                    continue
            # Ensure USD present
            out["USD"] = Decimal("1")
            return out
    except Exception:
        return FALLBACK_RATES.copy()


def parse_price_string(s: Optional[str]) -> Optional[tuple[Decimal, str]]:
    """Parse a price string, returning (amount, currency_code).

    Examples: "$1,200 USD" -> (1200, "USD"), "1200 CNY" -> (1200, "CNY"),
    "¥1200" -> (1200, "CNY") by default for ¥.
    """
    if not s:
        return None
    text = str(s).strip()
    # currency symbol mapping
    sym_map = {
        "$": "USD",
        "€": "EUR",
        "£": "GBP",
        "¥": "CNY",
        "元": "CNY",
    }

    # find currency code like USD/EUR/JPY
    m_code = re.search(r"([A-Z]{3})\b", text)
    code = None
    if m_code:
        code = m_code.group(1).upper()

    # find symbol
    for sym, ccy in sym_map.items():
        if sym in text:
            code = code or ccy
            break

    # find number
    m_num = re.search(r"([\d,]+(?:\.\d+)?)", text.replace("\u00A0", " "))
    if not m_num:
        return None
    num = m_num.group(1).replace(",", "")
    try:
        amt = Decimal(num)
    except InvalidOperation:
        return None

    if not code:
        # default to USD if currency not present
        code = "USD"

    return amt, code


def to_usd(amount: Decimal, ccy: str, rates: Optional[Dict[str, Decimal]] = None) -> Optional[Decimal]:
    """Convert amount in currency `ccy` to USD using provided rates mapping (units per 1 USD).

    If rates is None, attempt to fetch live rates.
    """
    if rates is None:
        rates = _fetch_rates_base_usd()

    c = ccy.upper()
    if c == "USD":
        return amount
    rate = rates.get(c)
    if rate is None or rate == 0:
        return None
    # rates: units per 1 USD, so USD = amount / rate
    try:
        return (amount / rate).quantize(Decimal("0.01"))
    except Exception:
        return None


def parse_price_to_usd(s: Optional[str], rates: Optional[Dict[str, Decimal]] = None) -> Optional[Decimal]:
    parsed = parse_price_string(s)
    if not parsed:
        return None
    amt, ccy = parsed
    return to_usd(amt, ccy, rates)
