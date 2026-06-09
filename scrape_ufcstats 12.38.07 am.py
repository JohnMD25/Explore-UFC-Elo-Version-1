"""
Minimal ufcstats.com scraper.
Walks the alphabetical fighter index and pulls career stats.
Note: ufcstats.com does not publish rankings; merge rankings
from another source (ufc.com / Wikipedia) into the 'rank' column.
"""
import string
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE = "http://ufcstats.com/statistics/fighters"
HEADERS = {"User-Agent": "Mozilla/5.0 (research script)"}


def fetch(url: str) -> BeautifulSoup:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return BeautifulSoup(r.text, "lxml")


def parse_index_page(letter: str) -> list[dict]:
    soup = fetch(f"{BASE}?char={letter}&page=all")
    rows = soup.select("tr.b-statistics__table-row")
    fighters = []
    for row in rows:
        cells = row.select("td")
        if len(cells) < 11:
            continue
        link = row.select_one("a")
        if not link:
            continue
        fighters.append({
            "name": link.get_text(strip=True),
            "profile_url": link["href"],
            "weight_class": cells[5].get_text(strip=True),
            "wins": int(cells[7].get_text(strip=True) or 0),
            "losses": int(cells[8].get_text(strip=True) or 0),
            "draws": int(cells[9].get_text(strip=True) or 0),
        })
    return fighters


def parse_profile(url: str) -> dict:
    soup = fetch(url)
    items = soup.select("li.b-list__box-list-item")
    out = {}
    for li in items:
        title_el = li.select_one("i")
        if not title_el:
            continue
        key = title_el.get_text(strip=True).rstrip(":").lower()
        value = li.get_text(strip=True).replace(title_el.get_text(strip=True), "").strip()
        out[key] = value
    return {
        "SLpM": _num(out.get("slpm")),
        "str_acc": _pct(out.get("str. acc.")),
        "SApM": _num(out.get("sapm")),
        "str_def": _pct(out.get("str. def")),
        "td_avg": _num(out.get("td avg.")),
        "td_acc": _pct(out.get("td acc.")),
        "td_def": _pct(out.get("td def.")),
        "sub_avg": _num(out.get("sub. avg.")),
    }


def _num(v):
    try: return float(v)
    except: return None


def _pct(v):
    if not v: return None
    try: return float(v.replace("%", ""))
    except: return None


def build_dataset(out_path="data/fighters.csv"):
    all_rows = []
    for letter in string.ascii_lowercase:
        print(f"Index page: {letter}")
        try:
            page_fighters = parse_index_page(letter)
        except Exception as e:
            print(f"  index error: {e}")
            continue
        for f in page_fighters:
            try:
                f.update(parse_profile(f["profile_url"]))
            except Exception as e:
                print(f"  profile error {f['name']}: {e}")
            all_rows.append(f)
            time.sleep(0.4)  # be polite
    df = pd.DataFrame(all_rows)
    df["rank"] = pd.NA  # fill later from a rankings source
    import os
    os.makedirs("data", exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows -> {out_path}")


if __name__ == "__main__":
    build_dataset()