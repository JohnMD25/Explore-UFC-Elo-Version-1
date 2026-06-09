# 🥊 Explore UFC Elo

Live, auto-updating Streamlit app that ranks every UFC fighter by peak Elo rating, with searchable profiles, head-to-head bout finder, fight network graph, and career landscape scatter.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/JohnMD25/Explore-UFC-Elo)
![Refresh](https://github.com/JohnMD25/Explore-UFC-Elo/actions/workflows/refresh.yml/badge.svg)

## Features

- **Top 10 Overall** — peak Elo leaderboard plus biggest single-fight upsets
- **Top 10 by Weight Class** — same idea, sliced by division
- **Fighter Search** — Tale of the Tape, Wikipedia profile, UFC.com-style stats dashboard, full bout history, interactive Elo trajectory
- **Fight Finder** — head-to-head between any two fighters who actually met
- **Fight Network** — force-directed graph of the top N fighters connected by their bouts
- **Career Landscape** — scatter of UFC fights vs peak Elo

## Data pipeline

1. UFC fight stats are scraped daily from [Greco1899/scrape_ufc_stats](https://github.com/Greco1899/scrape_ufc_stats).
2. A GitHub Action pulls the four upstream CSVs and rebuilds `Fighter_Profiles.csv` by scraping Wikipedia for the top 250 fighters by peak Elo (with on-disk caching).
3. Updated CSVs are committed back to `main`.
4. Streamlit Cloud detects the commit and auto-redeploys.

## Local development
