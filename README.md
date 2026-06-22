# Support Ticket Triage

An end-to-end ML pipeline for automatically triaging customer support tickets.

## What it does
- Classifies ticket priority (Low / Medium / High / Critical)
- Predicts time to resolution
- Segments tickets into issue clusters

## Dataset
Customer Support Ticket Dataset (Kaggle) — 8,470 records with ticket type, priority, channel, response time, resolution time, and satisfaction rating.

## Setup
```bash
git clone <repo-url>
cd support-ticket-triage
uv sync
```

## Project Structure