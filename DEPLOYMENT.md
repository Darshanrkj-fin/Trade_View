# Deployment Guide

## Recommended Hosting Models

### Option 1: Current architecture

Use Render for the full stack. This repository is a Flask application that serves both the frontend shell and the backend APIs.

### Option 2: Netlify + backend split

If you specifically want Netlify, use it only for a separated static frontend and host the Flask backend elsewhere such as Render or Railway.

This split is not implemented in the current codebase yet. It would require extracting the frontend into a standalone static app that calls the backend over HTTP.

## Render

TradeWise AI Terminal is configured for Render with the included [render.yaml](/Users/R%20DARSHAN/PycharmProjects/Trade_view/render.yaml).

### Build settings

- Runtime: Python
- Build command: `pip install -r requirements.txt`
- Start command: `waitress-serve --host=0.0.0.0 --port=$PORT app:app`
- Health check: `/health`

### Environment variables

- `FLASK_DEBUG=0`
- `TRADEWISE_MARKET_CACHE_TTL=60`
- `TRADEWISE_ANALYSIS_CACHE_TTL=180`
- `TRADEWISE_NEWS_CACHE_TTL=120`

Optional:

- `TRADEWISE_DIAGNOSTICS_WINDOW`
- `TRADEWISE_HOST` if your host requires an explicit override
- `PORT` is typically injected by Render

## Production notes

- `/health` is the correct liveness endpoint for hosting platforms.
- `/ready` is useful for manual checks and confirms configuration is loadable.
- Netlify is not the correct full-stack host for the app in its current Flask form.
- TensorFlow is optional. Hosted deployments may use the baseline sklearn forecast path.
- News scraping can degrade partially if external page structures change or rate limits apply.
- The `reports/` directory is local filesystem storage for saved snapshots in v1 and is not multi-instance durable.
