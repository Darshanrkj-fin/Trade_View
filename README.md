# TradeWise AI Terminal

TradeWise AI Terminal is a Flask-based market research dashboard for Indian equities. It combines manual stock analysis, AI news recommendations, and portfolio optimization inside a custom HTML/CSS/JavaScript trading terminal UI.

This project is designed as a polished research/demo website and GitHub showcase. It is not a live trading engine or investment-advice platform.

## Live Demo

- Live website: add your deployed URL here after deployment
- Health check: `/health`

## What It Does

- Manual stock analysis with price history, technical indicators, sentiment, forecast output, calibration metrics, and backtests
- AI news recommendation flow with `BUY / HOLD / AVOID` signals and signal-performance summaries
- Portfolio optimization with momentum ranking, Markowitz allocation, CAPM metrics, turnover context, and efficient frontier visuals
- Exportable snapshots plus searchable report history

The screenshot above shows the Flask-based trading terminal UI with the dark dashboard layout, market panels, and portfolio/news analysis views.

## Quick Start

```bash
pip install -r requirements.txt
python app.py
```

## Deployment

This repo currently ships as a single Flask application, so the full stack is prepared for GitHub + Render deployment.

If you want to use Netlify, the recommended pattern is:

- Netlify for a separated static frontend
- Render or Railway for the Flask API backend

The current codebase is not a drop-in full-stack Netlify deployment because the backend is a persistent Python Flask server, not a Netlify Functions app.

- Render manifest: [render.yaml](/Users/R%20DARSHAN/PycharmProjects/Trade_view/render.yaml)
- Process file: [Procfile](/Users/R%20DARSHAN/PycharmProjects/Trade_view/Procfile)
- Deployment guide: [DEPLOYMENT.md](/Users/R%20DARSHAN/PycharmProjects/Trade_view/DEPLOYMENT.md)

Production start command:

```bash
waitress-serve --host=0.0.0.0 --port=$PORT app:app
```

Recommended production environment:

- `FLASK_DEBUG=0`
- `PORT` provided by the host
- optional cache TTL overrides through `TRADEWISE_*`

## Launch Checklist

- Push the latest default branch to GitHub
- For current architecture: create a Render web service from this repository
- For Netlify architecture: split the frontend from Flask APIs, then deploy the static frontend to Netlify and the backend to Render/Railway
- Confirm build command: `pip install -r requirements.txt`
- Confirm start command: `waitress-serve --host=0.0.0.0 --port=$PORT app:app`
- Set `FLASK_DEBUG=0`
- Verify `/health` and `/ready`
- Add the live demo URL above
- Add one deployed screenshot or GIF to this README

## Public Routes

- `GET /`
- `GET /health`
- `GET /ready`
- `GET /api/diagnostics`
- `GET /api/config`
- `GET /api/market-status`
- `POST /api/analyze-stock`
- `POST /api/benchmarks`
- `POST /api/news-recommendations`
- `POST /api/portfolio/optimize`
- `GET /api/reports`
- `GET /api/reports/<id>`
- `POST /api/reports`
- `POST /api/reports/<id>/pin`
- `DELETE /api/reports/<id>`

## Known Limitations

- News scraping may partially fail when external source markup changes
- TensorFlow is optional; hosted deployments may use the baseline sklearn forecast path
- Portfolio ranking is heuristic momentum scoring, not a trained institutional model
- Report history is stored on the local host filesystem in `reports/`
- This is a research and educational tool, not production trading software

## Project Structure

```text
Trade_view/
|-- app.py
|-- terminal_app/
|-- templates/
|-- static/
|-- stock_recommender.py
|-- news_scraper.py
|-- model.py
|-- portfolio_optimizer.py
|-- benchmark.py
|-- Procfile
|-- render.yaml
`-- DEPLOYMENT.md
```
