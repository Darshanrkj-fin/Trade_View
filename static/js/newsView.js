import { postJson } from "./api.js";
import { buildReportTitle, downloadCsv, downloadJson } from "./exportUtils.js";
import { renderBarChart } from "./plotlyHelpers.js";

function renderRecommendationTable(rows) {
  if (!rows.length) {
    return `<div class="empty-state">No recommendations available.</div>`;
  }

  const body = rows.map((row) => `
    <tr>
      <td>${row.ticker}</td>
      <td><span class="tag ${String(row.recommendation || "").toLowerCase()}">${row.recommendation}</span></td>
      <td>${row.impact_score ?? "N/A"}</td>
      <td>${row.news_sentiment ?? "N/A"}</td>
      <td>${row.pred_change_pct ?? "N/A"}</td>
      <td>${row.composite_score ?? "N/A"}</td>
      <td>${row.reason_summary ?? "No explanation available."}</td>
    </tr>
  `).join("");

  return `
    <table class="table">
      <thead>
        <tr>
          <th>Ticker</th>
          <th>Signal</th>
          <th>Impact</th>
          <th>Sentiment</th>
          <th>Forecast %</th>
          <th>Score</th>
          <th>Why</th>
        </tr>
      </thead>
      <tbody>${body}</tbody>
    </table>
  `;
}

function renderNewsSnapshot(data, hosts) {
  const { summaryHost, tableHost, feedHost, performanceHost, performanceChartHost, chartHost, statusHost } = hosts;

  summaryHost.innerHTML = `
    <span class="summary-pill">Articles ${data.summary_counts.articles}</span>
    <span class="summary-pill">Stocks ${data.summary_counts.stocks_found}</span>
    <span class="summary-pill">BUY ${data.summary_counts.buy}</span>
    <span class="summary-pill">HOLD ${data.summary_counts.hold}</span>
    <span class="summary-pill">AVOID ${data.summary_counts.avoid}</span>
  `;

  tableHost.innerHTML = renderRecommendationTable(data.recommendations || []);
  feedHost.innerHTML = (data.articles || []).slice(0, 8).map((article) => `
    <article class="feed-card">
      <h3>${article.headline || "Untitled article"}</h3>
      <p>${article.source || "Unknown source"}</p>
      <p>${article.sentiment || article.text || "No extra detail available."}</p>
    </article>
  `).join("");

  const performance = data.performance_summary || {};
  performanceHost.innerHTML = [
    `
      <article class="signal-card">
        <div class="signal-label">Backtest Coverage</div>
        <div class="signal-value">${performance.coverage ?? 0}</div>
        <div class="mini-label">Recommendations with technical backtest context</div>
      </article>
    `,
    `
      <article class="signal-card">
        <div class="signal-label">BUY Signals</div>
        <div class="signal-value">${performance.buy_avg_backtest_profit ?? "N/A"}%</div>
        <div class="mini-label">Accuracy ${performance.buy_avg_accuracy ?? "N/A"}% | Buy & Hold ${performance.buy_avg_buy_hold ?? "N/A"}%</div>
      </article>
    `,
    `
      <article class="signal-card">
        <div class="signal-label">HOLD Signals</div>
        <div class="signal-value">${performance.hold_avg_backtest_profit ?? "N/A"}%</div>
        <div class="mini-label">Accuracy ${performance.hold_avg_accuracy ?? "N/A"}% | Buy & Hold ${performance.hold_avg_buy_hold ?? "N/A"}%</div>
      </article>
    `,
    `
      <article class="signal-card">
        <div class="signal-label">AVOID Signals</div>
        <div class="signal-value">${performance.avoid_avg_backtest_profit ?? "N/A"}%</div>
        <div class="mini-label">Accuracy ${performance.avoid_avg_accuracy ?? "N/A"}% | Buy & Hold ${performance.avoid_avg_buy_hold ?? "N/A"}%</div>
      </article>
    `,
  ].join("");

  statusHost.textContent = `Fetched ${data.fetched_at || "recently"} | coverage ${data.explanation_summary?.coverage || "unknown"}`;
  if (data.degradation?.is_partial) {
    statusHost.textContent += ` | degraded: ${(data.degradation.reasons || []).join(", ")}`;
  }
  if (data.explanation_summary?.top_composite_score !== undefined && data.explanation_summary?.top_composite_score !== null) {
    statusHost.textContent += ` | top score ${Number(data.explanation_summary.top_composite_score).toFixed(1)}`;
  }
  if (data.explanation_summary?.top_reason) {
    statusHost.textContent += ` | ${data.explanation_summary.top_reason}`;
  }

  const topRows = (data.recommendations || []).slice(0, 8);
  renderBarChart(chartHost, [{
    x: topRows.map((row) => row.ticker),
    y: topRows.map((row) => row.impact_score ?? 0),
    type: "bar",
    marker: { color: topRows.map((row) => {
      if (row.recommendation === "BUY") return "#78ffb2";
      if (row.recommendation === "AVOID") return "#ff667d";
      return "#ffcc6b";
    }) },
  }], {
    title: "Recommendation Spectrum",
  });

  renderBarChart(performanceChartHost, [
    {
      x: ["BUY", "HOLD", "AVOID"],
      y: [
        Number(performance.buy_avg_accuracy || 0),
        Number(performance.hold_avg_accuracy || 0),
        Number(performance.avoid_avg_accuracy || 0),
      ],
      type: "bar",
      name: "Accuracy",
      marker: { color: ["#78ffb2", "#ffcc6b", "#ff667d"] },
    },
    {
      x: ["BUY", "HOLD", "AVOID"],
      y: [
        Number(performance.buy_avg_backtest_profit || 0),
        Number(performance.hold_avg_backtest_profit || 0),
        Number(performance.avoid_avg_backtest_profit || 0),
      ],
      type: "bar",
      name: "Strategy Return",
      marker: { color: ["#56c7ff", "#5fb3ff", "#a98cff"] },
    },
  ], {
    title: "Recommendation Signal Performance",
    yaxis: { title: "Percent" },
  });
}

export function initNewsView(log) {
  const scanBtn = document.getElementById("news-scan-btn");
  const summaryHost = document.getElementById("news-summary");
  const tableHost = document.getElementById("news-table");
  const feedHost = document.getElementById("news-feed");
  const performanceHost = document.getElementById("news-performance");
  const performanceChartHost = document.getElementById("news-performance-chart");
  const chartHost = document.getElementById("news-chart");
  const statusHost = document.getElementById("news-status");
  const exportJsonBtn = document.getElementById("news-export-json-btn");
  const exportCsvBtn = document.getElementById("news-export-csv-btn");
  let latestNews = null;
  const hosts = { summaryHost, tableHost, feedHost, performanceHost, performanceChartHost, chartHost, statusHost };

  exportJsonBtn.addEventListener("click", () => {
    if (!latestNews) {
      log("No news snapshot available to export yet.");
      return;
    }
    downloadJson("news-recommendations", latestNews);
    postJson("/api/reports", {
      type: "news",
      title: buildReportTitle("News Recommendations"),
      payload: latestNews,
      metadata: { recommendations: latestNews.summary_counts?.recommendations, articles: latestNews.summary_counts?.articles },
    }).then(() => {
      window.dispatchEvent(new CustomEvent("reports:changed"));
    }).catch(() => {});
    log("Exported news recommendation snapshot.");
  });

  exportCsvBtn.addEventListener("click", () => {
    if (!latestNews?.recommendations?.length) {
      log("No recommendation rows available for CSV export yet.");
      return;
    }
    downloadCsv("news-recommendations", latestNews.recommendations);
    postJson("/api/reports", {
      type: "news_csv",
      title: buildReportTitle("News Recommendations CSV"),
      payload: { ...latestNews, export_format: "csv" },
      metadata: { recommendations: latestNews.summary_counts?.recommendations, format: "csv" },
    }).then(() => {
      window.dispatchEvent(new CustomEvent("reports:changed"));
    }).catch(() => {});
    log("Exported news recommendation board as CSV.");
  });

  async function runNewsScan() {
    summaryHost.innerHTML = "";
    tableHost.innerHTML = `<div class="empty-state">Scanning financial news sources...</div>`;
    feedHost.innerHTML = `<div class="empty-state">Building article feed...</div>`;
    performanceHost.innerHTML = `<div class="empty-state">Benchmarking signal quality...</div>`;
    performanceChartHost.innerHTML = "";
    statusHost.textContent = "Scanning sources and assembling recommendation context...";
    log("Scanning news sources and generating recommendations...");

    const data = await postJson("/api/news-recommendations", {});
    latestNews = data;
    renderNewsSnapshot(data, hosts);
    log("News recommendation pipeline completed.");
  }

  scanBtn.addEventListener("click", async () => {
    try {
      await runNewsScan();
    } catch (error) {
      latestNews = null;
      tableHost.innerHTML = `<div class="empty-state">${error.message}</div>`;
      feedHost.innerHTML = "";
      performanceHost.innerHTML = "";
      performanceChartHost.innerHTML = "";
      chartHost.innerHTML = `<div class="empty-state">${error.message}</div>`;
      statusHost.textContent = "News data degraded or unavailable.";
      log(`News scan failed: ${error.message}`);
    }
  });

  return {
    loadSnapshot(snapshot) {
      latestNews = snapshot;
      renderNewsSnapshot(snapshot, hosts);
      log("Loaded saved news snapshot.");
    },
    runDemo() {
      return runNewsScan();
    },
  };
}
