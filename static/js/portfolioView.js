import { postJson } from "./api.js";
import { buildReportTitle, downloadJson } from "./exportUtils.js";
import { renderBarChart, renderLineChart, renderPieChart } from "./plotlyHelpers.js";

function holdingsTable(holdings) {
  if (!holdings.length) {
    return `<div class="empty-state">No holdings added yet.</div>`;
  }

  return `
    <table class="table">
      <thead>
        <tr>
          <th>Ticker</th>
          <th>Qty</th>
          <th>Buy Price</th>
        </tr>
      </thead>
      <tbody>
        ${holdings.map((item) => `
          <tr>
            <td>${item.ticker}</td>
            <td>${item.qty}</td>
            <td>${item.buy_price}</td>
          </tr>
        `).join("")}
      </tbody>
    </table>
  `;
}

function renderPortfolioSnapshot(data, hosts) {
  const { metricsHost, suggestionsHost, weightsChart, allocationChart, frontierChart, rankingBenchmarkChart } = hosts;
  const metrics = data.metrics || {};
  const summary = data.portfolio_summary || {};
  const rankingValidation = data.ranking_validation || {};
  const topRanked = summary.top_ranked_asset ? `Top ranked ${summary.top_ranked_asset}` : "Ranking available after optimization";
  const betterMethodLabel = summary.ranking_validation_better_method === "current"
    ? "Current"
    : summary.ranking_validation_better_method === "legacy"
      ? "Legacy"
      : "N/A";
  const spreadLabel = summary.ranking_validation_outperformance_spread == null
    ? "No spread available"
    : `${Number(summary.ranking_validation_outperformance_spread).toFixed(2)}% avg return spread`;
  metricsHost.innerHTML = `
    <article class="signal-card">
      <div class="signal-label">Optimal Return</div>
      <div class="signal-value">${(Number(metrics.optimal?.return || 0) * 100).toFixed(2)}%</div>
    </article>
    <article class="signal-card">
      <div class="signal-label">Optimal Volatility</div>
      <div class="signal-value">${(Number(metrics.optimal?.volatility || 0) * 100).toFixed(2)}%</div>
    </article>
    <article class="signal-card">
      <div class="signal-label">Sharpe Ratio</div>
      <div class="signal-value">${Number(metrics.optimal?.sharpe || 0).toFixed(2)}</div>
    </article>
    <article class="signal-card">
      <div class="signal-label">Ranking Method</div>
      <div class="signal-value">${summary.ranking_label || "Momentum Heuristic"}</div>
      <div class="mini-label">${topRanked}</div>
    </article>
    <article class="signal-card">
      <div class="signal-label">Optimizer</div>
      <div class="signal-value">${summary.optimizer_label || "Sharpe optimization"}</div>
      <div class="mini-label">${(summary.assumption_notes || []).join(" ")}</div>
    </article>
    <article class="signal-card">
      <div class="signal-label">Turnover</div>
      <div class="signal-value">${(Number(summary.turnover_ratio || metrics.turnover_ratio || 0) * 100).toFixed(2)}%</div>
      <div class="mini-label">Estimated rebalance intensity versus current weights</div>
    </article>
    <article class="signal-card">
      <div class="signal-label">Ranking Benchmark</div>
      <div class="signal-value">${summary.most_improved_asset || "N/A"}</div>
      <div class="mini-label">${summary.ranking_benchmark_summary || "No ranking comparison available."}</div>
    </article>
    <article class="signal-card">
      <div class="signal-label">Validation Win Rate</div>
      <div class="signal-value">${summary.ranking_validation_win_rate ?? "N/A"}%</div>
      <div class="mini-label">${summary.ranking_validation_summary || "No rolling validation available."}</div>
    </article>
    <article class="signal-card">
      <div class="signal-label">Validation Winner</div>
      <div class="signal-value">${betterMethodLabel}</div>
      <div class="mini-label">${spreadLabel}</div>
    </article>
    <article class="signal-card">
      <div class="signal-label">Forward Sharpe</div>
      <div class="signal-value">${summary.ranking_validation_current_sharpe ?? "N/A"} / ${summary.ranking_validation_legacy_sharpe ?? "N/A"}</div>
      <div class="mini-label">Current vs legacy realized forward-window Sharpe</div>
    </article>
  `;

  suggestionsHost.innerHTML = Object.entries(data.trade_suggestions || {}).map(([ticker, item]) => `
    <article class="signal-card">
      <div class="signal-label">${ticker}</div>
      <div class="signal-value ${String(item.action || "").toLowerCase()}">${item.action}</div>
      <div class="mini-label">Shift ${Number(item.change_percent || 0).toFixed(2)}% | INR ${Number(item.amount_inr || 0).toFixed(2)} | ${data.ranking_context?.[ticker]?.summary || "No ranking context"}</div>
    </article>
  `).join("");

  renderPieChart(allocationChart, [{
    labels: Object.keys(data.optimized_weights || {}),
    values: Object.values(data.optimized_weights || {}),
    type: "pie",
    hole: 0.48,
    marker: { colors: ["#56c7ff", "#78ffb2", "#ffcc6b", "#ff667d", "#a98cff", "#6fffe9"] },
  }], { title: "Optimized Allocation" });

  renderLineChart(frontierChart, [{
    x: data.efficient_frontier?.vols || [],
    y: data.efficient_frontier?.rets || [],
    type: "scatter",
    mode: "lines",
    name: "Frontier",
    line: { color: "#56c7ff", width: 3 },
  }, {
    x: [Number(data.efficient_frontier?.current_point?.volatility || 0)],
    y: [Number(data.efficient_frontier?.current_point?.return || 0)],
    type: "scatter",
    mode: "markers",
    name: "Current Portfolio",
    marker: { color: "#ffcc6b", size: 11, symbol: "diamond" },
  }, {
    x: [Number(data.efficient_frontier?.optimal_point?.volatility || 0)],
    y: [Number(data.efficient_frontier?.optimal_point?.return || 0)],
    type: "scatter",
    mode: "markers",
    name: "Optimized Portfolio",
    marker: { color: "#78ffb2", size: 12, symbol: "star" },
  }], {
    title: "Efficient Frontier",
    xaxis: { title: "Volatility" },
    yaxis: { title: "Expected Return" },
  });

  renderBarChart(weightsChart, [
    {
      x: Object.keys(data.current_weights || {}),
      y: Object.values(data.current_weights || {}),
      type: "bar",
      name: "Current",
      marker: { color: "#ffcc6b" },
    },
    {
      x: Object.keys(data.optimized_weights || {}),
      y: Object.values(data.optimized_weights || {}),
      type: "bar",
      name: "Optimized",
      marker: { color: "#56c7ff" },
    },
  ], { title: "Current vs Optimized Weights" });

  const rankingBenchmark = data.ranking_benchmark || {};
  const benchmarkTickers = Object.keys(rankingBenchmark.current_scores || {});
  renderBarChart(rankingBenchmarkChart, [
    {
      x: benchmarkTickers,
      y: benchmarkTickers.map((ticker) => Number(rankingBenchmark.current_scores?.[ticker] || 0)),
      type: "bar",
      name: "Current Rank",
      marker: { color: "#56c7ff" },
    },
    {
      x: benchmarkTickers,
      y: benchmarkTickers.map((ticker) => Number(rankingBenchmark.legacy_scores?.[ticker] || 0)),
      type: "bar",
      name: "Legacy Rank",
      marker: { color: "#ffcc6b" },
    },
    {
      x: benchmarkTickers,
      y: benchmarkTickers.map((ticker) => Number(rankingBenchmark.rank_shift?.[ticker] || 0)),
      type: "bar",
      name: "Rank Shift",
      marker: { color: "#78ffb2" },
    },
    {
      x: ["Current", "Legacy"],
      y: [
        Number(rankingValidation.current_avg_forward_return || 0),
        Number(rankingValidation.legacy_avg_forward_return || 0),
      ],
      type: "bar",
      name: "Avg Forward Return",
      marker: { color: "#ff667d" },
    },
  ], { title: "Current vs Legacy Ranking" });
}

export function initPortfolioView(log, holdingsStorageKey) {
  const holdings = JSON.parse(localStorage.getItem(holdingsStorageKey) || "[]");
  const form = document.getElementById("holding-form");
  const holdingsHost = document.getElementById("portfolio-holdings");
  const metricsHost = document.getElementById("portfolio-metrics");
  const suggestionsHost = document.getElementById("portfolio-suggestions");
  const weightsChart = document.getElementById("portfolio-weights-chart");
  const allocationChart = document.getElementById("portfolio-allocation-chart");
  const frontierChart = document.getElementById("portfolio-frontier-chart");
  const rankingBenchmarkChart = document.getElementById("portfolio-ranking-benchmark-chart");
  const clearBtn = document.getElementById("portfolio-clear-btn");
  const optimizeBtn = document.getElementById("portfolio-optimize-btn");
  const exportBtn = document.getElementById("portfolio-export-btn");
  let latestPortfolio = null;
  const hosts = { metricsHost, suggestionsHost, weightsChart, allocationChart, frontierChart, rankingBenchmarkChart };

  exportBtn.addEventListener("click", () => {
    if (!latestPortfolio) {
      log("No portfolio snapshot available to export yet.");
      return;
    }
    downloadJson("portfolio-optimization", latestPortfolio);
    postJson("/api/reports", {
      type: "portfolio",
      title: buildReportTitle("Portfolio Snapshot"),
      payload: latestPortfolio,
      metadata: { holdings: latestPortfolio.holdings?.length || 0 },
    }).then(() => {
      window.dispatchEvent(new CustomEvent("reports:changed"));
    }).catch(() => {});
    log("Exported portfolio optimization snapshot.");
  });

  async function runOptimizationWithHoldings(nextHoldings) {
    holdings.length = 0;
    holdings.push(...nextHoldings);
    syncHoldings();

    metricsHost.innerHTML = `<div class="empty-state">Optimizing portfolio...</div>`;
    suggestionsHost.innerHTML = `<div class="empty-state">Preparing trade instructions...</div>`;
    log("Running portfolio optimization...");

    const data = await postJson("/api/portfolio/optimize", { holdings });
    latestPortfolio = { ...data, holdings: [...holdings] };
    renderPortfolioSnapshot(latestPortfolio, hosts);
    log("Portfolio optimization completed.");
  }

  function syncHoldings() {
    holdingsHost.innerHTML = holdingsTable(holdings);
    localStorage.setItem(holdingsStorageKey, JSON.stringify(holdings));
  }

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const ticker = document.getElementById("holding-ticker").value.trim().toUpperCase();
    const qty = Number(document.getElementById("holding-qty").value);
    const buyPrice = Number(document.getElementById("holding-buy-price").value);
    if (!ticker || qty <= 0 || buyPrice <= 0) {
      log("Holding rejected: invalid ticker, qty, or buy price.");
      return;
    }
    holdings.push({ ticker, qty, buy_price: buyPrice });
    syncHoldings();
    log(`Added holding ${ticker} (${qty} @ ${buyPrice}).`);
  });

    clearBtn.addEventListener("click", () => {
    holdings.length = 0;
    syncHoldings();
    latestPortfolio = null;
    metricsHost.innerHTML = `<div class="empty-state">Portfolio metrics will appear after optimization.</div>`;
    suggestionsHost.innerHTML = `<div class="empty-state">Trade suggestions will appear after optimization.</div>`;
    weightsChart.innerHTML = "";
    allocationChart.innerHTML = "";
    frontierChart.innerHTML = "";
    rankingBenchmarkChart.innerHTML = "";
    log("Portfolio holdings cleared.");
  });

  optimizeBtn.addEventListener("click", async () => {
    try {
      await runOptimizationWithHoldings([...holdings]);
    } catch (error) {
      latestPortfolio = null;
      metricsHost.innerHTML = `<div class="empty-state">${error.message}</div>`;
      suggestionsHost.innerHTML = "";
      weightsChart.innerHTML = "";
      allocationChart.innerHTML = "";
      frontierChart.innerHTML = "";
      rankingBenchmarkChart.innerHTML = "";
      log(`Portfolio optimization failed: ${error.message}`);
    }
  });

  syncHoldings();

  return {
    loadSnapshot(snapshot) {
      latestPortfolio = snapshot;
      const savedHoldings = Array.isArray(snapshot?.holdings) ? snapshot.holdings : [];
      if (savedHoldings.length) {
        holdings.length = 0;
        holdings.push(...savedHoldings);
        syncHoldings();
      }
      renderPortfolioSnapshot(snapshot, hosts);
      log("Loaded saved portfolio snapshot.");
    },
    runDemo() {
      return runOptimizationWithHoldings([
        { ticker: "RELIANCE.NS", qty: 20, buy_price: 2500 },
        { ticker: "INFY.NS", qty: 15, buy_price: 1500 },
        { ticker: "TCS.NS", qty: 10, buy_price: 3900 },
      ]);
    },
  };
}
