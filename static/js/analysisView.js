import { postJson } from "./api.js";
import { buildReportTitle, downloadJson } from "./exportUtils.js";
import { renderBarChart, renderLineChart } from "./plotlyHelpers.js";

function metricCard(label, value, sub = "", trend = "") {
  return `
    <article class="metric-card">
      <div class="metric-label">${label}</div>
      <div class="metric-value">${value}</div>
      <div class="metric-sub ${trend}">${sub}</div>
    </article>
  `;
}

function signalCard(label, value) {
  return `
    <article class="signal-card">
      <div class="signal-label">${label}</div>
      <div class="signal-value">${value ?? "N/A"}</div>
    </article>
  `;
}

function titleCase(value) {
  return String(value || "unknown")
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function safeNumber(value, fallback = 0) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : fallback;
}

function renderAnalysisSnapshot(data, hosts) {
  const {
    metricsHost,
    indicatorsHost,
    backtestsHost,
    benchmarksHost,
    benchmarkChartHost,
    chartHost,
    contextChartHost,
    deltaChartHost,
    signalChartHost,
    statusHost,
  } = hosts;

  const predTrend = data.pred_change_pct >= 0 ? "up" : "down";
  const qualityBand = titleCase(data.analysis_summary?.forecast_quality || data.model_info?.confidence_band);
  metricsHost.innerHTML = [
    metricCard("Last Price", `INR ${Number(data.last_price).toFixed(2)}`),
    metricCard("Realtime Price", `INR ${data.realtime_price ?? "N/A"}`),
    metricCard("Projected Change", `${data.pred_change_pct}%`, "Forecast delta", predTrend),
    metricCard("Model Confidence", `${(Number(data.confidence) * 100).toFixed(1)}%`, `${qualityBand} confidence`),
    metricCard("Forecast Mode", qualityBand, titleCase(data.analysis_summary?.forecast_basis || data.model_info?.forecast_basis)),
  ].join("");

  const note = (data.analysis_summary?.notes || [])[0];
  const validation = data.analysis_summary?.validation || {};
  statusHost.textContent = `${data.model_info?.label || "Forecast"} | ${titleCase(data.analysis_summary?.direction)} bias | fetched ${data.fetched_at}`;
  if (data.degradation?.is_partial) {
    statusHost.textContent += ` | degraded: ${(data.degradation.reasons || []).join(", ")}`;
  }
  if (note) {
    statusHost.textContent += ` | ${note}`;
  }
  if (validation.directional_accuracy !== undefined && validation.directional_accuracy !== null) {
    statusHost.textContent += ` | direction hit-rate ${(Number(validation.directional_accuracy) * 100).toFixed(0)}%`;
  }

  indicatorsHost.innerHTML = Object.entries(data.indicators || {})
    .map(([label, value]) => signalCard(label, value))
    .join("");

  if (data.risk) {
    indicatorsHost.innerHTML += [
      signalCard("Risk Confidence", `${(Number(data.risk.confidence || 0) * 100).toFixed(1)}%`),
      signalCard("Stop Loss", `INR ${Number(data.risk.stop_loss || 0).toFixed(2)}`),
      signalCard("Projected Range", `${Number(data.risk.projected_range?.[0] || 0).toFixed(2)} - ${Number(data.risk.projected_range?.[1] || 0).toFixed(2)}`),
    ].join("");
  }

  backtestsHost.innerHTML = (data.backtests || [])
    .map((item) => `
      <article class="signal-card">
        <div class="signal-label">${item.period}-day Backtest</div>
        <div class="signal-value">Return ${Number(item.profit).toFixed(2)}%</div>
        <div class="mini-label">Accuracy ${Number(item.accuracy).toFixed(1)}% | Buy & Hold ${Number(item.buy_hold).toFixed(2)}%</div>
      </article>
    `)
    .join("");

  const benchmark = data.benchmark_summary || {};
  const currentModel = benchmark.current_model || {};
  const baselineModel = benchmark.baseline_model || {};
  benchmarksHost.innerHTML = [
    `
      <article class="signal-card">
        <div class="signal-label">Calibration Winner</div>
        <div class="signal-value">${titleCase(String(benchmark.better_model || "unknown").replace("model", "model "))}</div>
        <div class="mini-label">${benchmark.windows_evaluated || 0} rolling windows evaluated</div>
      </article>
    `,
    `
      <article class="signal-card">
        <div class="signal-label">Current Model</div>
        <div class="signal-value">${currentModel.avg_mape !== undefined && currentModel.avg_mape !== null ? `${Number(currentModel.avg_mape).toFixed(2)}% MAPE` : "Unavailable"}</div>
        <div class="mini-label">Dir. accuracy ${currentModel.avg_directional_accuracy ?? "N/A"}% | RMSE ${currentModel.avg_rmse ?? "N/A"}</div>
      </article>
    `,
    `
      <article class="signal-card">
        <div class="signal-label">Baseline Model</div>
        <div class="signal-value">${baselineModel.avg_mape !== undefined && baselineModel.avg_mape !== null ? `${Number(baselineModel.avg_mape).toFixed(2)}% MAPE` : "Unavailable"}</div>
        <div class="mini-label">Dir. accuracy ${baselineModel.avg_directional_accuracy ?? "N/A"}% | RMSE ${baselineModel.avg_rmse ?? "N/A"}</div>
      </article>
    `,
    ...((benchmark.notes || []).slice(0, 2).map((benchmarkNote) => `
      <article class="signal-card">
        <div class="signal-label">Calibration Note</div>
        <div class="mini-label">${benchmarkNote}</div>
      </article>
    `)),
  ].join("");

  renderBarChart(benchmarkChartHost, [
    {
      x: ["Current", "Baseline"],
      y: [
        Number(currentModel.avg_directional_accuracy || 0),
        Number(baselineModel.avg_directional_accuracy || 0),
      ],
      type: "bar",
      name: "Directional Accuracy",
      marker: { color: ["#56c7ff", "#ffcc6b"] },
    },
    {
      x: ["Current", "Baseline"],
      y: [
        Number(currentModel.avg_mape || 0),
        Number(baselineModel.avg_mape || 0),
      ],
      type: "bar",
      name: "MAPE",
      marker: { color: ["#78ffb2", "#ff667d"] },
    },
  ], {
    title: "Calibration Comparison",
    yaxis: { title: "Percent" },
  });

  renderLineChart(chartHost, [
    {
      x: data.chart_series.historical.dates,
      y: data.chart_series.historical.close,
      type: "scatter",
      mode: "lines",
      name: "Historical",
      line: { color: "#56c7ff", width: 2 },
    },
    {
      x: data.chart_series.forecast.dates,
      y: data.chart_series.forecast.predictions,
      type: "scatter",
      mode: "lines",
      name: "Forecast",
      line: { color: "#78ffb2", width: 2, dash: "dash" },
    },
  ], {
    title: `${data.ticker} Forecast Matrix`,
  });

  const historicalDates = data.chart_series?.historical?.dates || [];
  const historicalClose = (data.chart_series?.historical?.close || []).map((value) => safeNumber(value));
  const historicalVolume = (data.chart_series?.historical?.volume || []).map((value) => safeNumber(value));
  const forecastDates = data.chart_series?.forecast?.dates || [];
  const forecastPredictions = (data.chart_series?.forecast?.predictions || []).map((value) => safeNumber(value));
  const lastHistoricalDate = historicalDates[historicalDates.length - 1];
  const lastHistoricalPrice = historicalClose[historicalClose.length - 1] ?? safeNumber(data.last_price);
  const forecastDeltaPct = forecastPredictions.map((price, index) => {
    const basePrice = index === 0 ? lastHistoricalPrice : forecastPredictions[index - 1];
    return basePrice ? ((price - basePrice) / basePrice) * 100 : 0;
  });

  renderLineChart(contextChartHost, [
    {
      x: historicalDates,
      y: historicalClose,
      type: "scatter",
      mode: "lines",
      name: "Close",
      line: { color: "#56c7ff", width: 2 },
      yaxis: "y1",
    },
    {
      x: historicalDates.slice(-60),
      y: historicalVolume.slice(-60),
      type: "bar",
      name: "Volume",
      marker: { color: "rgba(120, 255, 178, 0.35)" },
      yaxis: "y2",
    },
  ], {
    title: `${data.ticker} Price + Volume Context`,
    yaxis: { title: "Price" },
    yaxis2: {
      title: "Volume",
      overlaying: "y",
      side: "right",
      showgrid: false,
    },
  });

  renderBarChart(deltaChartHost, [
    {
      x: forecastDates,
      y: forecastDeltaPct,
      type: "bar",
      name: "Daily Delta %",
      marker: {
        color: forecastDeltaPct.map((value) => value >= 0 ? "#78ffb2" : "#ff667d"),
      },
    },
  ], {
    title: "Forecast Step Change",
    yaxis: { title: "Percent" },
  });

  const ma50Gap = data.last_price && data.indicators?.MA50
    ? ((safeNumber(data.last_price) - safeNumber(data.indicators.MA50)) / safeNumber(data.indicators.MA50)) * 100
    : 0;
  const ma200Gap = data.last_price && data.indicators?.MA200
    ? ((safeNumber(data.last_price) - safeNumber(data.indicators.MA200)) / safeNumber(data.indicators.MA200)) * 100
    : 0;
  const signalValues = [
    safeNumber(data.indicators?.RSI),
    safeNumber(data.indicators?.MACD),
    ma50Gap,
    ma200Gap,
    safeNumber(data.pred_change_pct),
  ];

  renderBarChart(signalChartHost, [
    {
      x: ["RSI", "MACD", "Vs MA50 %", "Vs MA200 %", "Forecast %"],
      y: signalValues,
      type: "bar",
      name: "Signal Snapshot",
      marker: {
        color: ["#56c7ff", "#a98cff", "#78ffb2", "#ffcc6b", safeNumber(data.pred_change_pct) >= 0 ? "#78ffb2" : "#ff667d"],
      },
    },
  ], {
    title: "Momentum + Trend Snapshot",
  });
}

export function initAnalysisView(log) {
  const form = document.getElementById("analysis-form");
  const tickerInput = document.getElementById("analysis-ticker");
  const termSelect = document.getElementById("analysis-term");
  const metricsHost = document.getElementById("analysis-metrics");
  const indicatorsHost = document.getElementById("analysis-indicators");
  const backtestsHost = document.getElementById("analysis-backtests");
  const benchmarksHost = document.getElementById("analysis-benchmarks");
  const benchmarkChartHost = document.getElementById("analysis-benchmark-chart");
  const chartHost = document.getElementById("analysis-chart");
  const contextChartHost = document.getElementById("analysis-context-chart");
  const deltaChartHost = document.getElementById("analysis-delta-chart");
  const signalChartHost = document.getElementById("analysis-signal-chart");
  const statusHost = document.getElementById("analysis-status");
  const exportBtn = document.getElementById("analysis-export-btn");
  let latestAnalysis = null;
  const hosts = {
    metricsHost,
    indicatorsHost,
    backtestsHost,
    benchmarksHost,
    benchmarkChartHost,
    chartHost,
    contextChartHost,
    deltaChartHost,
    signalChartHost,
    statusHost,
  };

  exportBtn.addEventListener("click", () => {
    if (!latestAnalysis) {
      log("No analysis snapshot available to export yet.");
      return;
    }
    downloadJson(`analysis-${latestAnalysis.ticker || "snapshot"}`, latestAnalysis);
    postJson("/api/reports", {
      type: "analysis",
      title: buildReportTitle("Analysis Snapshot", latestAnalysis.ticker || "unknown"),
      payload: latestAnalysis,
      metadata: { ticker: latestAnalysis.ticker, forecast_days: latestAnalysis.forecast_days },
    }).then(() => {
      window.dispatchEvent(new CustomEvent("reports:changed"));
    }).catch(() => {});
    log(`Exported analysis snapshot for ${latestAnalysis.ticker || "current view"}.`);
  });

  async function runAnalysisWithInputs(ticker, forecastDays) {
    localStorage.setItem("tradewise.analysis.ticker", ticker);
    localStorage.setItem("tradewise.analysis.term", String(forecastDays));
    tickerInput.value = ticker;
    termSelect.value = String(forecastDays);

    metricsHost.innerHTML = `<div class="empty-state">Running analysis for ${ticker}...</div>`;
    indicatorsHost.innerHTML = `<div class="empty-state">Computing signal stack...</div>`;
    backtestsHost.innerHTML = `<div class="empty-state">Running backtests...</div>`;
    benchmarksHost.innerHTML = `<div class="empty-state">Calibrating forecast against historical windows...</div>`;
    benchmarkChartHost.innerHTML = "";
    contextChartHost.innerHTML = "";
    deltaChartHost.innerHTML = "";
    signalChartHost.innerHTML = "";
    statusHost.textContent = "Fetching market data and model outputs...";
    log(`Running deep scan for ${ticker} on ${forecastDays}-day horizon...`);

    const data = await postJson("/api/analyze-stock", {
      ticker,
      forecast_days: forecastDays,
    });
    latestAnalysis = data;
    renderAnalysisSnapshot(data, hosts);
    log(`Analysis completed for ${ticker}.`);
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const ticker = tickerInput.value.trim().toUpperCase();
    const forecastDays = Number(termSelect.value);

    try {
      await runAnalysisWithInputs(ticker, forecastDays);
    } catch (error) {
      latestAnalysis = null;
      metricsHost.innerHTML = `<div class="empty-state">${error.message}</div>`;
      indicatorsHost.innerHTML = "";
      backtestsHost.innerHTML = "";
      benchmarksHost.innerHTML = "";
      benchmarkChartHost.innerHTML = "";
      contextChartHost.innerHTML = "";
      deltaChartHost.innerHTML = "";
      signalChartHost.innerHTML = "";
      chartHost.innerHTML = `<div class="empty-state">${error.message}</div>`;
      statusHost.textContent = "Analysis degraded or unavailable.";
      log(`Analysis failed: ${error.message}`);
    }
  });

  return {
    loadSnapshot(snapshot) {
      latestAnalysis = snapshot;
      if (snapshot?.ticker) {
        tickerInput.value = snapshot.ticker;
      }
      if (snapshot?.forecast_days) {
        termSelect.value = String(snapshot.forecast_days);
      }
      renderAnalysisSnapshot(snapshot, hosts);
      log(`Loaded saved analysis snapshot for ${snapshot?.ticker || "report"}.`);
    },
    runDemo() {
      return runAnalysisWithInputs("RELIANCE.NS", 30);
    },
  };
}
