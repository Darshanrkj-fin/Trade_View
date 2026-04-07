import { deleteJson, fetchJson, getJson, postJson } from "./api.js";
import { initAnalysisView } from "./analysisView.js";
import { initNewsView } from "./newsView.js";
import { initPortfolioView } from "./portfolioView.js";

const STORAGE_KEYS = {
  ticker: "tradewise.analysis.ticker",
  term: "tradewise.analysis.term",
  holdings: "tradewise.portfolio.holdings",
};

function log(message) {
  document.getElementById("console-log").textContent = message;
}

function showToast(title, detail = "", kind = "info") {
  const host = document.getElementById("toast-host");
  if (!host) {
    return;
  }

  const toast = document.createElement("div");
  toast.className = `toast ${kind}`;
  toast.innerHTML = `
    <div class="toast-title">${title}</div>
    ${detail ? `<div class="toast-detail">${detail}</div>` : ""}
  `;
  host.appendChild(toast);

  window.setTimeout(() => {
    toast.remove();
  }, 2800);
}

let reportRecords = [];

function renderReportHistory(records) {
  const host = document.getElementById("report-history-feed");
  if (!host) {
    return;
  }

  host.innerHTML = records.slice(0, 12).map((report) => `
    <article class="feed-card" data-report-id="${report.id}" data-report-type="${report.type}">
      <h3>${report.metadata?.pinned ? "Pinned | " : ""}${report.title}</h3>
      <p>${report.type}</p>
      <p>${report.created_at || "Unknown time"}</p>
      <p>${Object.entries(report.metadata || {}).map(([key, value]) => `${key}: ${value}`).join(" | ") || "No metadata"}</p>
      <div class="button-row">
        <button class="ghost-btn report-open-btn" type="button" data-report-id="${report.id}" data-report-type="${report.type}">Open Snapshot</button>
        <button class="ghost-btn report-pin-btn" type="button" data-report-id="${report.id}" data-pinned="${report.metadata?.pinned ? "1" : "0"}">${report.metadata?.pinned ? "Unpin" : "Pin"}</button>
        <button class="ghost-btn report-delete-btn" type="button" data-report-id="${report.id}">Delete</button>
      </div>
    </article>
  `).join("") || `<div class="empty-state">No saved reports yet.</div>`;
}

function applyReportFilters() {
  const searchValue = (document.getElementById("report-search")?.value || "").trim().toLowerCase();
  const typeValue = document.getElementById("report-type-filter")?.value || "all";

  const filtered = reportRecords.filter((report) => {
    const matchesType = typeValue === "all" || report.type === typeValue;
    const metadataText = Object.values(report.metadata || {}).join(" ").toLowerCase();
    const searchText = `${report.title} ${report.type} ${metadataText}`.toLowerCase();
    const matchesSearch = !searchValue || searchText.includes(searchValue);
    return matchesType && matchesSearch;
  });

  filtered.sort((a, b) => {
    const aPinned = a.metadata?.pinned ? 1 : 0;
    const bPinned = b.metadata?.pinned ? 1 : 0;
    if (aPinned !== bPinned) {
      return bPinned - aPinned;
    }
    return String(b.created_at || "").localeCompare(String(a.created_at || ""));
  });

  renderReportHistory(filtered);
}

async function refreshReportHistory() {
  const host = document.getElementById("report-history-feed");
  if (!host) {
    return;
  }

  try {
    const data = await getJson("/api/reports");
    reportRecords = data.reports || [];
    applyReportFilters();
  } catch (error) {
    host.innerHTML = `<div class="empty-state">Report history unavailable.</div>`;
  }
}

function activateView(viewName) {
  document.querySelectorAll(".nav-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.view === viewName);
  });
  document.querySelectorAll(".view").forEach((view) => {
    view.classList.toggle("active", view.id === `view-${viewName}`);
  });
  log(`View changed to ${viewName}.`);
}

async function bootstrap() {
  let analysisView;
  let newsView;
  let portfolioView;
  const termSelect = document.getElementById("analysis-term");
  const diagnosticsHost = document.getElementById("diagnostics-feed");
  const reportsHost = document.getElementById("report-history-feed");
  const marketStrip = document.getElementById("market-strip");

  log("Initializing terminal shell...");

  try {
    const config = await fetchJson("/api/config", { timeoutMs: 5000 });
    termSelect.innerHTML = config.term_options
      .map((option) => `<option value="${option.value}" ${option.value === config.forecast_days ? "selected" : ""}>${option.label} (${option.value}d)</option>`)
      .join("");

    const storedTicker = localStorage.getItem(STORAGE_KEYS.ticker);
    const storedTerm = localStorage.getItem(STORAGE_KEYS.term);
    if (storedTicker) {
      document.getElementById("analysis-ticker").value = storedTicker;
    }
    if (storedTerm) {
      termSelect.value = storedTerm;
    }
  } catch (error) {
    if (termSelect && !termSelect.innerHTML.trim()) {
      termSelect.innerHTML = `
        <option value="7">Short Term (7d)</option>
        <option value="30" selected>Medium Term (30d)</option>
        <option value="90">Long Term (90d)</option>
      `;
    }
    log(`Startup warning: ${error.message}. Loading remaining feeds in background.`);
  }

  Promise.allSettled([
    fetchJson("/api/market-status", { timeoutMs: 6000 }),
    fetchJson("/api/diagnostics", { timeoutMs: 4000 }),
    refreshReportHistory(),
  ]).then(([marketResult, diagnosticsResult]) => {
    if (marketResult?.status === "fulfilled") {
      const market = marketResult.value;
      const chips = Object.values(market.indices || {}).map((entry) => `
        <span class="market-chip ${entry.direction || ""}">
          ${entry.label}: <strong>${entry.current ?? "N/A"}</strong> (${entry.change_pct ?? "0"}%)
        </span>
      `);
      marketStrip.innerHTML = chips.join("") || `<span class="market-chip">Market status unavailable</span>`;
    } else {
      marketStrip.innerHTML = `<span class="market-chip">Market status delayed</span>`;
    }

    if (diagnosticsHost) {
      if (diagnosticsResult?.status === "fulfilled") {
        const diagnostics = diagnosticsResult.value;
        diagnosticsHost.innerHTML = (diagnostics.events || []).slice(0, 6).map((event) => `
          <article class="feed-card">
            <h3>${event.message}</h3>
            <p>${event.timestamp}</p>
            <p>${event.detail}</p>
          </article>
        `).join("") || `<div class="empty-state">No diagnostics events yet.</div>`;
      } else {
        diagnosticsHost.innerHTML = `<div class="empty-state">Diagnostics unavailable.</div>`;
      }
    }

    if (reportsHost && !reportsHost.innerHTML.trim()) {
      reportsHost.innerHTML = `<div class="empty-state">Report history unavailable.</div>`;
    }

    log("Terminal ready. Background feeds loaded.");
  });

  document.querySelectorAll(".nav-btn").forEach((btn) => {
    btn.addEventListener("click", () => activateView(btn.dataset.view));
  });

  document.getElementById("report-search")?.addEventListener("input", applyReportFilters);
  document.getElementById("report-type-filter")?.addEventListener("change", applyReportFilters);

  analysisView = initAnalysisView(log);
  newsView = initNewsView(log);
  portfolioView = initPortfolioView(log, STORAGE_KEYS.holdings);

  document.getElementById("welcome-sample-analysis")?.addEventListener("click", async () => {
    activateView("analysis");
    try {
      await analysisView?.runDemo();
      showToast("Sample Analysis Ready", "Loaded a demo RELIANCE.NS analysis.", "success");
    } catch (error) {
      log(`Demo analysis failed: ${error.message}`);
      showToast("Demo Analysis Failed", error.message, "error");
    }
  });

  document.getElementById("welcome-sample-news")?.addEventListener("click", async () => {
    activateView("news");
    try {
      await newsView?.runDemo();
      showToast("News Scan Ready", "Loaded a live recommendation scan.", "success");
    } catch (error) {
      log(`Demo news scan failed: ${error.message}`);
      showToast("News Demo Failed", error.message, "error");
    }
  });

  document.getElementById("welcome-sample-portfolio")?.addEventListener("click", async () => {
    activateView("portfolio");
    try {
      await portfolioView?.runDemo();
      showToast("Portfolio Demo Ready", "Loaded a sample allocation scenario.", "success");
    } catch (error) {
      log(`Demo portfolio failed: ${error.message}`);
      showToast("Portfolio Demo Failed", error.message, "error");
    }
  });

  document.getElementById("report-history-feed")?.addEventListener("click", async (event) => {
    const button = event.target.closest(".report-open-btn");
    const pinButton = event.target.closest(".report-pin-btn");
    const deleteButton = event.target.closest(".report-delete-btn");

    if (button) {
      const reportId = button.dataset.reportId;
      const reportType = button.dataset.reportType || "";
      if (!reportId) {
        return;
      }

      try {
        const report = await getJson(`/api/reports/${reportId}`);
        const payload = report.payload;
        if (reportType === "analysis") {
          activateView("analysis");
          analysisView?.loadSnapshot(payload);
        } else if (reportType === "news" || reportType === "news_csv") {
          activateView("news");
          newsView?.loadSnapshot(payload);
        } else if (reportType === "portfolio") {
          activateView("portfolio");
          portfolioView?.loadSnapshot(payload);
        } else {
          activateView("about");
          log(`Loaded report ${report.title}.`);
        }
      } catch (error) {
        log(`Failed to open saved report: ${error.message}`);
        showToast("Open Failed", error.message, "error");
      }
      return;
    }

    if (pinButton) {
      const reportId = pinButton.dataset.reportId;
      const currentlyPinned = pinButton.dataset.pinned === "1";
      try {
        await postJson(`/api/reports/${reportId}/pin`, { pinned: !currentlyPinned });
        await refreshReportHistory();
        log(`Report ${currentlyPinned ? "unpinned" : "pinned"}.`);
        showToast(currentlyPinned ? "Report Unpinned" : "Report Pinned", "Report history updated.", "success");
      } catch (error) {
        log(`Failed to update pin state: ${error.message}`);
        showToast("Pin Failed", error.message, "error");
      }
      return;
    }

    if (deleteButton) {
      const reportId = deleteButton.dataset.reportId;
      const reportCard = deleteButton.closest(".feed-card");
      const reportTitle = reportCard?.querySelector("h3")?.textContent || "this report";
      const confirmed = window.confirm(`Delete ${reportTitle}? This cannot be undone.`);
      if (!confirmed) {
        log("Report deletion canceled.");
        showToast("Deletion Canceled", "Report was kept in history.");
        return;
      }
      try {
        await deleteJson(`/api/reports/${reportId}`);
        await refreshReportHistory();
        log("Report deleted.");
        showToast("Report Deleted", "Snapshot removed from history.", "success");
      } catch (error) {
        log(`Failed to delete report: ${error.message}`);
        showToast("Delete Failed", error.message, "error");
      }
    }
  });

  window.addEventListener("reports:changed", () => {
    refreshReportHistory();
  });
}

bootstrap();
