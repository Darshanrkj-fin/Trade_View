const darkLayout = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: { color: "#e8f1ff", family: "Rajdhani, sans-serif" },
  margin: { l: 36, r: 18, t: 36, b: 36 },
  xaxis: { gridcolor: "rgba(86, 199, 255, 0.08)", zerolinecolor: "rgba(86, 199, 255, 0.08)" },
  yaxis: { gridcolor: "rgba(86, 199, 255, 0.08)", zerolinecolor: "rgba(86, 199, 255, 0.08)" },
};

export function renderLineChart(target, traces, layout = {}) {
  Plotly.newPlot(target, traces, { ...darkLayout, ...layout }, { responsive: true, displayModeBar: false });
}

export function renderBarChart(target, traces, layout = {}) {
  Plotly.newPlot(target, traces, { ...darkLayout, barmode: "group", ...layout }, { responsive: true, displayModeBar: false });
}

export function renderPieChart(target, traces, layout = {}) {
  Plotly.newPlot(target, traces, { ...darkLayout, ...layout }, { responsive: true, displayModeBar: false });
}
