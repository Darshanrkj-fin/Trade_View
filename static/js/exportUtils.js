function timestampSuffix() {
  return new Date().toISOString().replace(/[:.]/g, "-");
}

export function downloadJson(filenamePrefix, payload) {
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `${filenamePrefix}-${timestampSuffix()}.json`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(link.href);
}

export function downloadCsv(filenamePrefix, rows) {
  const safeRows = Array.isArray(rows) ? rows : [];
  if (!safeRows.length) {
    throw new Error("No rows available for CSV export.");
  }

  const headers = Array.from(
    safeRows.reduce((set, row) => {
      Object.keys(row || {}).forEach((key) => set.add(key));
      return set;
    }, new Set()),
  );

  const csv = [
    headers.join(","),
    ...safeRows.map((row) => headers.map((header) => {
      const value = row?.[header];
      const text = value === null || value === undefined ? "" : String(value);
      return `"${text.replace(/"/g, '""')}"`;
    }).join(",")),
  ].join("\n");

  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `${filenamePrefix}-${timestampSuffix()}.csv`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(link.href);
}

export function buildReportTitle(prefix, details = "") {
  const stamp = new Date().toLocaleString();
  return details ? `${prefix} - ${details} - ${stamp}` : `${prefix} - ${stamp}`;
}
