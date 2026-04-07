export async function fetchJson(url, options = {}) {
  const { timeoutMs = 15000, ...fetchOptions } = options;
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url, {
      headers: { "Content-Type": "application/json", ...(options.headers || {}) },
      ...fetchOptions,
      signal: controller.signal,
    });
    const payload = await response.json();
    if (!response.ok || !payload.ok) {
      throw new Error(payload.error || payload.message || "Request failed");
    }
    return payload.data;
  } catch (error) {
    if (error.name === "AbortError") {
      throw new Error("Request timed out");
    }
    throw error;
  } finally {
    window.clearTimeout(timeoutId);
  }
}

export function postJson(url, body) {
  return fetchJson(url, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function getJson(url) {
  return fetchJson(url);
}

export function deleteJson(url) {
  return fetchJson(url, {
    method: "DELETE",
  });
}
