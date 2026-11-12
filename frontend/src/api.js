// All backend calls live here. In dev, Vite proxies /api to FastAPI on port 8000.
const BASE = import.meta.env.VITE_API_URL || "/api";

let authToken = null;
let onUnauthorized = null;

export const setToken = (token) => {
  authToken = token || null;
};

// Called when a protected request comes back 401 (expired/invalid session).
export const setUnauthorizedHandler = (handler) => {
  onUnauthorized = handler;
};

function messageFrom(detail, fallback) {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    const joined = detail.map((d) => d.msg || "").filter(Boolean).join(" ");
    return joined || fallback;
  }
  return fallback;
}

async function request(path, options = {}) {
  const headers = new Headers(options.headers || {});
  if (authToken) headers.set("Authorization", `Bearer ${authToken}`);

  let res;
  try {
    res = await fetch(`${BASE}${path}`, { ...options, headers });
  } catch {
    throw new Error("Cannot reach the server. Is the backend running?");
  }

  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      /* response was not JSON */
    }
    if (res.status === 401 && authToken && onUnauthorized && !path.startsWith("/auth/")) {
      onUnauthorized();
    }
    throw new Error(messageFrom(detail, `Request failed (${res.status}).`));
  }
  return res.json();
}

const postJson = (path, body) =>
  request(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

function buildForm({ image, text, fusionType, explain }) {
  const fd = new FormData();
  if (image) fd.append("image", image);
  fd.append("text", text || "");
  if (fusionType) fd.append("fusion_type", fusionType);
  if (explain !== undefined) fd.append("explain", explain ? "true" : "false");
  return fd;
}

// ---- auth
export const register = ({ name, email, password }) =>
  postJson("/auth/register", { name, email, password });
export const login = ({ email, password }) => postJson("/auth/login", { email, password });
export const me = () => request("/auth/me");

// ---- app
export const getHealth = () => request("/health");
export const getModels = () => request("/models");

export const predict = (args) =>
  request("/predict", { method: "POST", body: buildForm(args) });

export const compare = ({ image, text }) =>
  request("/compare", { method: "POST", body: buildForm({ image, text }) });

export const ocr = (image) => {
  const fd = new FormData();
  fd.append("image", image);
  return request("/ocr", { method: "POST", body: fd });
};
