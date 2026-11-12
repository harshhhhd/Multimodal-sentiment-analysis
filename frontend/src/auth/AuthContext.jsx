import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import * as api from "../api.js";

const AuthContext = createContext(null);
const TOKEN_KEY = "indicsent_token";

// localStorage can throw (private windows, blocked storage), so every access is guarded.
function readToken() {
  try {
    return localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

function writeToken(token) {
  try {
    if (token) localStorage.setItem(TOKEN_KEY, token);
    else localStorage.removeItem(TOKEN_KEY);
  } catch {
    /* storage unavailable: the session just lasts until the tab closes */
  }
}

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(() => Boolean(readToken()));

  const clearSession = useCallback(() => {
    writeToken(null);
    api.setToken(null);
    setUser(null);
  }, []);

  // Restore the session on page load.
  useEffect(() => {
    api.setUnauthorizedHandler(clearSession);
    const saved = readToken();
    if (!saved) return undefined;

    api.setToken(saved);
    let cancelled = false;
    api
      .me()
      .then((u) => {
        if (!cancelled) setUser(u);
      })
      .catch(() => {
        if (!cancelled) clearSession();
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [clearSession]);

  const startSession = useCallback(({ token, user: u }) => {
    writeToken(token);
    api.setToken(token);
    setUser(u);
  }, []);

  const login = useCallback(
    async (email, password) => startSession(await api.login({ email, password })),
    [startSession]
  );

  const register = useCallback(
    async (name, email, password) => startSession(await api.register({ name, email, password })),
    [startSession]
  );

  const value = useMemo(
    () => ({ user, loading, login, register, logout: clearSession }),
    [user, loading, login, register, clearSession]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used inside <AuthProvider>");
  return ctx;
}
