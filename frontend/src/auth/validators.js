// Client-side checks. They mirror the server rules in backend/auth.py
// (the server always re-validates, so these are only for fast feedback).
const EMAIL_RE = /^[^@\s]+@[^@\s]+\.[^@\s]+$/;

export function validateName(value) {
  const name = value.trim().replace(/\s+/g, " ");
  if (name.length < 2) return "Enter your name (at least 2 characters).";
  if (name.length > 60) return "Name must be 60 characters or fewer.";
  return "";
}

export function validateEmail(value) {
  const email = value.trim();
  if (!email) return "Enter your email address.";
  if (email.length > 254 || !EMAIL_RE.test(email)) return "Enter a valid email address.";
  return "";
}

export function validatePassword(value) {
  if (value.length < 8) return "Use at least 8 characters.";
  if (value.length > 128) return "Use 128 characters or fewer.";
  return "";
}

export function validateConfirm(password, confirm) {
  return password === confirm ? "" : "Passwords do not match.";
}
