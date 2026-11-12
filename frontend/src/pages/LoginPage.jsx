import { useState } from "react";
import { Link, Navigate, useLocation } from "react-router-dom";
import AuthLayout from "../components/AuthLayout.jsx";
import TextField from "../components/TextField.jsx";
import PasswordField from "../components/PasswordField.jsx";
import { useAuth } from "../auth/AuthContext.jsx";
import { validateEmail } from "../auth/validators.js";

export default function LoginPage() {
  const { user, loading, login } = useAuth();
  const location = useLocation();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [errors, setErrors] = useState({});
  const [serverError, setServerError] = useState("");
  const [submitting, setSubmitting] = useState(false);

  if (loading) return null;
  if (user) {
    const target = (location.state && location.state.from && location.state.from.pathname) || "/";
    return <Navigate to={target} replace />;
  }

  const onSubmit = async (e) => {
    e.preventDefault();
    setServerError("");

    const next = {
      email: validateEmail(email),
      password: password ? "" : "Enter your password.",
    };
    setErrors(next);
    if (next.email || next.password) return;

    setSubmitting(true);
    try {
      await login(email.trim(), password);
    } catch (err) {
      setServerError(err.message || "Could not sign in.");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <AuthLayout
      title="Welcome back"
      subtitle="Sign in to analyse meme sentiment."
      footer={
        <>
          New here? <Link to="/register">Create an account</Link>
        </>
      }
    >
      <form className="auth-form" onSubmit={onSubmit} noValidate>
        {serverError && (
          <div className="alert" role="alert">
            {serverError}
          </div>
        )}

        <TextField
          id="login-email"
          label="Email"
          type="email"
          autoComplete="email"
          autoFocus
          value={email}
          onChange={setEmail}
          error={errors.email}
        />
        <PasswordField
          id="login-password"
          label="Password"
          autoComplete="current-password"
          value={password}
          onChange={setPassword}
          error={errors.password}
        />

        <button className="btn primary auth-submit" type="submit" disabled={submitting}>
          {submitting ? "Signing in..." : "Sign in"}
        </button>
      </form>
    </AuthLayout>
  );
}
