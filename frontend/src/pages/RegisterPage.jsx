import { useState } from "react";
import { Link, Navigate } from "react-router-dom";
import AuthLayout from "../components/AuthLayout.jsx";
import TextField from "../components/TextField.jsx";
import PasswordField from "../components/PasswordField.jsx";
import { useAuth } from "../auth/AuthContext.jsx";
import {
  validateConfirm,
  validateEmail,
  validateName,
  validatePassword,
} from "../auth/validators.js";

export default function RegisterPage() {
  const { user, loading, register } = useAuth();

  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [errors, setErrors] = useState({});
  const [serverError, setServerError] = useState("");
  const [submitting, setSubmitting] = useState(false);

  if (loading) return null;
  if (user) return <Navigate to="/" replace />;

  const setFieldError = (field, message) => setErrors((prev) => ({ ...prev, [field]: message }));

  const onSubmit = async (e) => {
    e.preventDefault();
    setServerError("");

    const next = {
      name: validateName(name),
      email: validateEmail(email),
      password: validatePassword(password),
      confirm: validateConfirm(password, confirm),
    };
    setErrors(next);
    if (Object.values(next).some(Boolean)) return;

    setSubmitting(true);
    try {
      await register(name.trim(), email.trim(), password);
    } catch (err) {
      setServerError(err.message || "Could not create the account.");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <AuthLayout
      title="Create your account"
      subtitle="Register to start analysing memes."
      footer={
        <>
          Already have an account? <Link to="/login">Sign in</Link>
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
          id="register-name"
          label="Full name"
          autoComplete="name"
          autoFocus
          value={name}
          onChange={setName}
          onBlur={() => name && setFieldError("name", validateName(name))}
          error={errors.name}
        />
        <TextField
          id="register-email"
          label="Email"
          type="email"
          autoComplete="email"
          value={email}
          onChange={setEmail}
          onBlur={() => email && setFieldError("email", validateEmail(email))}
          error={errors.email}
        />
        <PasswordField
          id="register-password"
          label="Password"
          autoComplete="new-password"
          value={password}
          onChange={setPassword}
          onBlur={() => password && setFieldError("password", validatePassword(password))}
          error={errors.password}
          hint="At least 8 characters."
        />
        <PasswordField
          id="register-confirm"
          label="Confirm password"
          autoComplete="new-password"
          value={confirm}
          onChange={setConfirm}
          onBlur={() => confirm && setFieldError("confirm", validateConfirm(password, confirm))}
          error={errors.confirm}
        />

        <button className="btn primary auth-submit" type="submit" disabled={submitting}>
          {submitting ? "Creating account..." : "Create account"}
        </button>
      </form>
    </AuthLayout>
  );
}
