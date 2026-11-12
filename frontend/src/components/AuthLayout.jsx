// Shared card layout for the login and register pages.
export default function AuthLayout({ title, subtitle, children, footer }) {
  return (
    <main className="auth-wrap">
      <section className="auth-card" aria-labelledby="auth-title">
        <p className="auth-brand">IndicSent-M</p>
        <h1 id="auth-title">{title}</h1>
        <p className="auth-sub">{subtitle}</p>
        {children}
        {footer && <p className="auth-switch">{footer}</p>}
      </section>
    </main>
  );
}
