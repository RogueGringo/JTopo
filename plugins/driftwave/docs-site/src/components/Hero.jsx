export default function Hero() {
  return (
    <section style={{
      position: 'relative', overflow: 'hidden', minHeight: '92vh',
      display: 'flex', alignItems: 'flex-end',
      paddingBottom: 'clamp(3rem, 8vw, 6rem)',
    }}>
      {/* Background */}
      <div style={{ position: 'absolute', inset: 0, zIndex: 0 }}>
        <img
          src={`${import.meta.env.BASE_URL}images/Hero_banner_—_mathematician's_cosmos_with_Doberman_silhouett.jpg`}
          alt=""
          style={{ width: '100%', height: '100%', objectFit: 'cover', objectPosition: 'center 25%', filter: 'brightness(0.3) saturate(0.65)' }}
        />
        <div style={{
          position: 'absolute', inset: 0,
          background: 'linear-gradient(to bottom, transparent 0%, rgba(15,13,8,0.25) 45%, var(--color-bg) 100%)',
        }} />
      </div>

      {/* Content */}
      <div style={{ position: 'relative', zIndex: 1, maxWidth: 820, padding: '0 var(--space-6)', margin: '0 auto' }}>
        <div style={{
          display: 'inline-flex', alignItems: 'center', gap: 'var(--space-2)',
          fontFamily: 'var(--font-mono)', fontSize: 'var(--text-xs)',
          letterSpacing: '0.14em', textTransform: 'uppercase',
          color: 'var(--color-primary)', marginBottom: 'var(--space-6)',
        }}>
          <span style={{
            width: 6, height: 6, borderRadius: '50%',
            background: 'var(--color-primary)',
            animation: 'pdot 2.4s ease-in-out infinite',
          }} />
          Adaptive Topological Field Theory
        </div>

        <h1 style={{
          fontFamily: 'var(--font-display)', fontSize: 'var(--text-3xl)',
          fontWeight: 400, letterSpacing: '-0.035em', lineHeight: 1.05,
          marginBottom: 'var(--space-6)',
        }}>
          <em style={{ fontStyle: 'italic', color: 'var(--color-primary)' }}>driftwave</em>
          <br />
          Topology as Cognition
        </h1>

        <p style={{
          fontSize: 'var(--text-lg)', color: 'var(--color-text-muted)',
          maxWidth: '58ch', marginBottom: 'var(--space-10)', lineHeight: 1.65,
        }}>
          A Claude Code plugin that operationalizes sheaf-valued persistent homology
          as a cognitive pipeline. Five axioms. Four abstraction layers. One principle:
          shape over count.
        </p>

        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 'var(--space-4)', marginBottom: 'var(--space-12)' }}>
          <a href="#install" style={{
            display: 'inline-flex', alignItems: 'center', gap: 'var(--space-2)',
            padding: 'var(--space-3) var(--space-8)',
            background: 'var(--color-primary)', color: 'var(--color-bg)',
            borderRadius: 'var(--radius-full)', fontSize: 'var(--text-sm)',
            fontWeight: 700, textDecoration: 'none',
          }}>Install Plugin →</a>
          <a href="#pipeline" style={{
            display: 'inline-flex', alignItems: 'center', gap: 'var(--space-2)',
            padding: 'var(--space-3) var(--space-8)',
            border: '1px solid var(--color-border)', color: 'var(--color-text-muted)',
            borderRadius: 'var(--radius-full)', fontSize: 'var(--text-sm)',
            fontWeight: 600, textDecoration: 'none',
          }}>How It Works</a>
        </div>

        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 'var(--space-8)' }}>
          {[
            { value: '4', label: 'Abstraction Layers' },
            { value: '5', label: 'Governing Axioms' },
            { value: '6', label: 'Skills' },
            { value: '0.935', label: 'Cross-Model r' },
          ].map(s => (
            <div key={s.label} style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-1)' }}>
              <span style={{
                fontFamily: 'var(--font-mono)', fontSize: 'var(--text-xl)',
                fontWeight: 500, color: 'var(--color-primary)', letterSpacing: '-0.02em',
              }}>{s.value}</span>
              <span style={{
                fontFamily: 'var(--font-mono)', fontSize: 'var(--text-xs)',
                letterSpacing: '0.1em', textTransform: 'uppercase', color: 'var(--color-text-faint)',
              }}>{s.label}</span>
            </div>
          ))}
        </div>
      </div>

      <style>{`@keyframes pdot{0%,100%{opacity:1;transform:scale(1)}50%{opacity:0.45;transform:scale(0.65)}}`}</style>
    </section>
  )
}
