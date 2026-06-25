const layers = [
  {
    id: 'L0', name: 'Raw Ingestion', command: '/dw-map', color: 'var(--color-text-faint)',
    image: 'Spectral_sum_heatmap_—_sigma_sweep_showing_peak_at_critical_.jpg',
    desc: 'Ingest raw artifacts into an unaveraged point cloud. Each datum is a discrete point. The variance structure IS the topological signal.',
    detail: 'Entropy gate rejects zero-variance inputs. NO_AVERAGING axiom enforced. One question at a time, one probe at a time.',
  },
  {
    id: 'L1', name: 'Persistent Clustering', command: '/dw-filter', color: 'var(--color-teal)',
    image: 'Vietoris-Rips_simplicial_complex_over_zeta_zeros.jpg',
    desc: 'Vietoris-Rips filtration identifies stable H₀ clusters. Long bars = real modules. Short bars = noise.',
    detail: 'Adaptive scale: epsilon_max = 95th percentile of pairwise distances. Number of clusters determined by data geometry, not fixed.',
  },
  {
    id: 'L2', name: 'Topological Synthesis', command: '/dw-ascend', color: 'var(--color-primary)',
    image: 'Betti_number_evolution_curves_—_zeta_vs_GUE_vs_Poisson.jpg',
    desc: 'H₁ persistent loops detect coherence constraints. Gini trajectory monitors quality in real-time.',
    detail: 'Positive Gini slope → ASCEND. Negative → REPROBE. Waypoints > 3 → SPLIT. Shape dominates count.',
  },
  {
    id: 'L3', name: 'Sheaf-Valued Review', command: '/dw-ascend --sheaf', color: 'var(--color-success)',
    image: 'Sheaf_fiber_bundle_over_simplicial_complex_—_gauge_connectio.jpg',
    desc: 'Output carries algebraic structure. Lie algebra fibers over the Rips complex. ker(L_F) = global consistency.',
    detail: 'Local sections must be compatible under restriction maps. Iterate until convergence. Surface to human if obstruction is fundamental.',
  },
]

export default function Pipeline() {
  return (
    <section id="pipeline" style={{ padding: 'clamp(3rem, 8vw, 6rem) 0' }}>
      <div style={{ maxWidth: 'var(--content-wide)', margin: '0 auto', padding: '0 var(--space-6)' }}>
        <div style={{ marginBottom: 'var(--space-12)' }}>
          <div style={{
            fontFamily: 'var(--font-mono)', fontSize: 'var(--text-xs)',
            letterSpacing: '0.12em', textTransform: 'uppercase',
            color: 'var(--color-primary)', marginBottom: 'var(--space-4)',
            display: 'flex', alignItems: 'center', gap: 'var(--space-3)',
          }}>
            The Pipeline
            <span style={{ flex: 1, maxWidth: '3rem', height: 1, background: 'var(--color-primary)', opacity: 0.4 }} />
          </div>
          <h2 style={{
            fontFamily: 'var(--font-display)', fontSize: 'var(--text-2xl)',
            fontWeight: 500, letterSpacing: '-0.03em', lineHeight: 1.08,
          }}>L0 → L1 → L2 → L3</h2>
          <p style={{
            fontSize: 'var(--text-base)', color: 'var(--color-text-muted)',
            marginTop: 'var(--space-4)', maxWidth: '58ch', lineHeight: 1.75,
          }}>
            Information flows upward through persistent homology. Each layer transition
            is gated by a topological phase transition — not a timer, not a checklist.
          </p>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-16)' }}>
          {layers.map((l, i) => (
            <div key={l.id} style={{
              display: 'grid', gridTemplateColumns: i % 2 === 0 ? '1fr 1fr' : '1fr 1fr',
              gap: 'clamp(2rem, 5vw, 4rem)', alignItems: 'center',
              direction: i % 2 === 1 ? 'rtl' : 'ltr',
            }}>
              <div style={{ direction: 'ltr' }}>
                <div style={{
                  display: 'inline-flex', alignItems: 'center', gap: 'var(--space-2)',
                  padding: 'var(--space-1) var(--space-3)',
                  background: 'var(--color-primary-highlight)', color: 'var(--color-primary)',
                  borderRadius: 'var(--radius-full)', fontFamily: 'var(--font-mono)',
                  fontSize: 'var(--text-xs)', fontWeight: 500, textTransform: 'uppercase',
                  letterSpacing: '0.08em', marginBottom: 'var(--space-4)',
                }}>
                  <span style={{ width: 5, height: 5, borderRadius: '50%', background: l.color }} />
                  {l.id}
                </div>
                <h3 style={{
                  fontFamily: 'var(--font-display)', fontSize: 'var(--text-xl)',
                  fontWeight: 500, letterSpacing: '-0.025em', marginBottom: 'var(--space-4)',
                }}>{l.name}</h3>
                <p style={{
                  fontSize: 'var(--text-base)', color: 'var(--color-text-muted)',
                  lineHeight: 1.8, marginBottom: 'var(--space-4)',
                }}>{l.desc}</p>
                <div style={{
                  background: 'var(--color-surface-offset)', borderRadius: 'var(--radius-md)',
                  padding: 'var(--space-3) var(--space-5)', fontFamily: 'var(--font-mono)',
                  fontSize: 'var(--text-xs)', color: 'var(--color-teal)', letterSpacing: '0.04em',
                  marginBottom: 'var(--space-4)',
                }}>{l.command}</div>
                <p style={{
                  fontSize: 'var(--text-sm)', color: 'var(--color-text-faint)', lineHeight: 1.72,
                }}>{l.detail}</p>
              </div>
              <div style={{
                borderRadius: 'var(--radius-xl)', overflow: 'hidden',
                aspectRatio: '4/3', boxShadow: 'var(--shadow-lg)', direction: 'ltr',
              }}>
                <img src={`${import.meta.env.BASE_URL}images/${l.image}`} alt={l.name}
                  style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
