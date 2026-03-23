import { useState } from 'react'

export default function Install() {
  const [copied, setCopied] = useState(false)
  const cmd = 'claude plugin add gh:RogueGringo/driftwave-v0'

  const copy = () => {
    navigator.clipboard.writeText(cmd).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    })
  }

  return (
    <section id="install" style={{ padding: 'clamp(3rem, 8vw, 6rem) 0' }}>
      <div style={{ maxWidth: 'var(--content-narrow)', margin: '0 auto', padding: '0 var(--space-6)', textAlign: 'center' }}>
        <div style={{
          fontFamily: 'var(--font-mono)', fontSize: 'var(--text-xs)',
          letterSpacing: '0.12em', textTransform: 'uppercase',
          color: 'var(--color-primary)', marginBottom: 'var(--space-4)',
        }}>Install</div>
        <h2 style={{
          fontFamily: 'var(--font-display)', fontSize: 'var(--text-2xl)',
          fontWeight: 500, letterSpacing: '-0.03em', lineHeight: 1.08,
          marginBottom: 'var(--space-6)',
        }}>One Command</h2>
        <p style={{
          fontSize: 'var(--text-base)', color: 'var(--color-text-muted)',
          marginBottom: 'var(--space-10)', lineHeight: 1.75,
        }}>
          After install, every Claude Code session sees the driftwave skills.
          The five axioms load. The pipeline runs.
        </p>

        <div style={{
          background: 'var(--color-surface)', border: '1px solid var(--color-border)',
          borderRadius: 'var(--radius-xl)', padding: 'var(--space-6) var(--space-8)',
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          gap: 'var(--space-4)', marginBottom: 'var(--space-8)',
        }}>
          <code style={{
            fontFamily: 'var(--font-mono)', fontSize: 'var(--text-sm)',
            color: 'var(--color-teal)', letterSpacing: '0.02em',
          }}>{cmd}</code>
          <button onClick={copy} style={{
            padding: 'var(--space-2) var(--space-5)',
            background: copied ? 'var(--color-success)' : 'var(--color-primary)',
            color: 'var(--color-bg)', borderRadius: 'var(--radius-full)',
            fontSize: 'var(--text-xs)', fontWeight: 700, border: 'none', cursor: 'pointer',
            fontFamily: 'var(--font-mono)', letterSpacing: '0.04em',
            transition: 'background var(--transition-interactive)',
          }}>{copied ? 'COPIED' : 'COPY'}</button>
        </div>

        <div style={{
          background: 'var(--color-surface)', border: '1px solid var(--color-border)',
          borderRadius: 'var(--radius-lg)', padding: 'var(--space-5) var(--space-6)',
          fontFamily: 'var(--font-mono)', fontSize: 'var(--text-xs)',
          color: 'var(--color-text-muted)', lineHeight: 2.2, textAlign: 'left',
          whiteSpace: 'pre',
        }}>
{`plugins/driftwave/
├── .claude-plugin/plugin.json
├── skills/
│   ├── dw-map/SKILL.md
│   ├── dw-filter/SKILL.md
│   ├── dw-ascend/SKILL.md
│   ├── wavefront/SKILL.md
│   ├── topological-brainstorm/SKILL.md
│   └── boundary-mode/SKILL.md
├── hooks/hooks.json
├── agents/gini-watchdog.md
└── README.md`}
        </div>
      </div>
    </section>
  )
}
