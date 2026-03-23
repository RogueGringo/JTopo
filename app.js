/* ═══════════════════════════════════════════
   B. Jones Portfolio — app.js
   Theme toggle, tabs, counters, scroll reveals,
   mobile menu, BibTeX copy
   ═══════════════════════════════════════════ */

(function () {
  'use strict';

  /* ─── Chart Data (baked from experiment results, eps=3.0) ─── */
  var CHART_DATA = {
    "K100": {
      "Zeta": {
        "0.250": 12.581465, "0.350": 12.482372, "0.400": 12.486117,
        "0.440": 12.489708, "0.460": 12.488384, "0.480": 12.485043,
        "0.490": 12.482742, "0.500": 12.480021, "0.510": 12.476845,
        "0.520": 12.473185, "0.540": 12.464329, "0.560": 12.453212,
        "0.600": 12.4245, "0.650": 12.384081, "0.750": 12.429955
      },
      "Random": {
        "0.250": 24.452096, "0.350": 24.278636, "0.400": 24.217621,
        "0.440": 24.212372, "0.460": 24.196412, "0.480": 24.187745,
        "0.490": 24.191457, "0.500": 24.195158, "0.510": 24.196099,
        "0.520": 24.193428, "0.540": 24.177981, "0.560": 24.153654,
        "0.600": 24.110624, "0.650": 24.106577, "0.750": 24.161106
      },
      "GUE": {
        "0.250": 15.627424, "0.350": 15.52271, "0.400": 15.522373,
        "0.440": 15.524901, "0.460": 15.525166, "0.480": 15.525402,
        "0.490": 15.525902, "0.500": 15.526629, "0.510": 15.527118,
        "0.520": 15.526672, "0.540": 15.51998, "0.560": 15.497535,
        "0.600": 15.453711, "0.650": 15.426708, "0.750": 15.418362
      }
    },
    "K200": {
      "Zeta": {
        "0.440": 11.796721, "0.480": 11.787438, "0.500": 11.784063,
        "0.520": 11.780065, "0.560": 11.773014
      },
      "GUE": {
        "0.440": 15.000902, "0.480": 15.00581, "0.500": 15.003759,
        "0.520": 14.996844, "0.560": 14.96691
      }
    }
  };

  /* ─── Canvas Color Resolution ─── */
  var CHART_COLORS = {};
  function resolveChartColors() {
    var cs = getComputedStyle(document.documentElement);
    CHART_COLORS.gold = cs.getPropertyValue('--color-primary').trim() || '#d08a28';
    CHART_COLORS.teal = cs.getPropertyValue('--color-teal').trim() || '#45a8b0';
    CHART_COLORS.gray = cs.getPropertyValue('--color-text-faint').trim() || '#544f3e';
    CHART_COLORS.bg = cs.getPropertyValue('--color-bg').trim() || '#0f0d08';
    CHART_COLORS.surface = cs.getPropertyValue('--color-surface').trim() || '#16140d';
    CHART_COLORS.text = cs.getPropertyValue('--color-text').trim() || '#d6d0be';
    CHART_COLORS.textMuted = cs.getPropertyValue('--color-text-muted').trim() || '#817a66';
    CHART_COLORS.border = cs.getPropertyValue('--color-border').trim() || '#35311e';
  }
  resolveChartColors();

  /* ─── Theme Toggle ─── */
  const root = document.documentElement;
  const toggle = document.getElementById('themeToggle');
  let theme = root.getAttribute('data-theme') || 'dark';
  root.setAttribute('data-theme', theme);

  const sunSVG = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="5"/><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/></svg>';
  const moonSVG = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>';

  function updateToggleIcon() {
    if (!toggle) return;
    toggle.innerHTML = theme === 'dark' ? sunSVG : moonSVG;
    toggle.setAttribute('aria-label', 'Switch to ' + (theme === 'dark' ? 'light' : 'dark') + ' mode');
  }
  updateToggleIcon();

  if (toggle) {
    toggle.addEventListener('click', function () {
      theme = theme === 'dark' ? 'light' : 'dark';
      root.setAttribute('data-theme', theme);
      updateToggleIcon();
      resolveChartColors();
      if (typeof redrawSigmaSweep === 'function') redrawSigmaSweep();
      if (typeof redrawPremiumChart === 'function') redrawPremiumChart();
      if (typeof restartParticles === 'function') restartParticles();
    });
  }

  /* ─── Mobile Menu ─── */
  const hamburger = document.getElementById('hamburgerBtn');
  const mobileNav = document.getElementById('mobileNav');

  if (hamburger && mobileNav) {
    hamburger.addEventListener('click', function () {
      mobileNav.classList.toggle('open');
      hamburger.setAttribute('aria-expanded', mobileNav.classList.contains('open'));
    });

    // Close on link click
    mobileNav.querySelectorAll('a').forEach(function (link) {
      link.addEventListener('click', function () {
        mobileNav.classList.remove('open');
        hamburger.setAttribute('aria-expanded', 'false');
      });
    });
  }

  /* ─── Tab Switching ─── */
  const tabBtns = document.querySelectorAll('.tab-btn');
  const tabPanels = document.querySelectorAll('.tab-panel');

  tabBtns.forEach(function (btn) {
    btn.addEventListener('click', function () {
      var targetTab = btn.getAttribute('data-tab');

      tabBtns.forEach(function (b) {
        b.classList.remove('active');
        b.setAttribute('aria-selected', 'false');
      });
      tabPanels.forEach(function (p) {
        p.classList.remove('active');
      });

      btn.classList.add('active');
      btn.setAttribute('aria-selected', 'true');

      var panel = document.getElementById('tab-' + targetTab);
      if (panel) panel.classList.add('active');
    });

    // Keyboard navigation for tabs
    btn.addEventListener('keydown', function (e) {
      var btns = Array.from(tabBtns);
      var idx = btns.indexOf(btn);
      var next;

      if (e.key === 'ArrowRight') {
        next = btns[(idx + 1) % btns.length];
      } else if (e.key === 'ArrowLeft') {
        next = btns[(idx - 1 + btns.length) % btns.length];
      } else if (e.key === 'Home') {
        next = btns[0];
      } else if (e.key === 'End') {
        next = btns[btns.length - 1];
      }

      if (next) {
        e.preventDefault();
        next.focus();
        next.click();
      }
    });
  });

  /* ─── Animated Counters ─── */
  function animateCounter(el) {
    var target = parseInt(el.getAttribute('data-count'), 10);
    var prefix = el.getAttribute('data-prefix') || '';
    var suffix = el.getAttribute('data-suffix') || '';
    var duration = 1800;
    var start = performance.now();

    function easeOutExpo(t) {
      return t === 1 ? 1 : 1 - Math.pow(2, -10 * t);
    }

    function step(now) {
      var elapsed = now - start;
      var progress = Math.min(elapsed / duration, 1);
      var eased = easeOutExpo(progress);
      var current = Math.round(eased * target);

      el.textContent = prefix + current.toLocaleString() + suffix;

      if (progress < 1) {
        requestAnimationFrame(step);
      }
    }

    requestAnimationFrame(step);
  }

  // Observe hero stats
  var statValues = document.querySelectorAll('.stat-value[data-count]');
  var countersAnimated = false;

  if (statValues.length > 0 && 'IntersectionObserver' in window) {
    var counterObserver = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting && !countersAnimated) {
          countersAnimated = true;
          statValues.forEach(animateCounter);
          counterObserver.disconnect();
        }
      });
    }, { threshold: 0.3 });

    statValues.forEach(function (el) {
      counterObserver.observe(el);
    });
  }

  /* ─── Scroll Reveals ─── */
  var reveals = document.querySelectorAll('.reveal');

  if (reveals.length > 0 && 'IntersectionObserver' in window) {
    var revealObserver = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          entry.target.classList.add('visible');
          revealObserver.unobserve(entry.target);
        }
      });
    }, {
      threshold: 0.08,
      rootMargin: '0px 0px -40px 0px'
    });

    reveals.forEach(function (el) {
      revealObserver.observe(el);
    });
  } else {
    // Fallback: show all
    reveals.forEach(function (el) {
      el.classList.add('visible');
    });
  }

  /* ─── BibTeX Copy ─── */
  var copyBtn = document.getElementById('copyBibtex');
  var bibtexBlock = document.getElementById('bibtex');

  if (copyBtn && bibtexBlock) {
    copyBtn.addEventListener('click', function () {
      // Get just the BibTeX text, not the button text
      var text = bibtexBlock.textContent.replace('Copy', '').trim();

      if (navigator.clipboard) {
        navigator.clipboard.writeText(text).then(function () {
          copyBtn.textContent = 'Copied!';
          setTimeout(function () {
            copyBtn.textContent = 'Copy';
          }, 2000);
        }).catch(function () {
          fallbackCopy(text);
        });
      } else {
        fallbackCopy(text);
      }
    });
  }

  function fallbackCopy(text) {
    var ta = document.createElement('textarea');
    ta.value = text;
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    try {
      document.execCommand('copy');
      if (copyBtn) {
        copyBtn.textContent = 'Copied!';
        setTimeout(function () {
          copyBtn.textContent = 'Copy';
        }, 2000);
      }
    } catch (e) {
      // Silently fail
    }
    document.body.removeChild(ta);
  }

  /* ─── Active Nav Highlight (optional) ─── */
  var navLinks = document.querySelectorAll('.nav-links a');
  var sections = [];

  navLinks.forEach(function (link) {
    var href = link.getAttribute('href');
    if (href && href.startsWith('#')) {
      var section = document.getElementById(href.slice(1));
      if (section) {
        sections.push({ link: link, section: section });
      }
    }
  });

  if (sections.length > 0 && 'IntersectionObserver' in window) {
    var navObserver = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          var id = entry.target.id;
          navLinks.forEach(function (l) {
            l.style.color = '';
          });
          sections.forEach(function (s) {
            if (s.section.id === id) {
              s.link.style.color = 'var(--color-text)';
            }
          });
        }
      });
    }, {
      threshold: 0.15,
      rootMargin: '-80px 0px -50% 0px'
    });

    sections.forEach(function (s) {
      navObserver.observe(s.section);
    });
  }

  /* ─── K-Progression Timeline ─── */
  var kTimeline = document.getElementById('kTimeline');
  if (kTimeline && 'IntersectionObserver' in window) {
    var waypoints = kTimeline.querySelectorAll('.k-waypoint');
    var timelineObserver = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          waypoints.forEach(function (wp) {
            var delay = parseInt(wp.getAttribute('data-delay') || '0', 10);
            setTimeout(function () {
              wp.classList.add('visible');
            }, delay);
          });
          timelineObserver.disconnect();
        }
      });
    }, { threshold: 0.15 });
    timelineObserver.observe(kTimeline);
  }

  /* ─── Hero Particle Animation ─── */
  var heroCanvas = document.getElementById('heroCanvas');
  var heroCtx = heroCanvas ? heroCanvas.getContext('2d') : null;
  var particles = [];
  var particleCount = window.innerWidth < 640 ? 15 : 46;
  var particleAnimId = null;

  function initParticles() {
    if (!heroCanvas || !heroCtx) return;
    var dpr = window.devicePixelRatio || 1;
    heroCanvas.width = heroCanvas.offsetWidth * dpr;
    heroCanvas.height = heroCanvas.offsetHeight * dpr;
    heroCtx.setTransform(dpr, 0, 0, dpr, 0, 0);

    particles = [];
    var w = heroCanvas.offsetWidth;
    var h = heroCanvas.offsetHeight;
    var centerX = w * 0.5;
    var targetY = h * 0.45;

    for (var i = 0; i < particleCount; i++) {
      var angle = (Math.PI * 2 * i) / particleCount + Math.random() * 0.3;
      var startR = Math.max(w, h) * 0.6 + Math.random() * 100;
      particles.push({
        x: centerX + Math.cos(angle) * startR,
        y: targetY + Math.sin(angle) * startR * 0.5,
        targetX: centerX + (Math.random() - 0.5) * w * 0.3,
        targetY: targetY + (Math.random() - 0.5) * 20,
        size: 1.5 + Math.random() * 1.5,
        alpha: 0.3 + Math.random() * 0.5,
        phase: Math.random() * Math.PI * 2,
        speed: 0.003 + Math.random() * 0.004
      });
    }
  }

  function drawParticles() {
    if (!heroCtx) return;
    var w = heroCanvas.offsetWidth;
    var h = heroCanvas.offsetHeight;
    heroCtx.clearRect(0, 0, w, h);

    var color = CHART_COLORS.gold || '#d08a28';

    for (var i = 0; i < particles.length; i++) {
      var p = particles[i];
      p.phase += p.speed;
      var progress = Math.min(1, p.phase / (Math.PI * 2));
      var ease = 1 - Math.pow(1 - progress, 3);

      p.x += (p.targetX - p.x) * 0.008;
      p.y += (p.targetY - p.y) * 0.008;

      var breath = 0.7 + 0.3 * Math.sin(p.phase * 0.5);

      heroCtx.beginPath();
      heroCtx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
      heroCtx.fillStyle = color;
      heroCtx.globalAlpha = p.alpha * breath * ease;
      heroCtx.fill();

      for (var j = i + 1; j < particles.length; j++) {
        var q = particles[j];
        var dx = p.x - q.x;
        var dy = p.y - q.y;
        var dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 120) {
          heroCtx.beginPath();
          heroCtx.moveTo(p.x, p.y);
          heroCtx.lineTo(q.x, q.y);
          heroCtx.strokeStyle = color;
          heroCtx.globalAlpha = (1 - dist / 120) * 0.15 * ease;
          heroCtx.lineWidth = 0.5;
          heroCtx.stroke();
        }
      }
    }
    heroCtx.globalAlpha = 1;
    particleAnimId = requestAnimationFrame(drawParticles);
  }

  function restartParticles() {
    if (particleAnimId) cancelAnimationFrame(particleAnimId);
    initParticles();
    drawParticles();
  }

  if (!window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    initParticles();
    drawParticles();
    window.addEventListener('resize', function () {
      if (particleAnimId) cancelAnimationFrame(particleAnimId);
      initParticles();
      drawParticles();
    });
  }

})();
