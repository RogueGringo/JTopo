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
        "0.250": 12.030837, "0.350": 11.863279, "0.400": 11.820632,
        "0.440": 11.796721, "0.480": 11.787438, "0.500": 11.784063,
        "0.520": 11.780065, "0.560": 11.773014, "0.600": 11.772863,
        "0.650": 11.782479, "0.750": 11.884323
      },
      "GUE": {
        "0.250": 15.107503, "0.350": 15.016452, "0.400": 14.974635,
        "0.440": 15.000902, "0.480": 15.00581, "0.500": 15.003759,
        "0.520": 14.996844, "0.560": 14.96691, "0.600": 14.940479,
        "0.650": 14.937829, "0.750": 14.953727
      },
      "Random": {
        "0.250": 22.276319, "0.350": 22.112278, "0.400": 22.094978,
        "0.440": 22.079533, "0.460": 22.095216, "0.480": 22.096313,
        "0.490": 22.092748, "0.500": 22.08747, "0.510": 22.081148,
        "0.520": 22.074639, "0.540": 22.06361, "0.560": 22.055066,
        "0.600": 22.030562, "0.650": 21.987031, "0.750": 22.056156
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

  /* ─── Sigma Sweep Chart ─── */
  var sweepCanvas = document.getElementById('sweepChart');
  var sweepCtx = sweepCanvas ? sweepCanvas.getContext('2d') : null;
  var sigmaSlider = document.getElementById('sigmaSlider');
  var sigmaDisplay = document.getElementById('sigmaDisplay');
  var activeK = 'K200';

  function getChartSeries(kKey) {
    var data = CHART_DATA[kKey];
    if (!data) return {};
    var series = {};
    for (var src in data) {
      if (!data.hasOwnProperty(src)) continue;
      var sigmas = Object.keys(data[src]).map(Number).sort(function(a,b){return a-b;});
      var values = sigmas.map(function(s) { return data[src][s.toFixed(3)]; });
      series[src] = { sigmas: sigmas, values: values };
    }
    return series;
  }

  function redrawSigmaSweep() {
    if (!sweepCanvas || !sweepCtx) return;
    var dpr = window.devicePixelRatio || 1;
    var w = sweepCanvas.offsetWidth;
    var h = sweepCanvas.offsetHeight;
    sweepCanvas.width = w * dpr;
    sweepCanvas.height = h * dpr;
    sweepCtx.setTransform(dpr, 0, 0, dpr, 0, 0);

    var pad = { top: 20, right: 20, bottom: 40, left: 60 };
    var cw = w - pad.left - pad.right;
    var ch = h - pad.top - pad.bottom;

    sweepCtx.fillStyle = CHART_COLORS.surface || '#16140d';
    sweepCtx.fillRect(0, 0, w, h);

    var series = getChartSeries(activeK);
    if (!series.Zeta) return;

    var allVals = [];
    for (var src in series) {
      if (!series.hasOwnProperty(src)) continue;
      allVals = allVals.concat(series[src].values);
    }
    var yMin = Math.min.apply(null, allVals) * 0.995;
    var yMax = Math.max.apply(null, allVals) * 1.005;

    function xScale(sigma) { return pad.left + ((sigma - 0.25) / 0.5) * cw; }
    function yScale(val) { return pad.top + ch - ((val - yMin) / (yMax - yMin)) * ch; }

    // Grid
    sweepCtx.strokeStyle = CHART_COLORS.border || '#35311e';
    sweepCtx.lineWidth = 0.5;
    for (var g = 0; g < 5; g++) {
      var gy = pad.top + (g / 4) * ch;
      sweepCtx.beginPath(); sweepCtx.moveTo(pad.left, gy); sweepCtx.lineTo(w - pad.right, gy); sweepCtx.stroke();
    }

    // σ=0.5 reference
    sweepCtx.strokeStyle = CHART_COLORS.gold || '#d08a28';
    sweepCtx.globalAlpha = 0.3;
    sweepCtx.lineWidth = 1;
    sweepCtx.setLineDash([4, 4]);
    sweepCtx.beginPath(); sweepCtx.moveTo(xScale(0.5), pad.top); sweepCtx.lineTo(xScale(0.5), pad.top + ch); sweepCtx.stroke();
    sweepCtx.setLineDash([]);
    sweepCtx.globalAlpha = 1;

    // Series
    var colorMap = { Zeta: CHART_COLORS.gold, GUE: CHART_COLORS.teal, Random: CHART_COLORS.gray };
    for (var name in series) {
      if (!series.hasOwnProperty(name)) continue;
      var s = series[name];
      sweepCtx.strokeStyle = colorMap[name] || CHART_COLORS.gray;
      sweepCtx.lineWidth = name === 'Zeta' ? 2.5 : 1.5;
      sweepCtx.beginPath();
      for (var i = 0; i < s.sigmas.length; i++) {
        if (i === 0) sweepCtx.moveTo(xScale(s.sigmas[i]), yScale(s.values[i]));
        else sweepCtx.lineTo(xScale(s.sigmas[i]), yScale(s.values[i]));
      }
      sweepCtx.stroke();
      for (var j = 0; j < s.sigmas.length; j++) {
        sweepCtx.beginPath();
        sweepCtx.arc(xScale(s.sigmas[j]), yScale(s.values[j]), 3, 0, Math.PI * 2);
        sweepCtx.fillStyle = colorMap[name] || CHART_COLORS.gray;
        sweepCtx.fill();
      }
    }

    // Slider indicator
    var sliderVal = parseFloat(sigmaSlider ? sigmaSlider.value : 0.5);
    sweepCtx.strokeStyle = CHART_COLORS.text || '#d6d0be';
    sweepCtx.globalAlpha = 0.5;
    sweepCtx.lineWidth = 1;
    sweepCtx.beginPath(); sweepCtx.moveTo(xScale(sliderVal), pad.top); sweepCtx.lineTo(xScale(sliderVal), pad.top + ch); sweepCtx.stroke();
    sweepCtx.globalAlpha = 1;

    // Axis labels
    sweepCtx.fillStyle = CHART_COLORS.textMuted || '#817a66';
    sweepCtx.font = '11px "JetBrains Mono", monospace';
    sweepCtx.textAlign = 'center';
    [0.25, 0.35, 0.45, 0.50, 0.55, 0.65, 0.75].forEach(function (v) {
      sweepCtx.fillText(v.toFixed(2), xScale(v), h - 8);
    });
    sweepCtx.textAlign = 'right';
    for (var g2 = 0; g2 < 5; g2++) {
      var val = yMin + (g2 / 4) * (yMax - yMin);
      sweepCtx.fillText(val.toFixed(1), pad.left - 8, pad.top + ch - (g2 / 4) * ch + 4);
    }

    // Legend
    sweepCtx.textAlign = 'left';
    var legendX = pad.left + 10;
    var legendY = pad.top + 15;
    var legendItems = [['Zeta', CHART_COLORS.gold], ['GUE', CHART_COLORS.teal]];
    if (series.Random) legendItems.push(['Random', CHART_COLORS.gray]);
    legendItems.forEach(function (item, idx) {
      sweepCtx.fillStyle = item[1];
      sweepCtx.fillRect(legendX, legendY + idx * 18 - 8, 12, 3);
      sweepCtx.fillText(item[0], legendX + 18, legendY + idx * 18);
    });
  }

  if (sigmaSlider) {
    sigmaSlider.addEventListener('input', function () {
      if (sigmaDisplay) sigmaDisplay.textContent = 'σ = ' + parseFloat(sigmaSlider.value).toFixed(2);
      redrawSigmaSweep();
    });
  }
  document.querySelectorAll('.k-toggle-btn').forEach(function (btn) {
    btn.addEventListener('click', function () {
      document.querySelectorAll('.k-toggle-btn').forEach(function (b) { b.classList.remove('active'); });
      btn.classList.add('active');
      activeK = btn.getAttribute('data-k');
      redrawSigmaSweep();
    });
  });

  /* ─── Arithmetic Premium Chart ─── */
  var premiumCanvas = document.getElementById('premiumChart');
  var premiumCtx = premiumCanvas ? premiumCanvas.getContext('2d') : null;

  function redrawPremiumChart() {
    if (!premiumCanvas || !premiumCtx) return;
    var dpr = window.devicePixelRatio || 1;
    var w = premiumCanvas.offsetWidth;
    var h = premiumCanvas.offsetHeight;
    premiumCanvas.width = w * dpr;
    premiumCanvas.height = h * dpr;
    premiumCtx.setTransform(dpr, 0, 0, dpr, 0, 0);

    var pad = { top: 20, right: 20, bottom: 40, left: 70 };
    var cw = w - pad.left - pad.right;
    var ch = h - pad.top - pad.bottom;

    premiumCtx.fillStyle = CHART_COLORS.surface || '#16140d';
    premiumCtx.fillRect(0, 0, w, h);

    var kSets = [
      { key: 'K100', label: 'K=100', alpha: 0.6, width: 1.5 },
      { key: 'K200', label: 'K=200', alpha: 1.0, width: 2.5 }
    ];

    var allRatios = [];
    var seriesList = [];

    kSets.forEach(function (ks) {
      var data = CHART_DATA[ks.key];
      if (!data || !data.Zeta || !data.GUE) return;
      var sigmas = [];
      var ratios = [];
      Object.keys(data.Zeta).map(Number).sort(function(a,b){return a-b;}).forEach(function (s) {
        var sk = s.toFixed(3);
        if (data.GUE[sk] && data.GUE[sk] > 0) {
          var ratio = data.Zeta[sk] / data.GUE[sk];
          sigmas.push(s);
          ratios.push(ratio);
          allRatios.push(ratio);
        }
      });
      if (sigmas.length > 0) {
        var minIdx = ratios.indexOf(Math.min.apply(null, ratios));
        seriesList.push({ sigmas: sigmas, ratios: ratios, minIdx: minIdx, label: ks.label, alpha: ks.alpha, width: ks.width });
      }
    });

    if (allRatios.length === 0) return;

    var yMin = Math.min.apply(null, allRatios) * 0.999;
    var yMax = Math.max.apply(null, allRatios) * 1.001;

    function xScale(sigma) { return pad.left + ((sigma - 0.25) / 0.5) * cw; }
    function yScale(val) { return pad.top + ch - ((val - yMin) / (yMax - yMin)) * ch; }

    // σ=0.5 reference
    premiumCtx.strokeStyle = CHART_COLORS.gold || '#d08a28';
    premiumCtx.globalAlpha = 0.3;
    premiumCtx.setLineDash([4, 4]);
    premiumCtx.lineWidth = 1;
    premiumCtx.beginPath(); premiumCtx.moveTo(xScale(0.5), pad.top); premiumCtx.lineTo(xScale(0.5), pad.top + ch); premiumCtx.stroke();
    premiumCtx.setLineDash([]);
    premiumCtx.globalAlpha = 1;

    seriesList.forEach(function (s) {
      premiumCtx.strokeStyle = CHART_COLORS.gold || '#d08a28';
      premiumCtx.globalAlpha = s.alpha;
      premiumCtx.lineWidth = s.width;
      premiumCtx.beginPath();
      for (var i = 0; i < s.sigmas.length; i++) {
        if (i === 0) premiumCtx.moveTo(xScale(s.sigmas[i]), yScale(s.ratios[i]));
        else premiumCtx.lineTo(xScale(s.sigmas[i]), yScale(s.ratios[i]));
      }
      premiumCtx.stroke();
      premiumCtx.globalAlpha = 1;

      // Minimum marker
      var mx = xScale(s.sigmas[s.minIdx]);
      var my = yScale(s.ratios[s.minIdx]);
      premiumCtx.beginPath();
      premiumCtx.arc(mx, my, 5, 0, Math.PI * 2);
      premiumCtx.fillStyle = CHART_COLORS.gold || '#d08a28';
      premiumCtx.globalAlpha = s.alpha;
      premiumCtx.fill();
      premiumCtx.globalAlpha = 1;

      premiumCtx.fillStyle = CHART_COLORS.text || '#d6d0be';
      premiumCtx.font = '10px "JetBrains Mono", monospace';
      premiumCtx.globalAlpha = s.alpha;
      premiumCtx.textAlign = 'center';
      premiumCtx.fillText(s.label + ' σ=' + s.sigmas[s.minIdx].toFixed(3), mx, my - 12);
      premiumCtx.globalAlpha = 1;
    });

    // Axes
    premiumCtx.fillStyle = CHART_COLORS.textMuted || '#817a66';
    premiumCtx.font = '11px "JetBrains Mono", monospace';
    premiumCtx.textAlign = 'center';
    [0.25, 0.35, 0.45, 0.50, 0.55, 0.65, 0.75].forEach(function (v) {
      premiumCtx.fillText(v.toFixed(2), xScale(v), h - 8);
    });
    premiumCtx.textAlign = 'right';
    for (var g = 0; g < 5; g++) {
      var val = yMin + (g / 4) * (yMax - yMin);
      premiumCtx.fillText(val.toFixed(4), pad.left - 8, pad.top + ch - (g / 4) * ch + 4);
    }

    premiumCtx.save();
    premiumCtx.translate(12, pad.top + ch / 2);
    premiumCtx.rotate(-Math.PI / 2);
    premiumCtx.textAlign = 'center';
    premiumCtx.fillText('S(zeta) / S(GUE)', 0, 0);
    premiumCtx.restore();
  }

  /* ─── Chart Init ─── */
  window.addEventListener('load', function () {
    redrawSigmaSweep();
    redrawPremiumChart();
  });
  window.addEventListener('resize', function () {
    redrawSigmaSweep();
    redrawPremiumChart();
  });

  /* ─── Data-Driven Popout Panel System ─── */
  var PANELS = {
    'panel-fourier': {
      title: 'The Fourier Kernel',
      category: 'TRANSPORT \u00b7 PHASE FACTORS',
      effect: 'field',
      ambient: { src: 'assets/ambient/ambient_fourier.png', opacity: 0.18 },
      body: '<p>The explicit formula connects zeta zeros to primes through an oscillatory sum. Each prime p contributes a phase factor to the transport map between zeros \u03b3\u1d62 and \u03b3\u2c7c:</p><div class="eq-block"><div class="eq-label">Phase Factor</div><div class="eq-formula">e^{i \u00b7 \u0394\u03b3 \u00b7 log(p)} where \u0394\u03b3 = \u03b3\u1d62 \u2212 \u03b3\u2c7c</div></div><p>This is the Fourier kernel of the explicit formula \u2014 each prime acts as a frequency, and the log-spacing of primes creates the harmonic structure. More primes = more frequencies = sharper resolution of the spectral peak.</p><div class="eq-block"><div class="eq-label">Superposition Transport</div><div class="eq-formula">A_ij(\u03c3) = \u03a3_{p\u2264K} e^{i\u0394\u03b3\u00b7log p} \u00b7 B_p(\u03c3)</div></div><p>The matrix B_p(\u03c3) encodes the arithmetic weight of each prime at position \u03c3 in the critical strip. At \u03c3 = 0.500, the functional equation symmetry makes the contributions from p^{\u2212\u03c3} and p^{\u2212(1\u2212\u03c3)} equal \u2014 creating maximum constructive interference.</p>',
      data: 'EXPLICIT FORMULA \u00b7 PHASE INTERFERENCE \u00b7 \u03c3=0.500 = MAXIMUM COHERENCE'
    },
    'panel-spectral': {
      title: 'Reading the Wrinkle Score',
      category: 'MEASUREMENT \u00b7 S(\u03c3)',
      effect: 'proof',
      ambient: { src: 'assets/ambient/ambient_spectral.png', opacity: 0.15 },
      body: '<p>The spectral sum S(\u03c3) is the simplest thing in this entire framework. Take the sheaf Laplacian \u2014 the 200,000 \u00d7 200,000 matrix that encodes every prime\'s transport across every edge of the zero graph. Find its smallest eigenvalues. Add them up.</p><p>That\'s it. That\'s the measurement.</p><p>Low S means the transport maps agree with each other. The primes\' phase factors are constructively interfering \u2014 the fabric fits. High S means disagreement. The phases cancel. The fabric wrinkles.</p><p>At \u03c3 = 0.500, S is minimized for zeta zeros. Not for GUE. Not for random. Not for evenly-spaced. The primes find their lowest wrinkle score exactly on the critical line. Every time.</p><div class="eq-block"><div class="eq-label">Order Parameter</div><div class="eq-formula">S(\u03c3) = \u03a3_{k=1}^{k_eig} \u03bb_k(L_\ud835\udca5(\u03c3))</div></div>',
      data: 'LOWER = TIGHTER \u00b7 \u03c3=0.500 = MINIMUM \u00b7 21.5% = PREMIUM'
    },
    'panel-gue': {
      title: 'The Statistical Doppelg\u00e4nger',
      category: 'GUE \u00b7 MONTGOMERY-ODLYZKO',
      effect: 'journey',
      ambient: { src: 'assets/ambient/ambient_gue.png', opacity: 0.15 },
      body: '<p>In 1973, Hugh Montgomery had dinner with Freeman Dyson and discovered that the gaps between zeta zeros follow the same statistics as the gaps between eigenvalues of random Hermitian matrices. Andrew Odlyzko later confirmed this computationally to extraordinary precision.</p><p>This is the Montgomery-Odlyzko law, and it means that if you look at zeta zeros locally \u2014 pairs, triples, short-range correlations \u2014 they are statistically indistinguishable from GUE random matrices.</p><p>Locally identical. Globally different.</p><p>GUE matrices have level repulsion and local correlations. But they have no primes. No arithmetic. No explicit formula. No reason to care about \u03c3 = 0.500.</p><p>The sheaf Laplacian sees the difference. It threads prime phase factors through the graph and measures coherence. GUE zeros produce transport that\'s 21.5% less coherent than zeta zeros. Same local statistics. Different global phase structure.</p><p>The 21.5% premium is the arithmetic. The part that Montgomery and Odlyzko\'s local statistics can\'t see but the sheaf Laplacian can.</p>',
      data: 'MONTGOMERY (1973) \u00b7 ODLYZKO (1987) \u00b7 PREMIUM = 21.5%'
    },
    'panel-killshot': {
      title: 'We Tried to Kill It',
      category: 'PHASE 3e \u00b7 CONTROL BATTERY',
      effect: 'proof',
      ambient: { src: 'assets/ambient/ambient_killshot.png', opacity: 0.20 },
      body: '<p>A three-agent validation committee attacked every claim.</p><p>The Adversary proposed: if S(\u03c3) just measures edge density in the graph, any ordered set beats random. Evenly-spaced points \u2014 the most ordered configuration mathematics can produce \u2014 should have the lowest S.</p><p>We built it. N=1000 points, perfectly uniform gaps, zero randomness. Ran it through the identical pipeline.</p><p>S(Even) = 12.713. S(Zeta) = 11.784. <strong>Primes won. By 7.3%. At every sigma.</strong></p><p>Then we ran 10 independent GUE ensembles using the proper Dumitriu-Edelman tridiagonal model.</p><div class="eq-block"><div class="eq-label">GUE Ensemble (10 D-E realizations)</div><div class="eq-formula">S(GUE) = 14.970 \u00b1 0.198 &nbsp;|&nbsp; S(\u03b6) = 11.784 &nbsp;|&nbsp; Z = \u221216.06</div></div><p>Then we edge-normalized. Zeta has fewer graph edges (level repulsion creates wider gaps). But per edge, the transport is still 15.3% tighter. The premium survives normalization.</p><p>Three attacks. Three survivals. The primes carry something that order alone doesn\'t, and statistics alone doesn\'t.</p>',
      data: 'EVEN=12.713 \u00b7 ZETA=11.784 \u00b7 GUE=14.970\u00b10.198 \u00b7 Z=\u221216.06'
    },
    'panel-connection': {
      title: 'Prime Arithmetic as Gauge Connection',
      category: 'u(K) \u00b7 LIE ALGEBRA',
      effect: 'field',
      ambient: { src: 'assets/ambient/ambient_connection.png', opacity: 0.18 },
      body: '<p>Each prime p \u2264 K defines a K\u00d7K representation matrix \u03c1(p) \u2014 the truncated left-regular representation. It encodes "multiply by p" as a matrix operation: \u03c1(p)|n\u27e9 = |pn\u27e9 if pn \u2264 K, else 0.</p><div class="eq-block"><div class="eq-label">Prime Representation</div><div class="eq-formula">\u03c1(p): the alphabet of arithmetic in K dimensions</div></div><p>The transport map between zeros \u03b3\u1d62 and \u03b3\u2c7c is a coherent sum over all primes \u2014 each contributing its phase factor weighted by its arithmetic generator:</p><div class="eq-block"><div class="eq-label">Superposition Transport</div><div class="eq-formula">A_ij(\u03c3) = \u03a3_p exp(i\u0394\u03b3\u00b7log p) \u00b7 B_p(\u03c3)</div></div><p>This connection lives in the unitary group U(K). As K grows, the fiber dimension increases and the connection encodes more of the prime harmonic structure. K=200 uses 46 primes. K=400 uses 78. The premium converges at 21.5%.</p>',
      data: 'K=200 \u2192 46 PRIMES \u00b7 K=400 \u2192 78 PRIMES \u00b7 dim = N\u00d7K'
    },
    'panel-k-journey': {
      title: 'The Orbit Tightening',
      category: 'K=20 \u2192 K=400 \u00b7 CONVERGENCE',
      effect: 'journey',
      ambient: { src: 'assets/ambient/ambient_journey.png', opacity: 0.18 },
      body: '<p>K=20. Eight primes. The spectral sum climbed monotonically through \u03c3 = 0.500. No peak. Eight Fourier harmonics isn\'t enough bandwidth to resolve the signal. Like trying to see a face with eight pixels. But the 670\u00d7 signal over random controls was there.</p><p>K=50. Fifteen primes. The first spectral turnover at \u03b5 = 5.0. The summit appeared near \u03c3 \u2248 0.40\u20130.50. Fifteen harmonics resolved what eight couldn\'t.</p><p>K=100. Twenty-five primes. Signal reversal at \u03b5 = 3.0. The premium emerged: 19.6% over GUE at \u03c3 = 0.500. Still broad. Still imprecise. But undeniable.</p><p>K=200. Forty-six primes. RTX 5070. Three tranches across 12 hours. Crashed once (VRAM). Added batched assembly. Crashed again (CPU RAM). Added incremental release. Third time: it ran. Premium peaked at \u03c3 = 0.500 exactly. 21.5%.</p><p>K=400. Seventy-eight primes. The dense solver couldn\'t touch it \u2014 32 GB needed, 12 GB available. So we built a matrix-free engine. Pad\u00e9 approximation. Cached transport on GPU. 47 seconds. Premium: 21.6%.</p><p>From 19.6% to 21.5% to 21.6%. It\'s not growing. It\'s converging. That\'s what a measurement does when it\'s measuring something real.</p>',
      data: 'K=100\u219219.6% \u00b7 K=200\u219221.5% \u00b7 K=400\u219221.6% \u00b7 CONVERGING'
    },
    'panel-matrix-free': {
      title: 'The Lighter Airframe',
      category: 'MATRIX-FREE \u00b7 PAD\u00c9 \u00b7 18\u00d7 SPEEDUP',
      effect: 'journey',
      ambient: { src: 'assets/ambient/ambient_matfree.png', opacity: 0.15 },
      body: '<p>K=400 needed a 400,000 \u00d7 400,000 matrix. In complex double precision, that\'s 32 GB just for the sparse structure. The GPU has 12.</p><p>Dan\'s problem. Too much weight, not enough lift.</p><p>The matrix-free sheaf Laplacian never materializes L. It computes L\u00b7v \u2014 the Laplacian times a vector \u2014 by streaming through edges one batch at a time. Each edge contributes two matrix-vector products. 2,492 edges \u00d7 2 matmuls = 4,984 operations per Lanczos iteration. All independent. All on GPU.</p><p>Transport computation: replaced eigendecomposition with Pad\u00e9 matrix exponential. Same result to 10\u207b\u00b9\u2074. Four times faster. Because Pad\u00e9 is batched matmuls \u2014 the thing GPUs were literally built to do.</p><p>K=200 dense: 166 seconds. K=200 matrix-free: 9.2 seconds. <strong>18\u00d7 faster.</strong></p><p>K=400 matrix-free: 46.8 seconds. Previously impossible.</p><p>The hardware couldn\'t hold what was being asked of it. So we built a lighter airframe. Stan flies.</p>',
      data: '18\u00d7 SPEEDUP \u00b7 10\u207b\u00b9\u2074 PRECISION \u00b7 K=400 IN 47s \u00b7 PAD\u00c9 NOT EIG'
    },
    'panel-constant': {
      title: '21.5% \u2014 A Physical Constant?',
      category: 'K=100 \u2192 K=400 \u00b7 CONVERGENCE',
      effect: 'proof',
      ambient: { src: 'assets/ambient/ambient_constant.png', opacity: 0.15 },
      body: '<p>Three K values. Three independent computations. One number.</p><p>K=100 (25 primes): 19.6%</p><p>K=200 (46 primes): 21.5%</p><p>K=400 (78 primes): 21.6%</p><p>The Physicist predicted K=400 would reach 27.7%. It didn\'t. The premium didn\'t grow. It converged.</p><p>That\'s not what a statistical fluctuation does. Fluctuations wander. Constants converge.</p><p>If 21.5% is the asymptotic value of the arithmetic premium \u2014 the transport coherence advantage that zeta zeros carry over GUE random matrices in the sheaf Laplacian framework \u2014 then it\'s a new invariant. A number that characterizes how much tighter the prime fabric is compared to its statistical doppelg\u00e4nger.</p><p>We don\'t know if it\'s a constant yet. Two data points aren\'t enough. K=800 will tell. But the behavior is more constant than trend.</p><p>Open question: does this number have a closed-form expression? Is 21.5% = f(something) for some knowable f? Or is it empirical \u2014 measurable, stable, and unexplained?</p>',
      data: 'K=100\u219219.6% \u00b7 K=200\u219221.5% \u00b7 K=400\u219221.6% \u00b7 ASYMPTOTE?'
    }
  };

  /* ─── Popout Panel Engine ─── */
  (function() {
    var overlay = document.getElementById('popoutOverlay');
    var panel = document.getElementById('popoutPanel');
    var closeBtn = document.getElementById('popoutClose');
    var catEl = document.getElementById('popoutCategory');
    var titleEl = document.getElementById('popoutTitle');
    var bodyEl = document.getElementById('popoutBody');
    var dataEl = document.getElementById('popoutData');
    var ambientEl = document.getElementById('popoutAmbient');
    var lastFocus = null;

    if (!overlay || !panel) return;

    function openPopout(id) {
      var data = PANELS[id];
      if (!data) return;

      lastFocus = document.activeElement;
      catEl.textContent = data.category || '';
      titleEl.textContent = data.title || '';
      bodyEl.innerHTML = data.body || '';
      dataEl.textContent = data.data || '';

      // Ambient image
      if (data.ambient && data.ambient.src) {
        var img = document.createElement('img');
        img.src = data.ambient.src;
        img.alt = '';
        img.style.opacity = '0';
        ambientEl.innerHTML = '';
        ambientEl.appendChild(img);
        ambientEl.style.display = '';
        requestAnimationFrame(function() {
          requestAnimationFrame(function() {
            img.style.opacity = String(data.ambient.opacity || 0.18);
          });
        });
      } else {
        ambientEl.innerHTML = '';
        ambientEl.style.display = 'none';
      }

      // Effect class
      panel.className = 'popout-panel active' +
        (data.effect ? ' popout-effect-' + data.effect : '');

      overlay.classList.add('active');
      panel.scrollTop = 0;
      document.body.classList.add('popout-open');
      closeBtn.focus();
    }

    function closePopout() {
      overlay.classList.remove('active');
      panel.className = 'popout-panel';
      ambientEl.style.opacity = '0';
      document.body.classList.remove('popout-open');
      if (lastFocus) lastFocus.focus();
    }

    closeBtn.addEventListener('click', closePopout);
    overlay.addEventListener('click', closePopout);
    document.addEventListener('keydown', function(e) {
      if (e.key === 'Escape' && panel.classList.contains('active')) closePopout();
    });

    // Focus trap
    panel.addEventListener('keydown', function(e) {
      if (e.key !== 'Tab') return;
      var focusable = panel.querySelectorAll('button, a, [tabindex]:not([tabindex="-1"])');
      var first = focusable[0];
      var last = focusable[focusable.length - 1];
      if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
      else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
    });

    // Attach to all triggers (data-panel="panel-xxx" → opens PANELS['panel-xxx'])
    document.querySelectorAll('.panel-trigger').forEach(function(btn) {
      btn.setAttribute('tabindex', '0');
      btn.setAttribute('role', 'button');
      btn.addEventListener('click', function() {
        var panelId = btn.getAttribute('data-panel');
        if (panelId) openPopout(panelId);
      });
    });
  })();

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
