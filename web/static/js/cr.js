/**
 * CivicRoad shared client library (window.CR)
 * - safe HTML escaping, formatting, fetch wrapper with real error messages
 * - toasts, debounce, count-up, visibility-aware polling
 * - responsive sidebar drawer + live server health telemetry
 */
(() => {
  'use strict';

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

  // Escape anything that ends up inside innerHTML (names/addresses are user-controlled)
  const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));

  const isNum = (n) => n !== null && n !== undefined && n !== '' && Number.isFinite(+n);
  const fmt = {
    num: (n) => (isNum(n) ? Number(n).toLocaleString() : '—'),
    conf: (n) => (isNum(n) ? (+n).toFixed(3) : '—'),
    coord: (n) => (isNum(n) ? (+n).toFixed(4) : '—'),
    date: (ts) => {
      if (!ts) return '—';
      const d = new Date(ts);
      return Number.isNaN(+d) ? String(ts) : d.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' });
    },
    uptime: (s) => {
      if (!isNum(s)) return '—';
      const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
      return h ? `${h}h ${m}m` : `${m}m ${Math.floor(s % 60)}s`;
    },
  };

  async function api(path, opts = {}) {
    const init = { ...opts, headers: { ...(opts.body ? { 'Content-Type': 'application/json' } : {}), ...(opts.headers || {}) } };
    const res = await fetch(path, init);
    let data = null;
    try { data = await res.json(); } catch { /* non-JSON body */ }
    if (!res.ok) throw new Error((data && data.error) || `Request failed (HTTP ${res.status})`);
    return data;
  }

  function toast(message, type = 'info') {
    let root = $('#toast-root');
    if (!root) {
      root = document.createElement('div');
      root.id = 'toast-root';
      root.className = 'fixed bottom-4 right-4 left-4 sm:left-auto z-[100] flex flex-col gap-2 items-end pointer-events-none';
      document.body.appendChild(root);
    }
    const colors = { success: 'border-emerald-300 text-emerald-800', error: 'border-red-300 text-red-700', info: 'border-slate-300 text-slate-700' };
    const icon = { success: 'check_circle', error: 'error', info: 'info' }[type] || 'info';
    const el = document.createElement('div');
    el.setAttribute('role', 'status');
    el.className = `pointer-events-auto max-w-sm w-full sm:w-auto bg-white border ${colors[type] || colors.info} rounded-xl shadow-lg px-4 py-2.5 text-xs font-medium flex items-center gap-2 card-enter`;
    el.innerHTML = `<span class="material-symbols-outlined text-[18px]">${icon}</span><span>${esc(message)}</span>`;
    root.appendChild(el);
    setTimeout(() => { el.style.opacity = '0'; el.style.transition = 'opacity .3s'; setTimeout(() => el.remove(), 300); }, 3200);
  }

  const debounce = (fn, ms = 200) => {
    let t;
    return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
  };

  const reduceMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // Animate a number from its current value to `to`
  function countUp(el, to, ms = 600) {
    if (!el) return;
    const target = Number(to) || 0;
    const from = Number((el.dataset.v ?? '0')) || 0;
    el.dataset.v = target;
    if (reduceMotion() || from === target) { el.textContent = fmt.num(target); return; }
    const t0 = performance.now();
    const step = (t) => {
      const p = Math.min(1, (t - t0) / ms);
      el.textContent = fmt.num(Math.round(from + (target - from) * (1 - Math.pow(1 - p, 3))));
      if (p < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  // Run fn now and every `ms`, pausing while the tab is hidden; returns a controller
  function poll(fn, ms = 20000) {
    let timer = null, stopped = false;
    const tick = async () => {
      clearTimeout(timer);
      if (stopped) return;
      if (!document.hidden) { try { await fn(); } catch { /* callers handle their own errors */ } }
      timer = setTimeout(tick, ms);
    };
    document.addEventListener('visibilitychange', () => { if (!document.hidden) tick(); });
    tick();
    return { refresh: tick, stop: () => { stopped = true; clearTimeout(timer); } };
  }

  // ── Responsive drawer ────────────────────────────────────────────────────────
  function initDrawer() {
    const btn = $('#nav-toggle'), side = $('#sidebar'), scrim = $('#scrim');
    if (!btn || !side) return;
    const set = (open) => {
      side.classList.toggle('-translate-x-full', !open);
      scrim.classList.toggle('hidden', !open);
      btn.setAttribute('aria-expanded', String(open));
      document.body.classList.toggle('overflow-hidden', open && window.innerWidth < 1024);
    };
    btn.addEventListener('click', () => set(side.classList.contains('-translate-x-full')));
    scrim.addEventListener('click', () => set(false));
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape') set(false); });
    window.addEventListener('resize', () => { if (window.innerWidth >= 1024) set(false); });
    const active = $('#main-nav .nav-active');
    if (active) active.scrollIntoView({ inline: 'center', block: 'nearest' });
  }

  // ── Live telemetry from /api/health (real values only) ───────────────────────
  const setText = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
  async function refreshHealth() {
    const dot = $('#health-dot'), txt = $('#health-text');
    try {
      const h = await api('/api/health');
      if (dot) dot.className = 'w-2 h-2 rounded-full bg-emerald-500 animate-pulse';
      if (txt) txt.textContent = 'Server online';
      setText('t-model', h.model.file);
      setText('t-classes', (h.model.classes || []).join(', ') || '—');
      setText('t-device', h.model.device);
      setText('t-latency', h.model.last_inference_ms == null ? 'no inference yet' : `${h.model.last_inference_ms} ms (last image)`);
      setText('t-db', `${h.db.file} · ${fmt.num(h.db.records)} records`);
      setText('t-journal', `SQLite ${h.db.sqlite} · ${h.db.journal_mode} journal`);
      setText('t-map', `CARTO key: ${h.basemap.key_source}`);
      setText('t-uptime', `Uptime ${fmt.uptime(h.uptime_sec)}`);
    } catch {
      if (dot) dot.className = 'w-2 h-2 rounded-full bg-red-500';
      if (txt) txt.textContent = 'Server offline';
      setText('t-uptime', 'Cannot reach server');
    }
  }

  // Broken photo → generic sample, never a dead image icon
  document.addEventListener('error', (e) => {
    const t = e.target;
    if (t && t.tagName === 'IMG' && !t.dataset.fb) {
      t.dataset.fb = '1';
      t.src = '/sample_images/sample_city_pothole_annotated.jpg';
    }
  }, true);

  document.addEventListener('DOMContentLoaded', () => {
    initDrawer();
    poll(refreshHealth, 30000);
  });

  window.CR = { $, $$, esc, fmt, api, toast, debounce, countUp, poll, refreshHealth };
})();
