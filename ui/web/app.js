// Dubbing Pipeline Inspector — classic-script React + htm (no build step).
// Vendored UMD builds expose window.React / window.ReactDOM / window.htm.

const React                                                 = window.React;
const { useEffect, useMemo, useRef, useState, useCallback } = React;
const createRoot                                            = window.ReactDOM.createRoot;
const html                                                  = window.htm.bind(React.createElement);

// --- utilities --------------------------------------------------------------

const fmtTime = (s) => {
  if (s == null || isNaN(s)) return '0:00';
  const total = Math.max(0, Math.floor(s));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const sec = total % 60;
  return h > 0
    ? `${h}:${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`
    : `${m}:${String(sec).padStart(2, '0')}`;
};

const fmtBytes = (n) => {
  if (n == null) return '-';
  if (n < 1024) return `${n} B`;
  const u = ['KB', 'MB', 'GB', 'TB'];
  let v = n / 1024, i = 0;
  while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; }
  return `${v < 10 ? v.toFixed(1) : Math.round(v)} ${u[i]}`;
};

const useFetch = (url, deps = []) => {
  const [data, setData] = useState(null);
  const [err, setErr]   = useState(null);
  const [loading, setLoading] = useState(!!url);
  useEffect(() => {
    if (!url) { setData(null); return; }
    let abort = false;
    setLoading(true); setErr(null);
    fetch(url).then(r => {
      if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
      return r.json();
    }).then(d => { if (!abort) { setData(d); setLoading(false); } })
      .catch(e => { if (!abort) { setErr(e); setLoading(false); } });
    return () => { abort = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [url, ...deps]);
  return { data, err, loading };
};

const copyUrl = (u) =>
  navigator.clipboard?.writeText(u).catch(() => {});

const StateBadge = ({ state }) => {
  const map = {
    done:    'bg-emerald-500/15 text-emerald-300 border-emerald-500/30',
    ok:      'bg-emerald-500/15 text-emerald-300 border-emerald-500/30',
    running: 'bg-amber-500/15  text-amber-300  border-amber-500/30',
    failed:  'bg-rose-500/15   text-rose-300   border-rose-500/30',
    unknown: 'bg-slate-700/50  text-slate-300  border-slate-600',
  };
  const cls = map[state] || map.unknown;
  return html`<span class="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-md border text-[11px] font-medium ${cls}">
    <span class="w-1.5 h-1.5 rounded-full bg-current"></span>${state || 'unknown'}
  </span>`;
};

const ROLE_ORDER = ['original', 'lipsync', 'final', 'per_shot', 'preview', 'other'];
const ROLE_STYLE = {
  original: 'ring-sky-500/40 bg-sky-500/5',
  final:    'ring-emerald-500/40 bg-emerald-500/5',
  lipsync:  'ring-violet-500/40 bg-violet-500/5',
  per_shot: 'ring-indigo-500/40 bg-indigo-500/5',
  preview:  'ring-amber-500/40 bg-amber-500/5',
  other:    'ring-slate-500/40 bg-slate-500/5',
};
const ROLE_LABEL = {
  original: 'Original clip',
  final:    'Final / dubbed',
  lipsync:  'Lipsync',
  per_shot: 'Per-shot render',
  preview:  'Preview',
  other:    'Other',
};

// --- sidebar ----------------------------------------------------------------

const Sidebar = ({ jobs, activeJobId, onSelect }) => {
  const [q, setQ] = useState('');
  const filtered = useMemo(() => {
    if (!q) return jobs;
    const s = q.toLowerCase();
    return jobs.filter(j =>
      j.job_id.toLowerCase().includes(s) ||
      (j.slug || '').toLowerCase().includes(s) ||
      (j.config_slug || '').toLowerCase().includes(s)
    );
  }, [jobs, q]);
  return html`
    <aside class="w-96 shrink-0 border-r border-line/70 bg-panel/70 backdrop-blur-sm overflow-y-auto flex flex-col">
      <div class="px-5 py-5 border-b border-line/70 sticky top-0 bg-panel/95 backdrop-blur z-10">
        <div class="flex items-center gap-2 mb-3">
          <div class="w-8 h-8 rounded-lg bg-gradient-to-br from-accent to-accent2 flex items-center justify-center font-bold text-white text-sm">D</div>
          <div class="min-w-0">
            <div class="text-sm font-semibold leading-tight">Dubbing Inspector</div>
            <div class="text-[11px] text-slate-400">${jobs.length} job${jobs.length === 1 ? '' : 's'} · newest first</div>
          </div>
        </div>
        <input
          value=${q}
          placeholder="Search jobs…"
          onInput=${(e) => setQ(e.target.value)}
          class="w-full bg-slate-900/60 border border-line/60 rounded-md px-3 py-1.5 text-sm text-slate-200 placeholder:text-slate-500 focus:outline-none focus:border-accent" />
      </div>
      <div class="p-3 space-y-2 flex-1">
        ${filtered.map(j => {
          const active = activeJobId === j.job_id;
          return html`
            <button key=${j.job_id}
              onClick=${() => onSelect(j.job_id)}
              class="w-full text-left px-3 py-2.5 rounded-xl border transition-colors ${active
                ? 'bg-accent/15 border-accent/50 ring-1 ring-accent/40'
                : 'bg-slate-900/40 border-line/60 hover:bg-slate-800/50 hover:border-line'}">
              <div class="flex items-center justify-between gap-2">
                <div class="font-mono text-[12px] truncate text-slate-100">${j.slug || j.job_id}</div>
                <${StateBadge} state=${j.state} />
              </div>
              <div class="text-[10px] text-slate-500 font-mono truncate mt-1">${j.config_slug || ''}</div>
              <div class="flex items-center gap-2 mt-1.5 text-[10px] text-slate-400">
                ${j.timestamp && html`<span>${j.timestamp}</span>`}
                ${j.lipsync_model && html`<span class="px-1 py-[1px] rounded bg-violet-500/10 text-violet-300">${j.lipsync_model}</span>`}
                ${j.language && html`<span class="px-1 py-[1px] rounded bg-sky-500/10 text-sky-300 uppercase">${j.language}</span>`}
              </div>
            </button>
          `;
        })}
        ${filtered.length === 0 && html`
          <div class="text-slate-500 text-sm px-3 py-8 text-center">
            ${jobs.length === 0
              ? html`No jobs under <code class="text-xs">data/jobs/</code>.`
              : html`No jobs match <strong>${q}</strong>.`}
          </div>`}
      </div>
    </aside>
  `;
};

// --- Overview tab -----------------------------------------------------------

const PhaseRow = ({ p, max }) => {
  const w = max ? (p.wall_s / max) * 100 : 0;
  return html`
    <div class="flex items-center gap-3 py-1.5 border-b border-line/40 last:border-b-0">
      <div class="w-5 h-5 rounded-full flex items-center justify-center shrink-0
                  ${p.done ? 'bg-emerald-500/20 text-emerald-300' : 'bg-slate-700 text-slate-400'}">
        ${p.done
          ? html`<svg class="w-3 h-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3"><path d="m5 12 5 5L20 7"/></svg>`
          : html`<span class="w-2 h-2 rounded-full bg-current"></span>`}
      </div>
      <div class="font-mono text-xs text-slate-200 w-56 shrink-0 truncate">${p.name}</div>
      <div class="flex-1 h-1.5 bg-slate-800/70 rounded-full overflow-hidden">
        <div class="h-full bg-gradient-to-r from-accent to-accent2 rounded-full"
             style=${{ width: `${w}%` }}></div>
      </div>
      <div class="font-mono text-xs text-slate-400 w-14 text-right">${p.wall_s.toFixed(1)}s</div>
    </div>
  `;
};

const InfoGrid = ({ cfg }) => {
  if (!cfg) return null;
  const keys = Object.keys(cfg).filter(k => cfg[k] !== null && cfg[k] !== '');
  return html`
    <div class="grid grid-cols-2 lg:grid-cols-3 gap-2 mt-3">
      ${keys.map(k => {
        const v = cfg[k];
        const str = Array.isArray(v) ? v.join(' – ')
                   : typeof v === 'object' ? JSON.stringify(v)
                   : String(v);
        return html`
          <div key=${k} class="rounded-lg bg-slate-900/50 border border-line/50 px-3 py-2">
            <div class="text-[10px] uppercase tracking-wider text-slate-500">${k}</div>
            <div class="font-mono text-xs text-slate-200 truncate" title=${str}>${str}</div>
          </div>`;
      })}
    </div>
  `;
};

const Overview = ({ job }) => {
  const phases = job.status?.phases || [];
  const maxPhase = phases.reduce((m, p) => Math.max(m, p.wall_s), 0);
  const totalPhase = phases.reduce((m, p) => m + p.wall_s, 0);
  const verify = job.verify || {};
  const vmetrics = Object.entries(verify).filter(([k, v]) =>
    typeof v === 'number' || typeof v === 'boolean' || typeof v === 'string');
  const finalMp4 = job.videos.find(v => v.role === 'final');

  return html`
    <div class="h-full overflow-y-auto">
      <div class="max-w-5xl mx-auto p-6 space-y-5">
        <!-- summary header -->
        <div class="rounded-2xl border border-line/60 bg-panel/40 p-5">
          <div class="flex flex-wrap items-start gap-4">
            <div class="flex-1 min-w-0">
              <div class="text-xs uppercase tracking-widest text-slate-500">Job</div>
              <div class="font-mono text-base text-slate-100 break-all mt-0.5">${job.job_id}</div>
              <div class="flex items-center gap-3 mt-2 text-xs text-slate-400">
                <${StateBadge} state=${job.state} />
                ${job.timestamp && html`<span>${job.timestamp}</span>`}
                <span>${job.n_files} files · ${fmtBytes(job.total_bytes)}</span>
              </div>
            </div>
            ${finalMp4 && html`
              <a href=${finalMp4.url} download
                 class="px-4 py-2 rounded-lg bg-indigo-500 hover:bg-indigo-400 text-white text-sm font-medium transition-colors inline-flex items-center gap-2">
                <svg class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="m7 10 5 5 5-5"/><path d="M12 15V3"/></svg>
                Download final
              </a>`}
          </div>
          ${job.config && html`
            <div class="mt-5">
              <div class="text-xs uppercase tracking-widest text-slate-500 mb-1">Config</div>
              <${InfoGrid} cfg=${job.config} />
            </div>`}
        </div>

        <!-- phases -->
        ${phases.length > 0 && html`
          <div class="rounded-2xl border border-line/60 bg-panel/40 p-5">
            <div class="flex items-center justify-between mb-3">
              <h2 class="text-sm font-semibold">Pipeline phases</h2>
              <div class="text-xs text-slate-400 font-mono">${totalPhase.toFixed(1)}s total</div>
            </div>
            <div class="divide-y divide-line/40">
              ${phases.map((p, i) => html`<${PhaseRow} key=${i} p=${p} max=${maxPhase} />`)}
            </div>
          </div>`}

        <!-- verify metrics -->
        ${vmetrics.length > 0 && html`
          <div class="rounded-2xl border border-line/60 bg-panel/40 p-5">
            <h2 class="text-sm font-semibold mb-3">Verification</h2>
            <div class="grid grid-cols-2 md:grid-cols-4 gap-2">
              ${vmetrics.map(([k, v]) => html`
                <div key=${k} class="rounded-lg bg-slate-900/50 border border-line/50 px-3 py-2">
                  <div class="text-[10px] uppercase tracking-wider text-slate-500">${k}</div>
                  <div class="font-mono text-sm text-slate-200 truncate" title=${String(v)}>
                    ${typeof v === 'boolean' ? (v ? '✓' : '✗') : String(v)}
                  </div>
                </div>`)}
            </div>
            ${verify.peak_vram_mib && html`
              <div class="mt-4">
                <div class="text-xs uppercase tracking-widest text-slate-500 mb-1">Peak VRAM (MiB)</div>
                <div class="flex gap-4 text-sm font-mono">
                  ${Object.entries(verify.peak_vram_mib).map(([gpu, mib]) => html`
                    <div key=${gpu} class="flex items-center gap-2">
                      <span class="text-slate-500">GPU ${gpu}</span>
                      <span class="text-slate-100">${mib}</span>
                    </div>`)}
                </div>
              </div>`}
          </div>`}

        <!-- stage summary -->
        ${job.stages.length > 0 && html`
          <div class="rounded-2xl border border-line/60 bg-panel/40 p-5">
            <h2 class="text-sm font-semibold mb-3">Artifacts per stage</h2>
            <div class="grid grid-cols-2 md:grid-cols-4 gap-2">
              ${job.stages.map(s => html`
                <div key=${s.name} class="rounded-lg bg-slate-900/50 border border-line/50 px-3 py-2">
                  <div class="font-mono text-xs text-slate-300">${s.name}/</div>
                  <div class="text-[11px] text-slate-400 mt-0.5">${s.count} file${s.count === 1 ? '' : 's'} · ${fmtBytes(s.bytes)}</div>
                </div>`)}
            </div>
          </div>`}
      </div>
    </div>
  `;
};

// --- Compare tab (side-by-side video players) -------------------------------

const Compare = ({ job }) => {
  const videosByRole = useMemo(() => {
    const m = {};
    for (const v of job.videos) (m[v.role] ||= []).push(v);
    return m;
  }, [job.videos]);

  // Default panels: pick representative videos per role; collapse to roles that actually exist.
  const availableRoles = ROLE_ORDER.filter(r => videosByRole[r]?.length);
  const defaultPanels = (() => {
    const pick = (r) => videosByRole[r]?.[0]?.rel;
    const picks = [pick('original'), pick('lipsync'), pick('final')].filter(Boolean);
    if (picks.length < 2) {
      // Fall back to first two of anything.
      return job.videos.slice(0, 2).map(v => v.rel);
    }
    return picks;
  })();

  const [panels, setPanels] = useState(defaultPanels);
  useEffect(() => { setPanels(defaultPanels); }, [job.job_id]);

  const [speed, setSpeed]   = useState(1);
  const [sync, setSync]     = useState(true);
  const [playing, setPlaying] = useState(false);
  const [cur, setCur]       = useState(0);
  const [dur, setDur]       = useState(0);
  const refs = useRef({});

  useEffect(() => {
    for (const el of Object.values(refs.current)) {
      if (el) el.playbackRate = speed;
    }
  }, [speed, panels]);

  const first = () => refs.current[panels[0]];

  const onPlay = () => {
    const anchor = first();
    if (!anchor) return;
    if (anchor.paused) {
      anchor.play(); setPlaying(true);
      if (sync) {
        panels.slice(1).forEach(p => {
          const el = refs.current[p];
          if (el) { el.currentTime = anchor.currentTime; el.play().catch(() => {}); }
        });
      }
    } else {
      anchor.pause(); setPlaying(false);
      if (sync) panels.slice(1).forEach(p => refs.current[p]?.pause());
    }
  };

  const onSeek = (t) => {
    panels.forEach(p => {
      const el = refs.current[p];
      if (el) el.currentTime = t;
    });
    setCur(t);
  };

  const onAnchorTime = () => {
    const anchor = first();
    if (!anchor) return;
    setCur(anchor.currentTime);
    if (sync) {
      panels.slice(1).forEach(p => {
        const el = refs.current[p];
        if (el && Math.abs(el.currentTime - anchor.currentTime) > 0.25) {
          el.currentTime = anchor.currentTime;
        }
      });
    }
  };

  if (job.videos.length === 0) {
    return html`<div class="p-10 text-slate-400">No video files in this job.</div>`;
  }

  return html`
    <div class="h-full flex flex-col">
      <!-- controls -->
      <div class="px-6 py-3 border-b border-line/50 bg-panel/40 flex flex-wrap items-center gap-4 text-sm">
        <button onClick=${onPlay}
                class="w-10 h-10 rounded-full bg-accent hover:bg-indigo-500 flex items-center justify-center text-white">
          ${playing
            ? html`<svg class="w-5 h-5" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="5" width="4" height="14"/><rect x="14" y="5" width="4" height="14"/></svg>`
            : html`<svg class="w-5 h-5 ml-[1px]" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>`}
        </button>
        <div class="font-mono text-xs text-slate-300 min-w-[6rem]">${fmtTime(cur)} / ${fmtTime(dur)}</div>

        <div class="flex items-center gap-1 text-xs">
          ${[0.5, 0.75, 1, 1.25, 1.5, 2].map(s => html`
            <button key=${s}
              onClick=${() => setSpeed(s)}
              class="px-2 py-1 rounded ${Math.abs(speed - s) < 0.01 ? 'bg-accent text-white' : 'bg-slate-800 hover:bg-slate-700 text-slate-300'}">${s}x</button>`)}
        </div>

        <label class="flex items-center gap-2 ml-auto text-xs text-slate-300">
          <input type="checkbox" checked=${sync} onChange=${(e) => setSync(e.target.checked)}
                 class="accent-indigo-500 w-4 h-4" />
          Sync players
        </label>

        <div class="flex items-center gap-1 text-xs">
          <span class="text-slate-500">Panels</span>
          ${[2, 3, 4].map(n => html`
            <button key=${n}
              disabled=${job.videos.length < n}
              onClick=${() => setPanels(job.videos.slice(0, n).map(v => v.rel))}
              class="px-2 py-1 rounded ${panels.length === n ? 'bg-accent text-white' : 'bg-slate-800 hover:bg-slate-700 text-slate-300 disabled:opacity-30'}">${n}</button>`)}
        </div>
      </div>

      <!-- players grid -->
      <div class="flex-1 overflow-auto px-6 py-5">
        <div class="grid gap-5 ${panels.length <= 1 ? 'grid-cols-1'
                                : panels.length === 2 ? 'grid-cols-1 lg:grid-cols-2'
                                : 'grid-cols-1 lg:grid-cols-2 2xl:grid-cols-3'}">
          ${panels.map((rel, idx) => {
            const v = job.videos.find(x => x.rel === rel);
            if (!v) return null;
            return html`
              <div key=${rel + idx} class="rounded-2xl overflow-hidden border bg-black/40 ring-1 ${ROLE_STYLE[v.role] || ROLE_STYLE.other}">
                <div class="px-4 py-2 flex items-center justify-between bg-slate-900/60 border-b border-line/40">
                  <div class="flex items-center gap-2 min-w-0">
                    <span class="text-[10px] uppercase tracking-widest text-slate-300">${ROLE_LABEL[v.role] || v.role}</span>
                    <select
                      class="bg-transparent text-xs font-mono text-slate-200 border border-line/50 rounded px-1 py-0.5 min-w-0 max-w-[20rem] truncate"
                      value=${rel}
                      onChange=${(e) => {
                        const next = [...panels];
                        next[idx] = e.target.value;
                        setPanels(next);
                      }}>
                      ${job.videos.map(opt => html`
                        <option key=${opt.rel} value=${opt.rel}>[${opt.role}] ${opt.rel}</option>`)}
                    </select>
                  </div>
                  <div class="text-[11px] text-slate-500 shrink-0 font-mono">${fmtBytes(v.size)}</div>
                </div>
                <div class="relative aspect-video bg-black">
                  <video
                    ref=${(el) => { refs.current[rel] = el; }}
                    src=${v.url}
                    controls=${idx !== 0}
                    muted=${idx !== 0}
                    onLoadedMetadata=${idx === 0 ? (e) => setDur(e.target.duration) : null}
                    onTimeUpdate=${idx === 0 ? onAnchorTime : null}
                    onPlay=${idx === 0 ? () => setPlaying(true) : null}
                    onPause=${idx === 0 ? () => setPlaying(false) : null}
                    class="absolute inset-0 w-full h-full">
                  </video>
                </div>
                <div class="px-3 py-1.5 flex items-center gap-2 text-[11px] text-slate-400 border-t border-line/40">
                  <code class="truncate font-mono text-slate-300 flex-1">${v.rel}</code>
                  ${idx === 0 && html`<span class="text-[10px] text-slate-500">anchor (audio)</span>`}
                </div>
              </div>
            `;
          })}
        </div>

        <!-- shared scrubber -->
        <div class="mt-5 rounded-2xl border border-line/60 bg-panel/40 p-4">
          <div class="flex items-center gap-3">
            <input type="range" min="0" max=${dur || 0} step="0.05" value=${cur}
                   onInput=${(e) => onSeek(parseFloat(e.target.value))}
                   class="speed flex-1" />
            <span class="font-mono text-xs text-slate-400 w-28 text-right">${fmtTime(cur)} / ${fmtTime(dur)}</span>
          </div>
          <div class="text-[10px] text-slate-500 mt-2">
            Only the leftmost (anchor) player's audio is unmuted so you can hear one at a time. Switch anchor by reordering panels.
          </div>
        </div>
      </div>
    </div>
  `;
};

// --- Files tab (tree browser + previews) ------------------------------------

const Icon = ({ kind }) => {
  const base = 'w-4 h-4 shrink-0';
  if (kind === 'video') return html`<svg class="${base} text-sky-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="4" width="20" height="16" rx="2"/><path d="m10 9 5 3-5 3Z"/></svg>`;
  if (kind === 'audio') return html`<svg class="${base} text-violet-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/></svg>`;
  if (kind === 'image') return html`<svg class="${base} text-amber-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="9" cy="9" r="2"/><path d="m21 15-5-5L5 21"/></svg>`;
  if (kind === 'json' || kind === 'text')
    return html`<svg class="${base} text-emerald-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/></svg>`;
  return html`<svg class="${base} text-slate-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/></svg>`;
};

const TreeNode = ({ node, path, onPick, picked, depth = 0 }) => {
  const [open, setOpen] = useState(depth < 2);
  const dirs  = Object.entries(node.dirs || {});
  const files = node.files || [];
  if (depth > 0 && !open) {
    return html`
      <div>
        <button onClick=${() => setOpen(true)}
          style=${{ paddingLeft: `${depth * 12}px` }}
          class="flex items-center gap-1.5 w-full px-2 py-1 text-xs hover:bg-slate-800/40 rounded">
          <svg class="w-3 h-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m9 18 6-6-6-6"/></svg>
          <span class="font-mono text-slate-200">${path}/</span>
          <span class="text-[10px] text-slate-500 ml-auto">${files.length + dirs.length}</span>
        </button>
      </div>`;
  }
  return html`
    <div>
      ${depth > 0 && html`
        <button onClick=${() => setOpen(false)}
          style=${{ paddingLeft: `${depth * 12}px` }}
          class="flex items-center gap-1.5 w-full px-2 py-1 text-xs hover:bg-slate-800/40 rounded">
          <svg class="w-3 h-3 rotate-90" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m9 18 6-6-6-6"/></svg>
          <span class="font-mono text-slate-200">${path}/</span>
        </button>`}
      ${dirs.map(([k, v]) => html`<${TreeNode} key=${k} node=${v} path=${k} onPick=${onPick} picked=${picked} depth=${depth + 1} />`)}
      ${files.map(f => html`
        <button key=${f.rel}
          onClick=${() => onPick(f)}
          style=${{ paddingLeft: `${(depth + 1) * 12}px` }}
          class="flex items-center gap-2 w-full px-2 py-1 text-xs hover:bg-slate-800/40 rounded ${picked?.rel === f.rel ? 'bg-accent/20 ring-1 ring-accent/40' : ''}">
          <${Icon} kind=${f.kind} />
          <span class="font-mono text-slate-200 truncate flex-1 text-left">${f.name}</span>
          <span class="text-[10px] text-slate-500">${fmtBytes(f.size)}</span>
        </button>`)}
    </div>
  `;
};

const PreviewPane = ({ file }) => {
  const [text, setText] = useState(null);
  const [err, setErr]   = useState(null);
  useEffect(() => {
    if (!file) return;
    setText(null); setErr(null);
    if (file.kind === 'json' || file.kind === 'text') {
      fetch(file.url).then(r => r.text()).then(t => setText(t)).catch(e => setErr(e));
    }
  }, [file?.url]);

  if (!file) {
    return html`<div class="p-6 text-slate-500 text-sm">Select a file to preview.</div>`;
  }
  return html`
    <div class="h-full flex flex-col">
      <div class="px-5 py-3 border-b border-line/50 bg-panel/40 flex items-center gap-3">
        <${Icon} kind=${file.kind} />
        <div class="min-w-0 flex-1">
          <div class="font-mono text-sm text-slate-100 truncate">${file.name}</div>
          <div class="font-mono text-[11px] text-slate-500 truncate">${file.rel} · ${fmtBytes(file.size)}</div>
        </div>
        <button onClick=${() => copyUrl(file.url)}
          class="text-[11px] px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300">Copy URL</button>
        <a href=${file.url} target="_blank"
          class="text-[11px] px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300">Open</a>
        <a href=${file.url} download
          class="text-[11px] px-2 py-1 rounded bg-indigo-500/90 hover:bg-indigo-400 text-white">Download</a>
      </div>
      <div class="flex-1 overflow-auto p-5">
        ${file.kind === 'video' && html`
          <video src=${file.url} controls class="w-full max-h-[70vh] rounded-xl bg-black"></video>`}
        ${file.kind === 'audio' && html`
          <audio src=${file.url} controls class="w-full"></audio>`}
        ${file.kind === 'image' && html`
          <img src=${file.url} class="max-w-full max-h-[75vh] mx-auto rounded-xl border border-line/50" />`}
        ${(file.kind === 'json' || file.kind === 'text') && html`
          <pre class="text-xs font-mono text-slate-200 whitespace-pre-wrap break-all">${err ? `Error: ${err.message}` : (text ?? 'Loading…')}</pre>`}
        ${file.kind === 'binary' && html`
          <div class="text-sm text-slate-400">Binary file (${file.ext}). Download to inspect.</div>`}
        ${file.kind === 'other' && html`
          <div class="text-sm text-slate-400">No inline preview. Download to inspect.</div>`}
      </div>
    </div>
  `;
};

const Files = ({ job }) => {
  const [picked, setPicked] = useState(null);
  useEffect(() => { setPicked(null); }, [job.job_id]);
  return html`
    <div class="h-full grid grid-cols-1 lg:grid-cols-[22rem_1fr]">
      <aside class="border-r border-line/50 overflow-y-auto bg-panel/30">
        <div class="px-4 py-3 border-b border-line/50 text-xs uppercase tracking-widest text-slate-400">
          Files · ${job.n_files}
        </div>
        <div class="p-2">
          <${TreeNode} node=${job.tree} path="" onPick=${setPicked} picked=${picked} depth=${0} />
        </div>
      </aside>
      <div class="min-h-0">
        <${PreviewPane} file=${picked} />
      </div>
    </div>
  `;
};

// --- Images tab (compare-grid for restore/diagnostic PNGs) ------------------

const Images = ({ job }) => {
  const groups = useMemo(() => {
    const g = {};
    for (const im of job.images) {
      const dir = im.rel.includes('/') ? im.rel.split('/').slice(0, -1).join('/') : '(root)';
      (g[dir] ||= []).push(im);
    }
    return g;
  }, [job.images]);
  const dirs = Object.keys(groups).sort();
  if (job.images.length === 0) {
    return html`<div class="p-10 text-slate-400">No images in this job.</div>`;
  }
  return html`
    <div class="h-full overflow-y-auto p-6 space-y-6">
      ${dirs.map(d => html`
        <section key=${d}>
          <div class="font-mono text-xs text-slate-400 mb-2">${d}/ <span class="text-slate-500">(${groups[d].length})</span></div>
          <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
            ${groups[d].map(im => html`
              <a key=${im.rel} href=${im.url} target="_blank"
                 class="group rounded-xl overflow-hidden border border-line/60 bg-black/40 hover:border-accent/50 transition-colors">
                <img src=${im.url} class="w-full aspect-video object-contain bg-black" />
                <div class="px-2 py-1 text-[11px] font-mono text-slate-300 truncate">${im.name}</div>
              </a>`)}
          </div>
        </section>`)}
    </div>
  `;
};

// --- Clips tab (MT-driven, only when has_mt) --------------------------------

const Clips = ({ job }) => {
  const videoRef = useRef(null);
  const [playingId, setPlayingId] = useState(null);
  const [showAll, setShowAll] = useState(false);
  const segs = job.segments || [];
  const originalVideo = job.videos.find(v => v.role === 'original') || job.videos[0];
  const visible = showAll ? segs : segs.slice(0, 10);

  if (segs.length === 0) {
    return html`<div class="p-10 text-slate-500 text-sm">This job has no MT segments to inspect.</div>`;
  }
  return html`
    <div class="h-full overflow-y-auto">
      <div class="max-w-3xl mx-auto px-6 py-6">
        <div class="mb-5">
          <h1 class="text-2xl font-semibold tracking-tight">Translated clips</h1>
          <p class="text-sm text-slate-400 mt-1">Original speech aligned with its translation and per-segment synthesized audio.</p>
        </div>
        <div class="rounded-2xl bg-panel/60 border border-line/60 p-5">
          <div class="rounded-xl overflow-hidden bg-black">
            ${originalVideo
              ? html`<video ref=${videoRef} src=${originalVideo.url} controls class="w-full aspect-video"></video>`
              : html`<div class="aspect-video flex items-center justify-center text-slate-500">No source video</div>`}
          </div>
          <div class="mt-6">
            <div class="flex items-baseline gap-3 mb-3">
              <h3 class="text-base font-semibold">Speech segments</h3>
              <span class="text-xs text-slate-400">showing ${visible.length} of ${segs.length}</span>
            </div>
            <div class="space-y-2">
              ${visible.map((s, i) => html`
                <div key=${s.id} class="rounded-xl border border-line/60 bg-slate-900/40 px-4 py-3">
                  <div class="flex items-center gap-3">
                    <div class="font-mono text-sm text-slate-400 w-8">#${i + 1}</div>
                    <button onClick=${() => { if (videoRef.current) { videoRef.current.currentTime = s.start; videoRef.current.play().catch(() => {}); } }}
                      class="font-mono text-sm text-slate-200 hover:text-sky-300">
                      ${s.start.toFixed(1)}s – ${s.end.toFixed(1)}s
                    </button>
                    <div class="text-xs text-slate-500 w-14">${(s.end - s.start).toFixed(1)}s</div>
                    ${s.tts_url && html`<audio controls src=${s.tts_url} class="flex-1 h-8 min-w-0"></audio>`}
                  </div>
                  <div class="grid grid-cols-1 md:grid-cols-2 gap-3 mt-2 text-sm">
                    <div><span class="text-[10px] uppercase tracking-widest text-sky-300">Original</span><div>${s.src_text}</div></div>
                    <div><span class="text-[10px] uppercase tracking-widest text-violet-300">Translated</span><div>${s.final_text || s.mt_text}</div></div>
                  </div>
                </div>
              `)}
            </div>
            ${segs.length > 10 && html`
              <div class="flex justify-center mt-4">
                <button onClick=${() => setShowAll(a => !a)}
                  class="w-10 h-10 rounded-full bg-slate-800 hover:bg-slate-700 text-slate-300 flex items-center justify-center">
                  <svg class="w-5 h-5 transition-transform ${showAll ? 'rotate-180' : ''}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m6 9 6 6 6-6"/></svg>
                </button>
              </div>`}
          </div>
        </div>
      </div>
    </div>
  `;
};

// --- top-level app ----------------------------------------------------------

const App = () => {
  const { data: jobsData, err, loading } = useFetch('/api/jobs');
  const jobs = jobsData?.jobs || [];
  const [jobId, setJobId] = useState(null);
  const [mode, setMode]   = useState('overview');

  useEffect(() => {
    if (!jobId && jobs.length > 0) setJobId(jobs[0].job_id);
  }, [jobs, jobId]);

  const { data: job, err: jobErr, loading: jobLoading } =
    useFetch(jobId ? `/api/jobs/${encodeURIComponent(jobId)}` : null);

  const modes = [
    ['overview', 'Overview', html`<svg class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/></svg>`],
    ['compare',  'Compare',  html`<svg class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="4" width="9" height="16" rx="2"/><rect x="13" y="4" width="9" height="16" rx="2"/></svg>`],
    ['images',   'Images',   html`<svg class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="9" cy="9" r="2"/><path d="m21 15-5-5L5 21"/></svg>`],
    ['files',    'Files',    html`<svg class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/></svg>`],
  ];
  // Clips tab is only useful when the job carries MT output.
  if (job?.has_mt) {
    modes.push(['clips', 'Clips', html`<svg class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="5" width="7" height="14" rx="1.5"/><rect x="14" y="5" width="7" height="14" rx="1.5"/></svg>`]);
  }

  return html`
    <div class="h-screen flex overflow-hidden">
      <${Sidebar} jobs=${jobs} activeJobId=${jobId} onSelect=${setJobId} />
      <main class="flex-1 flex flex-col min-w-0">
        <header class="px-6 py-3 border-b border-line/70 bg-panel/60 flex items-center gap-4 shrink-0">
          <div class="min-w-0">
            ${job
              ? html`
                <div class="text-sm font-semibold truncate">${job.slug || job.job_id}</div>
                <div class="text-xs text-slate-400 font-mono truncate">${job.config_slug || ''}</div>`
              : html`<div class="text-sm text-slate-400">${loading ? 'Loading…' : 'Pick a job'}</div>`}
          </div>
          <div class="ml-auto flex items-center gap-1 rounded-xl bg-slate-800/70 p-1">
            ${modes.map(([k, label, icon]) => html`
              <button key=${k} onClick=${() => setMode(k)}
                class="px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${mode === k ? 'bg-accent text-white' : 'text-slate-300 hover:bg-slate-700'}">
                <span class="inline-flex items-center gap-2">${icon}${label}</span>
              </button>`)}
          </div>
        </header>
        <section class="flex-1 min-h-0">
          ${err && html`<div class="p-6 text-red-400 text-sm">Failed to load jobs: ${err.message}</div>`}
          ${jobErr && html`<div class="p-6 text-red-400 text-sm">Failed to load job: ${jobErr.message}</div>`}
          ${jobLoading && html`<div class="p-6 text-slate-400 text-sm">Loading job…</div>`}
          ${job && (
            mode === 'overview' ? html`<${Overview} job=${job} />`
          : mode === 'compare'  ? html`<${Compare}  job=${job} />`
          : mode === 'images'   ? html`<${Images}   job=${job} />`
          : mode === 'files'    ? html`<${Files}    job=${job} />`
          : mode === 'clips'    ? html`<${Clips}    job=${job} />`
                                : null
          )}
          ${!job && !loading && !jobLoading && jobs.length === 0 && html`
            <div class="p-10 text-slate-500 text-sm">No jobs under <code>data/jobs/</code>.</div>`}
        </section>
      </main>
    </div>
  `;
};

createRoot(document.getElementById('root')).render(html`<${App} />`);
