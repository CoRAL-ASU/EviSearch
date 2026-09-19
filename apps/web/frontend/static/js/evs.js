// Shared helpers for EviSearch pages: fetch, escaping, toasts, the reviewer's name, drafts kept in the browser, cell-state
// labels and a small PDF page viewer that highlights a quote.
window.EVS = (function () {
    const $ = (id) => document.getElementById(id);
    const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
    const enc = encodeURIComponent;

    async function request(url, opts) {
        let res;
        try { res = await fetch(url, opts); } catch (e) { return {success: false, error: 'network error: ' + e.message}; }
        let body = null;
        try { body = await res.json(); } catch (e) { body = {success: false, error: `HTTP ${res.status}`}; }
        if (body && body.success === undefined) body.success = res.ok;
        return body;
    }
    const get = (url) => request(url);
    const post = (url, body) => request(url, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body || {})});

    function toast(message, kind = 'info', ms = 4000, action) {
        const box = document.createElement('div');
        box.className = `alert alert-${kind} shadow py-2 text-sm max-w-md`;
        box.innerHTML = `<span>${esc(message)}</span>`;
        if (action) {
            const b = document.createElement('button');
            b.className = 'btn btn-xs';
            b.textContent = action.label;
            b.onclick = () => { box.remove(); action.run(); };
            box.appendChild(b);
        }
        $('toasts').appendChild(box);
        setTimeout(() => box.remove(), ms);
    }

    // ---- browser storage (per viewer; always optional) ----
    const store = {
        get(key, fallback = null) { try { const v = localStorage.getItem('evs.' + key); return v === null ? fallback : JSON.parse(v); } catch (e) { return fallback; } },
        set(key, value) { try { localStorage.setItem('evs.' + key, JSON.stringify(value)); } catch (e) {} },
        del(key) { try { localStorage.removeItem('evs.' + key); } catch (e) {} },
    };

    // ---- reviewer name (shared across pages, also read by the older pages' key) ----
    function reviewer() { return ($('reviewer') && $('reviewer').value.trim()) || ''; }
    function needReviewer() {
        const name = reviewer();
        if (!name) { toast('Enter your name (top right) first — every saved review records who made it.', 'warning'); $('reviewer') && $('reviewer').focus(); }
        return name;
    }
    document.addEventListener('DOMContentLoaded', () => {
        const input = $('reviewer');
        if (!input) return;
        let saved = store.get('reviewer', '');
        try { saved = saved || localStorage.getItem('evisearch.reviewer') || ''; } catch (e) {}
        input.value = saved;
        input.addEventListener('change', () => { store.set('reviewer', input.value.trim()); try { localStorage.setItem('evisearch.reviewer', input.value.trim()); } catch (e) {} });
    });

    // ---- unsaved-changes guard ----
    const dirty = new Set();
    window.addEventListener('beforeunload', (e) => { if (dirty.size) { e.preventDefault(); e.returnValue = ''; } });

    // ---- labels ----
    const STATE = {
        verified: ['Verified on the page', 'badge-success'],
        flagged: ['Flagged for review', 'badge-warning'],
        corrected: ['Corrected by a reviewer', 'badge-info'],
        accepted: ['Confirmed by a reviewer', 'badge-success'],
        not_reported: ['Not reported', 'badge-ghost'],
        unverified: ['No page check', 'badge-ghost'],
    };
    const stateBadge = (s) => { const [label, cls] = STATE[s] || [s, 'badge-ghost']; return `<span class="badge badge-sm ${cls}">${esc(label)}</span>`; };
    const when = (t) => { if (!t) return ''; const d = typeof t === 'number' ? new Date(t * 1000) : new Date(t); return isNaN(d) ? String(t) : d.toLocaleString([], {month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'}); };
    const shortDoc = (id) => String(id || '').replace(/^NCT\d+_/, '').replace(/_/g, ' ');

    // ---- page viewer: the server renders the page and says where the quote sits, so nothing PDF-related runs here ----
    function PdfViewer(container) {
        this.el = container;
        this.docId = null;
        this.page = 1;
        this.pages = null;
        this.quote = '';
        this.token = 0;
        this.scale = 1.6;
    }
    PdfViewer.prototype.open = async function (docId, page, quote) {
        this.docId = docId;
        this.page = Math.max(1, page || 1);
        this.quote = quote || '';
        await this.render();
    };
    PdfViewer.prototype.render = async function () {
        const token = ++this.token;
        const docId = this.docId, page = this.page;
        this.el.innerHTML = '<div class="p-6 text-sm opacity-60">Loading the page…</div>';
        const r = await get(`/api/documents/${enc(docId)}/page/${page}/find?q=${enc(this.quote.slice(0, 400))}&scale=${this.scale}`);
        if (token !== this.token) return;
        if (!r.success) { this.el.innerHTML = `<div class="p-6 text-sm text-error">${esc(r.error)}</div>`; return; }
        this.pages = r.pages;
        const bar = `<div class="flex items-center justify-between gap-2 px-2 py-1 text-xs sticky top-0 bg-base-300 z-10">
            <button class="btn btn-xs" data-go="-1" ${page <= 1 ? 'disabled' : ''}>‹ Prev</button>
            <span>Page ${page} of ${r.pages}${this.quote ? (r.found ? ' · quote highlighted' : ' · quote not found in the page text (a table or figure image?)') : ''}</span>
            <button class="btn btn-xs" data-go="1" ${page >= r.pages ? 'disabled' : ''}>Next ›</button></div>`;
        const pct = (v, whole) => (100 * v / whole).toFixed(3) + '%';  // percentages so the page scales to the panel
        this.el.innerHTML = bar + `<div class="relative w-full" id="evs-page">
            <img src="/api/documents/${enc(docId)}/page/${page}.png?scale=${this.scale}" width="${r.width}" height="${r.height}" alt="page ${page}" class="block w-full h-auto" />
            ${(r.rects || []).map((b) => `<div class="evs-hl absolute" style="left:${pct(b[0], r.width)};top:${pct(b[1], r.height)};width:${pct(b[2] - b[0], r.width)};height:${pct(b[3] - b[1], r.height)};
                background:rgba(250,204,21,.32);outline:1px solid rgba(250,204,21,.9);pointer-events:none"></div>`).join('')}</div>`;
        this.el.querySelectorAll('[data-go]').forEach((b) => b.onclick = () => { this.page += Number(b.dataset.go); this.render(); });
        const mark = this.el.querySelector('.evs-hl');
        if (mark) mark.scrollIntoView({block: 'center'});
    };

    return {$, esc, enc, get, post, toast, store, reviewer, needReviewer, dirty, STATE, stateBadge, when, shortDoc, PdfViewer};
})();
