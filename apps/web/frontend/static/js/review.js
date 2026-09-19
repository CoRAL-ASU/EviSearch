// Cell review: a queue with flagged cells first, what each arm answered, and the paper open at the cited page.
(function () {
    const {$, esc, enc, get, post, toast, store, needReviewer, dirty, stateBadge, when, shortDoc, PdfViewer} = EVS;
    const TABLE = $('page').dataset.table;
    const params = new URLSearchParams(location.search);
    let runs = [], run = params.get('run') || '', docFilter = params.get('doc') || '', kind = 'flagged', unrevOnly = true;
    let queue = [], items = [], current = null;   // current = {doc_id, column}
    const cellsCache = new Map();                  // doc -> {columns, schema_id, definitions}
    const viewer = new PdfViewer($('pdf'));
    const ruleState = {record: null};
    let ruleContext = null;

    const draftKey = (doc, col) => `cell.${run}.${doc}.${col}`;
    const cellOf = (doc, col) => (cellsCache.get(doc) || {columns: []}).columns.find((c) => c.column === col);

    async function start() {
        const t = await get(`/api/tables/${enc(TABLE)}`);
        if (!t.success) return toast(t.error, 'error');
        runs = t.runs.slice().reverse();
        if (!runs.length) { $('cell').innerHTML = '<div class="card bg-base-100 shadow"><div class="card-body">No run to review yet.</div></div>'; return; }
        if (!runs.some((r) => r.run === run)) run = t.showcase_run || runs[0].run;
        $('run').innerHTML = runs.map((r) => `<option value="${esc(r.run)}" ${r.run === run ? 'selected' : ''}>${esc(r.run)} — ${r.flagged} flagged, ${r.reviewed} reviewed</option>`).join('');
        await loadQueue();
    }

    async function loadQueue(keep) {
        const r = await get(`/api/runs/${enc(run)}/queue`);
        if (!r.success) { $('queue').innerHTML = `<div class="p-4 text-error">${esc(r.error)}</div>`; return; }
        queue = r.items;
        $('doc').innerHTML = '<option value="">All papers</option>' + r.docs.map((d) =>
            `<option value="${esc(d.doc_id)}" ${d.doc_id === docFilter ? 'selected' : ''}>${esc(shortDoc(d.name))}</option>`).join('');
        const c = r.counts;
        $('progress').innerHTML = `flagged <b>${c.flagged.reviewed}/${c.flagged.total}</b> reviewed · A ≠ B <b>${c.disputed.reviewed}/${c.disputed.total}</b> · other <b>${c.other.reviewed}/${c.other.total}</b>`;
        if (!keep && !current) chooseView();
        renderQueue(keep);
    }

    const pick = (k, unrev) => queue.filter((it) => (k === 'all' || it.kind === k) && (!docFilter || it.doc_id === docFilter) && (!unrev || !it.reviewed));
    function chooseView() {
        // open on the first view that has something: cells still to review, else the same cells already reviewed
        for (const [k, unrev] of [['flagged', true], ['disputed', true], ['flagged', false], ['disputed', false], ['all', true], ['all', false]]) {
            if (pick(k, unrev).length) {
                kind = k; unrevOnly = unrev;
                $('unrev').checked = unrev;
                document.querySelectorAll('#kind button').forEach((x) => x.classList.toggle('btn-active', x.dataset.k === k));
                return;
            }
        }
    }

    function renderQueue(keep) {
        items = pick(kind, unrevOnly);
        $('queue').innerHTML = items.length ? items.map((it, i) => `<div class="qi p-2 flex gap-2 items-start" data-i="${i}">
                <span class="dot mt-1 s-${it.state} border border-base-content/30" title="${esc((EVS.STATE[it.state] || [it.state])[0])}"></span>
                <div class="min-w-0 flex-1">
                    <div class="truncate" title="${esc(it.column)}">${esc(it.column)}</div>
                    <div class="text-xs opacity-60 truncate">${esc(shortDoc(it.doc_id))} · ${esc(it.value || '—')}</div>
                </div>${it.reviewed ? '<span class="badge badge-xs badge-info">reviewed</span>' : ''}</div>`).join('')
            : '<div class="p-4 opacity-60">Nothing left in this filter. Switch to “All cells” or turn off “to review only”.</div>';
        $('queue').querySelectorAll('.qi').forEach((el) => el.onclick = () => select(items[Number(el.dataset.i)]));
        const keepIt = keep && items.find((it) => it.doc_id === keep.doc_id && it.column === keep.column);
        if (keepIt) markSelected(keepIt); else if (items.length && !current) select(items[0]);
        else if (current) markSelected(current);
    }
    function markSelected(it) {
        const i = items.findIndex((x) => x.doc_id === it.doc_id && x.column === it.column);
        $('queue').querySelectorAll('.qi').forEach((el, k) => el.classList.toggle('sel', k === i));
        const el = $('queue').querySelector('.qi.sel');
        if (el) el.scrollIntoView({block: 'nearest'});
    }

    function unsavedOk() {
        if (!current) return true;
        const ta = document.querySelector('#cell .val');
        if (!ta) return true;
        const cell = cellOf(current.doc_id, current.column);
        const saved = (cell && cell.review && cell.review.value !== null && cell.review.value !== undefined) ? cell.review.value : (cell ? cell.value : '');
        if (ta.value === saved) return true;
        return confirm('This cell has unsaved text. Leave it without saving?');
    }

    async function select(it) {
        if (!it || (current && current.doc_id === it.doc_id && current.column === it.column)) return;
        if (!unsavedOk()) return;
        current = {doc_id: it.doc_id, column: it.column};
        markSelected(it);
        const u = new URL(location.href);
        u.searchParams.set('run', run); u.searchParams.set('doc', it.doc_id); u.searchParams.set('column', it.column);
        history.replaceState(null, '', u);
        if (!cellsCache.has(it.doc_id)) {
            $('cell').innerHTML = '<div class="card bg-base-100 shadow"><div class="card-body p-6 opacity-60">Loading the paper\'s cells…</div></div>';
            const r = await get(`/api/runs/${enc(run)}/docs/${enc(it.doc_id)}/cells`);
            if (!r.success) { $('cell').innerHTML = `<div class="card bg-base-100 shadow"><div class="card-body text-error">${esc(r.error)}</div></div>`; return; }
            cellsCache.set(it.doc_id, r);
        }
        renderCell();
    }

    function armBlock(label, value, reasoning, evidence, extra) {
        return `<div class="border border-base-300 rounded p-2 space-y-1">
            <div class="flex items-center justify-between gap-2"><span class="text-xs uppercase tracking-wide opacity-60">${esc(label)}</span>${extra || ''}</div>
            <div class="font-medium break-words">${value === null || value === undefined ? '<span class="opacity-50">did not answer</span>' : esc(value) || '<span class="opacity-50">empty</span>'}</div>
            ${(evidence || []).map((e) => `<button class="btn btn-xs btn-outline ev" data-page="${e.page}" data-quote="${esc(e.quote)}">p${e.page} · ${esc(e.modality)}${e.verdict ? ' · ' + esc(e.verdict) : ''}</button>`).join(' ')}
            ${reasoning ? `<details class="text-xs opacity-70"><summary>why</summary><div class="whitespace-pre-wrap">${esc(reasoning)}</div></details>` : ''}</div>`;
    }

    function renderCell() {
        const data = cellsCache.get(current.doc_id);
        const cell = cellOf(current.doc_id, current.column);
        if (!cell) { $('cell').innerHTML = '<div class="card bg-base-100 shadow"><div class="card-body">This column is not in the run\'s output.</div></div>'; return; }
        const rv = cell.review && cell.review.value !== null && cell.review.value !== undefined ? cell.review : null;
        const draft = store.get(draftKey(current.doc_id, current.column));
        const value = draft ? draft.text : (rv ? rv.value : cell.value);
        const reasons = data.reasons || [];
        $('cell').innerHTML = `<div class="card bg-base-100 shadow"><div class="card-body p-4 space-y-3">
            <div class="flex flex-wrap items-center gap-2">
                <h2 class="text-lg font-semibold">${esc(cell.column)}</h2>${stateBadge(cell.state)}
                ${cell.disputed ? '<span class="badge badge-sm badge-outline">Agent A ≠ Agent B</span>' : ''}
                <span class="text-xs opacity-60">${esc(shortDoc(data.name))} · ${esc(cell.group || '')}</span></div>
            ${cell.flagged && cell.flag_reason ? `<div class="alert alert-warning py-2 text-sm"><span>Flagged: ${esc(cell.flag_reason)}</span></div>` : ''}
            ${cell.definition ? `<div class="text-sm bg-base-200 rounded p-2"><span class="opacity-60">Definition (${esc(data.definitions.source)}${data.definitions.version ? ' v' + data.definitions.version : ''}):</span> ${esc(cell.definition)}</div>` : ''}
            <div class="grid gap-2 md:grid-cols-3">
                ${armBlock('Agent A · reads the paper', cell.a, cell.a_reasoning, cell.a_evidence)}
                ${armBlock('Agent B · searches it', cell.b, cell.b_reasoning, cell.b_evidence)}
                ${armBlock('Arbiter · checked the page', cell.value, cell.reasoning, cell.evidence,
                    `<span class="text-xs">${cell.verified === true ? '✓ verified' : cell.verified === false ? 'not verified' : ''}${cell.decided_by ? ' · ' + esc(cell.decided_by) : ''}</span>`)}
            </div>
            <div class="space-y-2">
                <div class="flex items-center gap-2"><span class="text-sm font-semibold">Your review</span>
                    ${rv ? `<span class="text-xs opacity-70">${esc(rv.by || '—')} · ${esc(when(rv.at))} · ${esc(rv.reason || '')}</span>
                        <button class="btn btn-xs undo">Undo</button>` : ''}
                    ${draft ? '<span class="badge badge-xs badge-accent">unsaved</span>' : ''}</div>
                <textarea class="textarea textarea-bordered w-full val" rows="2">${esc(value)}</textarea>
                <div class="flex flex-wrap gap-2 items-center">
                    <button class="btn btn-sm btn-success conf">Confirm the value</button>
                    <select class="select select-sm select-bordered why"><option value="">why it is wrong (required to correct)…</option>
                        ${reasons.map((r) => `<option ${rv && rv.reason === r ? 'selected' : ''}>${esc(r)}</option>`).join('')}</select>
                    <input class="input input-sm input-bordered note flex-1 min-w-[12rem]" placeholder="note: what is right and where you saw it" value="${esc(rv ? rv.note : '')}" />
                    <button class="btn btn-sm btn-primary save">Save correction</button>
                    <button class="btn btn-sm nr" title="Record that the paper does not report this">Not reported</button>
                    <button class="btn btn-sm btn-outline rule">Propose rule…</button>
                </div>
                <div class="text-xs opacity-60">The machine's value stays in the history; the table and export show your value.</div>
            </div>
            ${(cell.review && cell.review.history || []).length ? `<details class="text-xs opacity-70"><summary>Review history (${cell.review.history.length})</summary>
                ${cell.review.history.slice().reverse().map((h) => `<div>${esc(when(h.at))} · ${esc(h.by || '—')} · ${esc(h.value)}${h.reason ? ' (' + esc(h.reason) + ')' : ''}${h.undone ? ' — undone' : ''}</div>`).join('')}</details>` : ''}
        </div></div>`;

        const ta = document.querySelector('#cell .val');
        ta.addEventListener('input', () => {
            const saved = rv ? rv.value : cell.value;
            if (ta.value === saved) { store.del(draftKey(current.doc_id, current.column)); dirty.delete('cell'); }
            else { store.set(draftKey(current.doc_id, current.column), {text: ta.value, at: new Date().toISOString()}); dirty.add('cell'); }
        });
        document.querySelectorAll('#cell .ev').forEach((b) => b.onclick = () => viewer.open(current.doc_id, Number(b.dataset.page), b.dataset.quote));
        document.querySelector('#cell .conf').onclick = () => save(cell.value, '', document.querySelector('#cell .note').value);
        document.querySelector('#cell .save').onclick = () => {
            const reason = document.querySelector('#cell .why').value;
            if (!reason && ta.value !== cell.value) return toast('Pick why the value is wrong', 'warning');
            save(ta.value, reason, document.querySelector('#cell .note').value);
        };
        document.querySelector('#cell .nr').onclick = () => {
            const reason = document.querySelector('#cell .why').value || 'should be Not reported';
            save('Not reported', reason, document.querySelector('#cell .note').value);
        };
        document.querySelector('#cell .rule').onclick = () => openRule(cell);
        const undo = document.querySelector('#cell .undo');
        if (undo) undo.onclick = async () => {
            const by = needReviewer(); if (!by) return;
            const r = await post(`/api/runs/${enc(run)}/docs/${enc(current.doc_id)}/undo`, {event_id: rv.event_id, by});
            if (!r.success) return toast(r.error, 'error');
            toast('Review undone', 'success', 2500);
            await refreshCell();
        };
        const first = (cell.evidence || [])[0] || (cell.a_evidence || [])[0] || (cell.b_evidence || [])[0];
        if (first) viewer.open(current.doc_id, first.page, first.quote);
        else viewer.open(current.doc_id, 1, '');
    }

    async function save(value, reason, note) {
        const by = needReviewer(); if (!by) return;
        const r = await post(`/api/runs/${enc(run)}/docs/${enc(current.doc_id)}/review`,
            {column: current.column, value, reason, note, by});
        if (!r.success) return toast(r.error, 'error');
        store.del(draftKey(current.doc_id, current.column));
        dirty.delete('cell');
        const saved = {...current};
        const eventId = r.event.event_id;
        toast(r.state === 'accepted' ? 'Confirmed' : 'Correction saved', 'success', 8000, {label: 'Undo', run: async () => {
            const u = await post(`/api/runs/${enc(run)}/docs/${enc(saved.doc_id)}/undo`, {event_id: eventId, by});
            if (!u.success) return toast(u.error, 'error');
            current = saved;
            await refreshCell();
            toast('Review undone', 'success', 2500);
        }});
        await refreshCell();
        const next = items[items.findIndex((x) => x.doc_id === saved.doc_id && x.column === saved.column) + 1];
        if (next && unrevOnly) select(next);
    }

    async function refreshCell() {
        const r = await get(`/api/runs/${enc(run)}/docs/${enc(current.doc_id)}/cells`);
        if (r.success) cellsCache.set(current.doc_id, r);
        await loadQueue(current);
        renderCell();
    }

    function openRule(cell) {
        const rv = cell.review && cell.review.value !== null ? cell.review : null;
        ruleContext = {column: cell.column, definition: cell.definition, schema_id: cellsCache.get(current.doc_id).schema_id || TABLE,
            doc_id: current.doc_id, kind: 'extraction_review', before: cell.value, after: rv ? rv.value : '', reason: rv ? rv.reason : ''};
        ruleState.record = null;
        $('rule-col').textContent = `${cell.column} — ${shortDoc(current.doc_id)}`;
        $('rule-note').value = rv && rv.note ? rv.note : '';
        $('rule-out').innerHTML = '';
        $('rule-add').disabled = true;
        $('rule-dlg').showModal();
    }
    $('rule-draft').onclick = async () => {
        const by = needReviewer(); if (!by) return;
        if (!$('rule-note').value.trim()) return toast('Say what the rule should be', 'warning');
        $('rule-out').textContent = 'Drafting the rule and checking the knowledge base…';
        const r = await post('/api/conventions/propose', {...ruleContext, feedback: $('rule-note').value, by});
        if (!r.success) { $('rule-out').textContent = r.error; return; }
        if (!r.is_convention) { $('rule-out').innerHTML = `<div class="alert">Not a rule for every paper: ${esc(r.why_not)}. The correction alone is saved.</div>`; return; }
        RuleScope.show($('rule-out'), r, ruleState, ruleContext.schema_id, (v) => { $('rule-add').disabled = v === 'blocked' || v === 'checking'; });
    };
    $('rule-add').onclick = async () => {
        const r = await post('/api/conventions', {record: ruleState.record, by: needReviewer()});
        $('rule-out').insertAdjacentHTML('beforeend', `<div class="alert mt-2">${r.success ? (r.merged_into ? 'Merged into ' + esc(r.merged_into) : 'Stored as ' + esc(r.convention.id) + ' — approve it on the Knowledge page to use it in later runs') : esc(r.error)}</div>`);
        $('rule-add').disabled = true;
    };

    // filters
    document.querySelectorAll('#kind button').forEach((b) => b.onclick = () => {
        document.querySelectorAll('#kind button').forEach((x) => x.classList.remove('btn-active'));
        b.classList.add('btn-active'); kind = b.dataset.k; renderQueue(current);
    });
    $('unrev').onchange = () => { unrevOnly = $('unrev').checked; renderQueue(current); };
    $('doc').onchange = () => { docFilter = $('doc').value; renderQueue(current); };
    $('run').onchange = async () => { if (!unsavedOk()) { $('run').value = run; return; } run = $('run').value; current = null; cellsCache.clear(); await loadQueue(); };

    // j / k move through the queue, but never while typing
    document.addEventListener('keydown', (e) => {
        if (['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement.tagName) || e.metaKey || e.ctrlKey || e.altKey) return;
        if (e.key !== 'j' && e.key !== 'k') return;
        const i = items.findIndex((x) => current && x.doc_id === current.doc_id && x.column === current.column);
        const next = items[Math.min(items.length - 1, Math.max(0, (i < 0 ? 0 : i) + (e.key === 'j' ? 1 : -1)))];
        if (next) { e.preventDefault(); select(next); }
    });

    start().then(() => {
        const col = params.get('column');
        if (col && docFilter) {
            kind = 'all'; unrevOnly = false;
            $('unrev').checked = false;
            document.querySelectorAll('#kind button').forEach((x) => x.classList.toggle('btn-active', x.dataset.k === 'all'));
            renderQueue();
            select({doc_id: docFilter, column: col});
        }
    });
})();
