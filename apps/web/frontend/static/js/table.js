// The table workspace: Overview, Schema, Papers, Table. One table = one schema.
// Extractions are jobs the reviewer starts and watches; they are never named or listed as "runs" in the UI.
(function () {
    const {$, esc, enc, get, post, toast, store, needReviewer, dirty, stateBadge, when, shortDoc} = EVS;
    const TABLE = $('page').dataset.table;
    const api = (p) => `/api/tables/${enc(TABLE)}${p || ''}`;
    let T = null;            // /api/tables/<id>: table, papers, runs, steps, next_action
    let schema = null;       // the schema being shown (current draft or a locked version)
    let viewing = 'current'; // 'current' or a version number
    const open = new Set();  // expanded schema rows
    let noteEdit = null;
    let ruleContext = null;

    // ---------- loading ----------
    async function loadTable() {
        const r = await get(api());
        if (!r.success) { $('t-name').textContent = 'Table not found'; toast(r.error, 'error'); return false; }
        T = r;
        $('t-name').textContent = r.table.name;
        document.title = `${r.table.name} · EviSearch`;
        const locked = r.table.locked_versions || [];
        $('t-status').textContent = r.table.status === 'locked' ? `locked v${Math.max(...locked)}` : (locked.length ? `draft (after v${Math.max(...locked)})` : 'draft');
        $('t-status').className = 'badge ' + (r.table.status === 'locked' ? 'badge-success' : 'badge-warning');
        $('t-meta').textContent = `${r.table.fields} columns · ${r.papers.length} papers`;
        if (r.table.description) { $('t-desc').textContent = r.table.description; $('t-desc').classList.remove('hidden'); }
        return true;
    }

    // ---------- tabs ----------
    const TABS = ['overview', 'schema', 'papers', 'table'];
    const rendered = {};
    function showTab(name) {
        if (!TABS.includes(name)) name = 'overview';
        document.querySelectorAll('#tabs .tab').forEach((t) => t.classList.toggle('tab-active', t.dataset.tab === name));
        TABS.forEach((t) => $('p-' + t).classList.toggle('hidden', t !== name));
        if (location.hash !== '#' + name) history.replaceState(null, '', location.pathname + location.search + '#' + name);
        const fn = {overview: renderOverview, schema: renderSchemaTab, papers: renderPapers, table: renderGridTab}[name];
        if (!rendered[name]) { rendered[name] = true; fn(); }
    }
    document.querySelectorAll('#tabs .tab').forEach((t) => t.addEventListener('click', () => showTab(t.dataset.tab)));
    window.addEventListener('hashchange', () => showTab(location.hash.slice(1)));

    // ---------- Overview ----------
    function latestRun() { return T.runs.find((r) => r.run === T.showcase_run) || (T.runs.length ? T.runs[T.runs.length - 1] : null); }
    async function renderOverview() {
        $('o-steps').innerHTML = T.steps.map((s) => `<li class="step ${s.done ? 'step-primary' : ''}" data-content="${s.done ? '✓' : '•'}">
            <div class="text-left md:text-center"><div class="font-semibold">${esc(s.label)}</div><div class="opacity-60">${esc(s.detail)}</div></div></li>`).join('');
        $('o-next').textContent = T.next_action.label;
        $('o-next').href = T.next_action.href;
        $('o-next').onclick = (e) => { if (T.next_action.href.startsWith(`/tables/${TABLE}#`)) { e.preventDefault(); showTab(T.next_action.href.split('#')[1]); } };
        const run = latestRun();
        $('o-run').innerHTML = run ? `<h3 class="font-semibold">Latest results</h3>
            <div class="text-sm">${run.done} of ${run.papers.length} papers extracted · ${run.flagged} cells flagged for review · ${run.reviewed} reviewed (${run.corrected} corrected)</div>
            <div class="flex gap-2"><a class="btn btn-sm btn-primary" href="/tables/${enc(TABLE)}/review">Review flagged cells</a>
            <a class="btn btn-sm" href="#table" data-run="${esc(run.run)}" id="o-open-table">Open the table</a></div>`
            : `<h3 class="font-semibold">Nothing extracted yet</h3><p class="text-sm">Lock a version on the Schema tab, then extract your papers from the Papers tab.</p>`;
        const btn = $('o-open-table');
        if (btn) btn.onclick = (e) => { e.preventDefault(); gridRun = btn.dataset.run; showTab('table'); };
        const ev = await get(`/api/feedback/events?schema_id=${enc(TABLE)}&limit=10`);
        $('o-activity').innerHTML = (ev.events || []).map((e) => `<li><span class="opacity-60">${esc(when(e.timestamp))}</span> ${e.by ? esc(e.by) + ' ' : ''}${esc(describeEvent(e))}</li>`).join('')
            || '<li class="opacity-60">Nothing yet.</li>';
    }
    function describeEvent(e) {
        const col = e.column ? ` “${e.column}”` : '';
        switch (e.event) {
            case 'definition_accept': return `accepted the definition of${col}`;
            case 'definition_edit': return `edited${col}${e.reason ? ' (' + e.reason + ')' : ''}`;
            case 'definition_answer': return `answered a question on${col}: ${e.answer}`;
            case 'definition_revise': return `agent revised${col}`;
            case 'schema_draft': return `created the table with ${e.fields} column${e.fields === 1 ? '' : 's'}`;
            case 'schema_lock': return `locked v${e.version}`;
            case 'cell_correct': return `${e.state === 'accepted' || e.before === e.after ? 'confirmed' : 'corrected'}${col} in ${shortDoc(e.doc_id)}${e.reason ? ' (' + e.reason + ')' : ''}`;
            case 'cell_undo': return `undid a review of${col}`;
            case 'convention_propose': return `proposed rule ${e.convention}`;
            case 'convention_decide': return `${e.op}d rule ${e.convention}`;
            case 'extraction_start': return 'started an extraction';
            case 'extraction_end': return `an extraction ${e.exit_code === 0 ? 'finished' : 'stopped (exit ' + e.exit_code + ')'}`;
            default: return e.event || e.source || 'event';
        }
    }

    // ---------- Schema ----------
    async function renderSchemaTab() {
        const locked = T.table.locked_versions || [];
        $('s-version').innerHTML = `<option value="current">Current (${T.table.status === 'locked' ? 'same as v' + Math.max(...locked) : 'draft'})</option>` +
            locked.slice().reverse().map((v) => `<option value="${v}">v${v}${v === 1 ? ' · definitions the owner locked' : ' (locked)'}</option>`).join('');
        $('s-dl').innerHTML = locked.length ? `<div class="font-semibold">Download a locked version</div>
            ${locked.slice().reverse().map((v) => `<div>v${v}: <a class="link" href="${api(`/versions/${v}.csv`)}">CSV</a> · <a class="link" href="${api(`/versions/${v}.xlsx`)}">Excel</a></div>`).join('')}
            <div class="font-semibold pt-2">Compare versions</div>
            <div class="flex gap-1 items-center"><select id="d-a" class="select select-xs select-bordered">${versionOptions(locked, locked.length > 1 ? locked[locked.length - 2] : locked[0])}</select> →
            <select id="d-b" class="select select-xs select-bordered">${versionOptions(locked, 'current')}</select>
            <button id="d-go" class="btn btn-xs">Show</button></div>` : '<div class="opacity-60">No locked version yet.</div>';
        const go = $('d-go');
        if (go) go.onclick = () => showDiff($('d-a').value, $('d-b').value);
        await loadSchema('current');
    }
    function versionOptions(locked, selected) {
        return ['current', ...locked.slice().reverse()].map((v) => `<option value="${v}" ${String(v) === String(selected) ? 'selected' : ''}>${v === 'current' ? 'current' : 'v' + v}</option>`).join('');
    }
    async function fetchSchema(v) {
        const r = await get(`/api/schemas/${enc(TABLE)}${v === 'current' ? '' : '?version=' + v}`);
        if (!r.success) throw new Error(r.error);
        return r.schema;
    }
    async function loadSchema(v) {
        viewing = v;
        try { schema = await fetchSchema(v); } catch (e) { $('s-rows').innerHTML = `<div class="p-4 text-error">${esc(e.message)}</div>`; return; }
        const ro = v !== 'current';
        $('s-lock').disabled = ro;
        renderRows();
    }
    $('s-version').addEventListener('change', () => loadSchema($('s-version').value));
    $('s-search').addEventListener('input', () => schema && renderRows());

    const draftKey = (col) => `draft.${TABLE}.${col}`;
    function matches(f) {
        const q = $('s-search').value.toLowerCase();
        return !q || f.name.toLowerCase().includes(q) || f.description.toLowerCase().includes(q);
    }
    function renderRows() {
        const shown = schema.fields.filter(matches);
        $('s-count').textContent = `${shown.length} of ${schema.fields.length}`;
        $('s-rows').innerHTML = shown.map((f) => `<div class="srow" data-col="${esc(f.name)}"></div>`).join('') || '<div class="p-4 opacity-60">No column matches.</div>';
        $('s-rows').querySelectorAll('.srow').forEach((row) => renderRow(row, schema.fields.find((f) => f.name === row.dataset.col)));
    }
    function renderRow(row, f) {
        const x = f['x-evisearch'], fx = x.facets || {}, ro = viewing !== 'current';
        const draft = ro ? null : store.get(draftKey(f.name));
        const isOpen = open.has(f.name);
        row.innerHTML = `<div class="row-head flex items-center gap-2 px-3 py-2 hover:bg-base-200">
                <span class="opacity-50 w-3">${isOpen ? '▾' : '▸'}</span>
                <span class="mono text-sm font-semibold min-w-[14rem] max-w-[22rem] truncate" title="${esc(f.name)}">${esc(f.name)}</span>
                ${draft ? '<span class="badge badge-sm badge-accent">unsaved draft</span>' : ''}
                <span class="def-1 text-sm opacity-70 flex-1">${esc(f.description)}</span></div>
            ${isOpen ? `<div class="px-8 pb-4 space-y-2">
                <div class="flex flex-wrap gap-1">${['characteristic', 'statistic', 'unit', 'subgroup', 'arm', 'category'].filter((k) => fx[k]).map((k) => `<span class="badge badge-sm badge-outline">${k}: ${esc(fx[k])}</span>`).join(' ')}
                    ${fx.cryptic && fx.cryptic.length ? `<span class="badge badge-sm badge-warning">unclear header: ${esc(fx.cryptic.join(', '))}</span>` : ''}
                    <span class="badge badge-sm badge-ghost">group: ${esc(x.group || '—')}</span></div>
                ${ro ? `<div class="text-sm whitespace-pre-wrap border border-base-300 rounded p-2">${esc(f.description)}</div>`
                     : `<textarea class="textarea textarea-bordered w-full text-sm def" rows="3">${esc(draft ? draft.text : f.description)}</textarea>
                        ${draft ? `<div class="text-xs text-accent">Restored your unsaved text from ${esc(when(draft.at))}. Save it or discard it.</div>` : ''}`}
                <div class="text-xs opacity-60">Answer format: ${esc(x.answer_format || '—')} · scoring: ${esc(x.eval_category || '—')} · “Not reported” when: ${esc(x.nr_policy || '—')} · confidence ${esc(x.confidence || '—')}</div>
                ${ro ? '' : `<div class="flex flex-wrap gap-2 items-center">
                    <select class="select select-xs select-bordered why"><option value="">why you edit it (required)…</option>
                        <option>wrong statistic</option><option>wrong population or subgroup</option><option>wrong arm</option>
                        <option>missing convention</option><option>unit or format</option><option>wording</option></select>
                    <input class="input input-xs input-bordered note w-64" placeholder="note (optional)" />
                    <button class="btn btn-xs btn-primary sav" ${draft ? '' : 'disabled'}>Save edit</button>
                    <button class="btn btn-xs dis" ${draft ? '' : 'disabled'}>Discard</button>
                    <button class="btn btn-xs btn-outline rule">Propose rule…</button></div>`}
            </div>` : ''}`;
        row.querySelector('.row-head').onclick = () => { isOpen ? open.delete(f.name) : open.add(f.name); renderRow(row, f); };
        if (!isOpen || ro) return;
        const ta = row.querySelector('.def');
        const setDirty = () => {
            const changed = ta.value !== f.description;
            if (changed) { store.set(draftKey(f.name), {text: ta.value, at: new Date().toISOString()}); dirty.add('schema:' + f.name); }
            else { store.del(draftKey(f.name)); dirty.delete('schema:' + f.name); }
            row.querySelector('.sav').disabled = !changed;
            row.querySelector('.dis').disabled = !changed;
        };
        ta.addEventListener('input', setDirty);
        const act = async (body, btn) => {
            const by = needReviewer(); if (!by) return;
            if (btn) btn.disabled = true;
            const r = await post(`/api/schemas/${enc(TABLE)}/review`, {column: f.name, by, ...body});
            if (!r.success) { toast(r.error, 'error'); if (btn) btn.disabled = false; return; }
            const i = schema.fields.findIndex((ff) => ff.name === f.name);
            schema.fields[i] = r.field;
            if (body.action === 'edit') { store.del(draftKey(f.name)); dirty.delete('schema:' + f.name); }
            renderRow(row, r.field);
            toast('Definition saved', 'success', 2000);
            T.table.status = 'draft';
        };
        row.querySelector('.sav').onclick = (e) => {
            const reason = row.querySelector('.why').value;
            if (!reason) return toast('Pick why you edited the definition', 'warning');
            act({action: 'edit', definition: ta.value, reason, note: row.querySelector('.note').value}, e.target);
        };
        row.querySelector('.dis').onclick = () => { store.del(draftKey(f.name)); dirty.delete('schema:' + f.name); renderRow(row, f); };
        row.querySelector('.rule').onclick = () => {
            ruleContext = {column: f.name, definition: ta.value, schema_id: TABLE, doc_id: T.table.example_doc, kind: 'schema_review'};
            noteEdit = null;
            $('rule-col').textContent = f.name;
            $('rule-note').value = row.querySelector('.note').value;
            $('rule-out').innerHTML = '';
            $('rule-add').disabled = true;
            $('rule-dlg').showModal();
        };
    }

    $('rule-draft').onclick = async () => {
        const by = needReviewer(); if (!by) return;
        if (!$('rule-note').value.trim()) return toast('Write what the knowledge should say', 'warning');
        $('rule-add').disabled = true;
        noteEdit = await NoteEdit.draft($('rule-out'), ruleContext, $('rule-note').value, by);
        $('rule-add').disabled = !noteEdit;
    };
    $('rule-add').onclick = async () => {
        const by = needReviewer(); if (!by || !noteEdit) return;
        const r = await NoteEdit.apply(noteEdit, {by, schema_id: TABLE});
        $('rule-out').insertAdjacentHTML('beforeend', `<div class="alert mt-2 text-sm">${NoteEdit.done(r)}</div>`);
        $('rule-add').disabled = true;
    };


    $('s-lock').onclick = async () => {
        const locked = T.table.locked_versions || [];
        const fields = schema.fields;
        let changed = fields.length;
        if (locked.length) {
            const last = await fetchSchema(Math.max(...locked));
            const prev = Object.fromEntries(last.fields.map((f) => [f.name, f.description]));
            changed = fields.filter((f) => prev[f.name] !== f.description).length + last.fields.filter((f) => !fields.some((g) => g.name === f.name)).length;
        }
        const next = T.table.version;
        $('lock-body').innerHTML = `<p>Locking saves the current definitions as <b>v${next}</b>. Papers are always extracted under a locked version, so results stay traceable.</p>
            <ul class="list-disc ml-5">
                <li>${changed} definition${changed === 1 ? '' : 's'} ${locked.length ? `changed since v${Math.max(...locked)}` : 'in this first version'}</li>
            </ul>${locked.length && !changed ? `<div class="alert alert-info text-sm">Nothing changed since v${Math.max(...locked)}; there is nothing to lock.</div>` : ''}`;
        $('lock-go').disabled = locked.length > 0 && changed === 0;
        $('lock-dlg').showModal();
    };
    $('lock-go').onclick = async () => {
        const by = needReviewer(); if (!by) return;
        const r = await post(`/api/schemas/${enc(TABLE)}/lock`, {by});
        $('lock-dlg').close();
        if (!r.success) return toast(r.error, 'error');
        toast(`Locked v${r.version} — papers can be extracted under it now.`, 'success', 6000);
        await loadTable(); rendered.schema = false; renderSchemaTab();
    };

    async function showDiff(a, b) {
        $('s-diff').classList.remove('hidden');
        $('s-diff-body').textContent = 'Comparing…';
        try {
            const [sa, sb] = await Promise.all([fetchSchema(a), fetchSchema(b)]);
            const A = Object.fromEntries(sa.fields.map((f) => [f.name, f.description]));
            const B = Object.fromEntries(sb.fields.map((f) => [f.name, f.description]));
            const names = [...new Set([...Object.keys(A), ...Object.keys(B)])];
            const changes = names.filter((n) => A[n] !== B[n]);
            const label = (v) => v === 'current' ? 'current' : 'v' + v;
            $('s-diff-body').innerHTML = `<div class="flex justify-between"><b>${label(a)} → ${label(b)}: ${changes.length} of ${names.length} definitions differ</b>
                <button class="btn btn-xs" id="diff-close">Close</button></div>` + changes.map((n) => `<div class="border-t border-base-300 pt-2 mt-2">
                <div class="mono text-xs font-semibold">${esc(n)}</div>
                <div class="text-sm">${A[n] === undefined ? '<i>added</i> ' + esc(B[n]) : B[n] === undefined ? '<i>removed</i>' : wordDiff(A[n], B[n])}</div></div>`).join('');
            $('diff-close').onclick = () => $('s-diff').classList.add('hidden');
        } catch (e) { $('s-diff-body').textContent = e.message; }
    }
    function wordDiff(a, b) {  // longest common subsequence over words
        const x = a.split(/(\s+)/), y = b.split(/(\s+)/);
        const n = x.length, m = y.length, L = Array.from({length: n + 1}, () => new Array(m + 1).fill(0));
        for (let i = n - 1; i >= 0; i--) for (let j = m - 1; j >= 0; j--) L[i][j] = x[i] === y[j] ? L[i + 1][j + 1] + 1 : Math.max(L[i + 1][j], L[i][j + 1]);
        let i = 0, j = 0, out = '';
        while (i < n && j < m) {
            if (x[i] === y[j]) { out += esc(x[i]); i++; j++; }
            else if (L[i + 1][j] >= L[i][j + 1]) { out += `<del>${esc(x[i])}</del>`; i++; }
            else { out += `<ins>${esc(y[j])}</ins>`; j++; }
        }
        while (i < n) out += `<del>${esc(x[i++])}</del>`;
        while (j < m) out += `<ins>${esc(y[j++])}</ins>`;
        return out;
    }

    async function waitJob(id, el, label) {
        for (;;) {
            const r = await get(`/api/jobs/${id}`);
            const job = r.job || {status: 'error', error: r.error};
            el.innerHTML = `${esc(label)}: <b>${esc(job.status)}</b>${job.error ? ' — ' + esc(job.error) : ''}`;
            if (!['queued', 'running'].includes(job.status)) return job;
            await new Promise((res) => setTimeout(res, 2500));
        }
    }

    // ---------- Papers ----------
    async function renderPapers() {
        const latest = latestRun();
        const statusIn = (doc) => { const p = latest && latest.papers.find((x) => x.doc_id === doc); return p ? p.status : null; };
        $('pp-table').innerHTML = `<thead><tr><th>Paper</th><th>Role</th><th>PDF</th><th>Parsed</th><th>Extracted</th><th></th></tr></thead><tbody>` +
            T.papers.map((p) => `<tr><td><div class="font-medium">${esc(shortDoc(p.name))}</div><div class="mono text-xs opacity-60">${esc(p.doc_id)}</div></td>
                <td>${p.role === 'example' ? '<span class="badge badge-accent badge-sm">example paper</span>' : 'paper'}${p.gold ? ' <span class="badge badge-outline badge-sm" title="Has expert gold values (benchmark)">gold</span>' : ''}</td>
                <td>${p.pdf ? `<a class="link" target="_blank" href="/api/documents/${enc(p.doc_id)}/pdf">open</a>` : '<span class="text-error">missing</span>'}</td>
                <td>${p.parsed ? '✓' : '<span class="badge badge-warning badge-sm">not yet</span>'}</td>
                <td>${statusIn(p.doc_id) ? stateChip(statusIn(p.doc_id)) : '<span class="opacity-50">—</span>'}</td>
                <td>${!p.parsed && p.pdf ? `<button class="btn btn-xs prep" data-doc="${esc(p.doc_id)}">Parse now</button>` : ''}</td></tr>`).join('') + '</tbody>';
        $('pp-table').querySelectorAll('.prep').forEach((b) => b.onclick = async () => {
            const r = await post('/api/documents/prepare', {doc_id: b.dataset.doc, by: needReviewer() || ''});
            if (!r.success) return toast(r.error, 'error');
            b.outerHTML = '<span class="text-xs">parsing…</span>';
            pollJobs();
        });
        renderJobs();
        const lib = await get('/api/documents/selectable');
        const have = new Set(T.papers.map((p) => p.doc_id));
        $('pp-lib').innerHTML = '<option value="">Add a paper from the library…</option>' + (lib.documents || []).filter((d) => !have.has(d.id)).map((d) =>
            `<option value="${esc(d.id)}">${esc(shortDoc(d.name))}${d.has_cached_parse ? '' : ' (not parsed)'}</option>`).join('');
    }
    function stateChip(s) {
        const cls = {ok: 'badge-success', running: 'badge-info', waiting: 'badge-ghost', failed: 'badge-error', error: 'badge-error'}[s] || 'badge-warning';
        return `<span class="badge badge-sm ${cls}">${esc(s === 'ok' ? 'done' : s)}</span>`;
    }
    async function addPaper(body) {
        const r = body instanceof FormData ? await fetch(api('/papers'), {method: 'POST', body}).then((x) => x.json()) : await post(api('/papers'), body);
        if (!r.success) return toast(r.error, 'error');
        toast(r.job ? `Added ${shortDoc(r.paper.name)}; parsing it now` : `Added ${shortDoc(r.paper.name)}`, 'success');
        await loadTable(); renderPapers(); if (r.job) pollJobs();
    }
    $('pp-add').onclick = () => { const d = $('pp-lib').value; if (!d) return toast('Pick a paper', 'warning'); addPaper({doc_id: d, by: needReviewer() || ''}); };
    $('pp-upload').onclick = () => {
        const f = $('pp-file').files[0]; if (!f) return toast('Choose a PDF', 'warning');
        const fd = new FormData(); fd.append('file', f); fd.append('by', needReviewer() || ''); addPaper(fd);
    };

    // ---------- Extraction (jobs only: the reviewer never picks or names one) ----------
    let polling = null;
    async function renderJobs() {
        const r = await get(api('/runs'));
        if (!r.success) return toast(r.error, 'error');
        T.runs = r.runs;
        const active = (r.jobs || []).filter((j) => ['queued', 'running'].includes(j.status));
        const recent = (r.jobs || []).filter((j) => !['queued', 'running'].includes(j.status)).slice(0, 2);
        const paper = (j) => `${j.docs.length} paper${j.docs.length > 1 ? 's' : ''}`;
        $('x-jobs').innerHTML = [...active, ...recent].map((j) => `<div class="card bg-base-100 shadow"><div class="card-body p-3 text-sm space-y-1">
            <div class="flex flex-wrap items-center gap-2"><b>${j.status === 'running' || j.status === 'queued' ? 'Extracting' : 'Extraction'}</b>${stateChip(j.status === 'done' ? 'ok' : j.status)}
                <span class="opacity-60">${esc(paper(j))} · definitions v${esc(j.version)} · started ${esc(when(j.started))} by ${esc(j.by || '—')}</span>
                ${j.status === 'running' ? `<button class="btn btn-xs btn-error btn-outline cancel" data-id="${esc(j.id)}">Stop</button>` : ''}
                <button class="btn btn-xs logb" data-id="${esc(j.id)}">Log</button></div>
            ${j.error ? `<div class="text-error">${esc(j.error)}</div>` : ''}
            ${progressOf(j)}
            <pre class="text-xs bg-base-200 p-2 rounded max-h-48 overflow-auto hidden" id="log-${esc(j.id)}"></pre></div></div>`).join('')
            || '<div class="text-sm opacity-60">Nothing is extracting right now.</div>';
        $('x-jobs').querySelectorAll('.cancel').forEach((b) => b.onclick = async () => {
            if (!confirm('Stop the extraction? Papers already finished keep their results.')) return;
            const res = await post(`/api/jobs/${b.dataset.id}/cancel`, {by: needReviewer() || ''});
            toast(res.success ? 'Extraction stopped' : res.error, res.success ? 'success' : 'error'); renderJobs();
        });
        $('x-jobs').querySelectorAll('.logb').forEach((b) => b.onclick = async () => {
            const pre = $('log-' + b.dataset.id);
            const res = await get(`/api/jobs/${b.dataset.id}/log?lines=80`);
            pre.textContent = res.log || '(empty)'; pre.classList.toggle('hidden');
        });
        if (active.length && !polling) polling = setInterval(() => { if (!document.hidden) renderJobs(); }, 8000);
        if (!active.length && polling) { clearInterval(polling); polling = null; }
    }
    // per-paper progress of the extraction this job started
    function progressOf(job) {
        const rec = (T.runs || []).find((x) => x.run === job.run);
        if (!rec) return '';
        return `<div class="space-y-1">${rec.papers.map((p) => `<div class="flex flex-wrap gap-2 items-center text-xs">
            <span class="w-72 truncate" title="${esc(p.doc_id)}">${esc(shortDoc(p.name))}</span>${stateChip(p.status)}
            ${Object.entries(p.stages || {}).map(([k, st]) => `<span class="badge badge-sm badge-outline">${esc({agent: 'Agent A', search: 'Agent B', reconciliation: 'Reconciliation'}[k] || k)}: ${st.batches_done !== undefined ? `${st.batches_done}/${st.batches}` : esc(st.status)}${st.duration_s ? ' · ' + Math.round(st.duration_s / 60) + ' min' : ''}</span>`).join('')}
            ${p.flagged ? `<span class="badge badge-warning badge-sm">${p.flagged} flagged</span>` : ''}${p.reviewed ? `<span class="badge badge-info badge-sm">${p.reviewed} reviewed</span>` : ''}
            ${p.error ? `<span class="text-error">${esc(p.error)}</span>` : ''}</div>`).join('')}</div>`;
    }
    function pollJobs() { if (!polling) polling = setInterval(async () => { await loadTable(); if (location.hash.slice(1) === 'papers') { renderPapers(); } }, 8000); }

    $('x-new').onclick = () => {
        const locked = T.table.locked_versions || [];
        if (!locked.length) return toast('Lock a schema version first (Schema tab)', 'warning');
        $('rd-version').innerHTML = locked.slice().reverse().map((v) => `<option value="${v}">v${v}</option>`).join('');
        $('rd-papers').innerHTML = T.papers.map((p) => `<label class="flex items-center gap-2 ${p.parsed ? '' : 'opacity-50'}">
            <input type="checkbox" class="checkbox checkbox-xs rd-p" value="${esc(p.doc_id)}" ${p.parsed ? '' : 'disabled'} />
            <span>${esc(shortDoc(p.name))}</span>${p.gold ? '<span class="badge badge-outline badge-xs">gold</span>' : ''}${p.parsed ? '' : '<span class="text-xs">(parse it above first)</span>'}</label>`).join('');
        const est = () => {
            const n = document.querySelectorAll('.rd-p:checked').length, x = T.extraction || {minutes_per_paper: 20, papers_at_once: 2};
            if (!n) { $('rd-est').textContent = 'Pick the papers to extract.'; return; }
            const waves = Math.ceil(n / x.papers_at_once), mins = Math.round(x.minutes_per_paper * waves * (x.hosted && n > 1 ? 1.5 : 1));
            const where = x.hosted ? `${x.model} via ${x.preset}, every batch and paper in parallel` : `the local GPU, ${x.papers_at_once} papers at a time`;
            const cost = x.usd_per_paper ? `, about $${(x.usd_per_paper * n).toFixed(2)}` : '';
            $('rd-est').textContent = `${n} paper${n > 1 ? 's' : ''}: about ${mins} min on ${where}${cost}.`;
        };
        document.querySelectorAll('.rd-p').forEach((c) => c.onchange = est);
        $('rd-all').onclick = () => { document.querySelectorAll('.rd-p:not(:disabled)').forEach((c) => c.checked = true); est(); };
        $('rd-none').onclick = () => { document.querySelectorAll('.rd-p').forEach((c) => c.checked = false); est(); };
        est();
        $('run-dlg').showModal();
    };
    $('rd-go').onclick = async () => {
        const by = needReviewer(); if (!by) return;
        const docs = [...document.querySelectorAll('.rd-p:checked')].map((c) => c.value);
        if (!docs.length) return toast('Pick at least one paper', 'warning');
        $('rd-go').disabled = true;
        const r = await post(api('/runs'), {docs, version: Number($('rd-version').value), by});
        $('rd-go').disabled = false;
        if (!r.success) return toast(r.error, 'error', 7000);
        $('run-dlg').close();
        toast(`Extracting ${docs.length} paper${docs.length > 1 ? 's' : ''} — progress appears below`, 'success');
        await loadTable(); renderJobs(); pollJobs();
    };

    // ---------- Table ----------
    let gridRun = new URLSearchParams(location.search).get('run');
    let grid = null;
    function renderGridTab() {
        const runs = T.runs.slice().reverse();
        if (!runs.length) { $('g-grid').innerHTML = '<div class="p-6 text-sm">Nothing extracted yet — start it on the Papers tab.</div>'; return; }
        if (!gridRun || !runs.some((r) => r.run === gridRun)) gridRun = (T.showcase_run && runs.some((r) => r.run === T.showcase_run) ? T.showcase_run : (runs.find((r) => r.status === 'ok' && !r.variant) || runs[0]).run);
        const rec = runs.find((r) => r.run === gridRun);
        $('g-what').textContent = rec ? `latest results · ${rec.done} of ${rec.papers.length} papers · ${esc(when(rec.started_at))}` : '';
        loadGrid();
    }
    ['g-group', 'g-search', 'g-flagged'].forEach((id) => $(id).addEventListener('input', drawGrid));
    async function loadGrid() {
        $('g-grid').innerHTML = '<div class="p-6 text-sm opacity-60">Loading…</div>';
        const r = await get(`/api/runs/${enc(gridRun)}/table`);
        if (!r.success) { $('g-grid').innerHTML = `<div class="p-6 text-error">${esc(r.error)}</div>`; return; }
        grid = r;
        $('g-csv').href = `/api/runs/${enc(gridRun)}/export.csv`;
        $('g-xlsx').href = `/api/runs/${enc(gridRun)}/export.xlsx`;
        const groups = [...new Set(r.columns.map((c) => c.group))];
        $('g-group').innerHTML = '<option value="">All column groups</option>' + groups.map((g) => `<option>${esc(g)}</option>`).join('');
        drawGrid();
    }
    function drawGrid() {
        if (!grid) return;
        const g = $('g-group').value, q = $('g-search').value.toLowerCase(), onlyFlag = $('g-flagged').checked;
        const cols = grid.columns.filter((c) => (!g || c.group === g) && (!q || c.name.toLowerCase().includes(q))
            && (!onlyFlag || grid.docs.some((d) => ['flagged', 'corrected', 'accepted'].includes((d.cells[c.name] || {}).s))));
        const counts = {};
        grid.docs.forEach((d) => cols.forEach((c) => { const s = (d.cells[c.name] || {}).s; if (s) counts[s] = (counts[s] || 0) + 1; }));
        $('g-legend').innerHTML = Object.entries(EVS.STATE).map(([k, [label]]) => `<span class="flex items-center gap-1"><span class="dot s-${k} border border-base-content/30"></span>${esc(label)} <b>${counts[k] || 0}</b></span>`).join('')
            + `<span class="opacity-60">definitions: ${esc(grid.definitions.source)}</span>`;
        const groupsRow = []; cols.forEach((c) => { const last = groupsRow[groupsRow.length - 1]; if (last && last.g === c.group) last.n++; else groupsRow.push({g: c.group, n: 1}); });
        $('g-grid').innerHTML = `<table class="text-xs"><thead>
            <tr><th class="sticky-col px-2 py-1 text-left">Paper</th>${groupsRow.map((x) => `<th colspan="${x.n}" class="px-2 py-1 text-left font-semibold truncate">${esc(x.g)}</th>`).join('')}</tr>
            <tr class="cols"><th class="sticky-col px-2 py-1"></th>${cols.map((c) => { const label = c.name.split(' | ').slice(1).join(' | ');
                return `<th class="px-2 py-1 text-left font-normal min-w-[8rem] max-w-[14rem]" title="${esc(c.name + '\n\n' + c.definition)}"><div class="line-clamp-3">${esc(label || (c.name === c.group ? '' : c.name))}</div></th>`; }).join('')}</tr></thead>
            <tbody>${grid.docs.map((d) => `<tr><td class="sticky-col px-2 py-1 font-medium whitespace-nowrap" title="${esc(d.doc_id)}">${esc(shortDoc(d.name))}</td>
                ${cols.map((c) => { const cell = d.cells[c.name]; if (!cell) return '<td class="px-2 py-1 opacity-30">—</td>';
                    const tip = `${c.name}\n${EVS.STATE[cell.s] ? EVS.STATE[cell.s][0] : cell.s}${cell.p ? ' · page ' + cell.p : ''}${cell.m !== null && cell.m !== undefined ? '\nmachine value: ' + cell.m : ''}${cell.q ? '\n“' + cell.q + '”' : ''}`;
                    return `<td class="px-2 py-1 s-${cell.s}"><div class="cellv" data-doc="${esc(d.doc_id)}" data-col="${esc(c.name)}" title="${esc(tip)}">${esc(cell.v)}</div></td>`; }).join('')}</tr>`).join('')}</tbody></table>`;
        $('g-grid').querySelectorAll('.cellv').forEach((el) => el.onclick = () => {
            location.href = `/tables/${enc(TABLE)}/review?run=${enc(gridRun)}&doc=${enc(el.dataset.doc)}&column=${enc(el.dataset.col)}`;  // run is carried silently
        });
    }

    // ---------- start ----------
    (async function () {
        if (!(await loadTable())) return;
        showTab(location.hash.slice(1) || 'overview');
        if ((T.runs || []).some((r) => r.status === 'running')) pollJobs();
    })();
})();
