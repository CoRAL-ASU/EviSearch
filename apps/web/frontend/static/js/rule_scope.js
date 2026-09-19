// A drafted convention with its integrity-gate verdict, shared by the Schema and Verify pages. The reviewer can change
// where the rule applies (column / family / table / global, header family, columns) and re-check the gate before adding
// it: the rule proposer tends to keep a rule inside the column family it was drafted from.
window.RuleScope = (function () {
    const esc = (s) => { const d = document.createElement('div'); d.textContent = s == null ? '' : String(s); return d.innerHTML; };
    const SCOPES = ['column', 'family', 'table', 'global'];

    function gateHtml(g, impact) {
        const cls = g.verdict === 'blocked' ? 'badge-error' : g.verdict === 'duplicate' ? 'badge-info' : 'badge-success';
        const rel = (g.relations || []).filter((x) => x.relation !== 'independent').map((x) =>
            `<li><span class="badge badge-sm">${esc(x.relation)}</span> ${esc(x.id)}: ${esc((x.instruction || '').slice(0, 140))} (<span class="opacity-60">${esc(x.reason)}</span>)</li>`).join('');
        return `<div><b>Gate:</b> <span class="badge ${cls}">${esc(g.verdict)}</span></div>
            ${rel ? `<ul class="list-disc ml-5 text-xs">${rel}</ul>` : ''}
            <div class="text-xs"><b>Applies to ${impact.length} column(s):</b> ${esc(impact.slice(0, 12).join(' · '))}${impact.length > 12 ? ' …' : ''}</div>`;
    }

    // Render `result` ({record, gate, impact}) into `el`; state.record always holds the record the gate last checked.
    function show(el, result, state, schemaId, onVerdict) {
        state.record = result.record;
        const t = result.record.trigger || {};
        el.innerHTML = `<div><b>Rule:</b> ${esc(result.record.instruction)}</div>
            <div class="text-xs">action ${esc(result.record.action.type)}${t.condition ? ' · when: ' + esc(t.condition) : ''}</div>
            <div class="flex flex-wrap gap-2 items-center text-xs mt-1">
                <span>Applies to</span>
                <select class="select select-xs select-bordered rs-scope">${SCOPES.map((s) => `<option ${s === t.scope ? 'selected' : ''}>${s}</option>`).join('')}</select>
                <input class="input input-xs input-bordered rs-family w-56" placeholder="header family, e.g. Mode of metastases" value="${esc(t.family || '')}" />
                <button type="button" class="btn btn-xs rs-check">Re-check scope</button>
            </div>
            <textarea class="textarea textarea-bordered textarea-xs w-full rs-columns mt-1" rows="2" placeholder="extra columns, one per line">${esc((t.columns || []).join('\n'))}</textarea>
            <div class="rs-gate"></div>`;
        const gate = el.querySelector('.rs-gate');
        const paint = (g, impact) => { gate.innerHTML = gateHtml(g, impact || []); onVerdict(g.verdict); };
        paint(result.gate, result.impact);
        el.querySelector('.rs-check').addEventListener('click', async () => {
            const scope = el.querySelector('.rs-scope').value;
            const family = el.querySelector('.rs-family').value.trim();
            const columns = el.querySelector('.rs-columns').value.split('\n').map((s) => s.trim()).filter(Boolean);
            const record = {...state.record, trigger: {...state.record.trigger, scope, family: family || null,
                                                       columns: scope === 'column' || scope === 'family' ? columns : []}};
            gate.textContent = 'Checking…';
            onVerdict('checking');
            const r = await fetch('/api/conventions/check', {method: 'POST', headers: {'Content-Type': 'application/json'},
                                                              body: JSON.stringify({record, schema_id: schemaId})}).then((x) => x.json());
            if (!r.success) { gate.textContent = r.error || 'check failed'; onVerdict('blocked'); return; }
            state.record = record;
            paint(r.gate, r.impact);
        });
    }

    return {show};
})();
