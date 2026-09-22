// A reviewer's note, turned into an edit to one knowledge note. Shared by the Table page (schema review) and the Review
// page (a corrected cell). The proposer names the note it would change, or a new one, and drafts the text; the reviewer
// may rewrite it before it is written. Nothing is stored until the reviewer adds it, and every later extraction reads it.
window.NoteEdit = (function () {
    const esc = (s) => { const d = document.createElement('div'); d.textContent = s == null ? '' : String(s); return d.innerHTML; };
    const post = (url, body) => fetch(url, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)})
        .then((r) => r.json()).catch((e) => ({success: false, error: String(e)}));

    // Draft an edit for `ctx` ({column, definition, before?, after?, reason?}) from the reviewer's `feedback`. Renders into
    // `el` and resolves to the edit (or null when the note is not general knowledge or the call failed).
    async function draft(el, ctx, feedback, by) {
        el.textContent = 'Drafting the edit and finding the notes that govern this column…';
        const r = await post('/api/notes/propose', {...ctx, feedback, by});
        if (!r.success) { el.textContent = r.error; return null; }
        if (!r.is_knowledge) {
            el.innerHTML = `<div class="alert text-sm">Not general knowledge${r.why ? ': ' + esc(r.why) : ''}. The correction alone is kept.</div>`;
            return null;
        }
        const p = r.proposal, gov = r.governing || [];
        el.innerHTML = `
            <div class="text-xs text-base-content/70">${p.new_note ? 'New note' : 'Edit to note'} <span class="mono">${esc(p.note)}</span>
                ${p.heading ? ` · section “${esc(p.heading)}”` : ''} · ${esc(p.role === 'definitions' ? 'what the column means' : 'how the value is found')}</div>
            <textarea class="textarea textarea-bordered w-full text-sm ne-text" rows="5">${esc(p.text)}</textarea>
            ${p.why ? `<div class="text-xs text-base-content/70">${esc(p.why)}</div>` : ''}
            ${gov.length ? `<details class="text-xs"><summary class="cursor-pointer text-base-content/70">Notes that govern this column now (${gov.length})</summary>
                <ul class="mt-1 space-y-0.5">${gov.map((n) => `<li><span class="mono">${esc(n.id)}</span> <span class="opacity-60">${esc(n.scope)} · ${esc(n.role)}</span></li>`).join('')}</ul></details>` : ''}`;
        return {note: p.note, heading: p.heading || '', role: p.role, why: p.why || '', text: () => el.querySelector('.ne-text').value};
    }

    // Write the edit; `extra` carries {by, schema_id, event}. Resolves to the API response.
    function apply(edit, extra) {
        return post('/api/notes/apply', {note: edit.note, text: edit.text(), heading: edit.heading, role: edit.role, why: edit.why, ...extra});
    }

    const done = (r) => r.success ? `${r.created ? 'Created' : 'Updated'} note <span class="mono">${esc(r.note)}</span>. Every later extraction reads it.`
        : esc(r.error);

    return {draft, apply, done};
})();
