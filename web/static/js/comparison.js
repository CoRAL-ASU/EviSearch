/**
 * CoRal Comparison Dashboard — Document list, 3-method cards, sidebar, group/column detail, PDF highlights
 */
(function () {
    const DASHBOARD_METHODS = ['landing_ai_baseline', 'gemini_native', 'pipeline'];
    const API = {
        documents: () => fetch('/api/documents').then(r => r.json()),
        dashboard: (docId) => fetch(`/api/documents/${encodeURIComponent(docId)}/dashboard`).then(r => r.json()),
        highlights: (docId, columnName) => fetch(`/api/documents/${encodeURIComponent(docId)}/highlights?column_name=${encodeURIComponent(columnName)}`).then(r => r.json()),
        pdf: (docId) => `/api/documents/${encodeURIComponent(docId)}/pdf`,
    };

    let currentDocId = null;
    let dashboardData = null;
    let pdfDoc = null;
    let pendingHighlightColumn = null;

    const METHOD_LABELS = {
        gemini_native: 'Gemini Native',
        landing_ai_baseline: 'Landing AI',
        pipeline: 'Pipeline',
    };

    const METHOD_CLASS = {
        gemini_native: 'method-gemini',
        landing_ai_baseline: 'method-landing',
        pipeline: 'method-pipeline',
    };

    function escapeHtml(s) {
        if (!s) return '';
        const div = document.createElement('div');
        div.textContent = s;
        return div.innerHTML;
    }

    function confidenceClass(c) {
        if (!c) return '';
        const x = c.toLowerCase();
        if (x === 'high') return 'confidence-high';
        if (x === 'medium') return 'confidence-medium';
        return 'confidence-low';
    }

    function buildDashboardFromComparison(compRes, reportRes) {
        const byGroup = compRes.by_group || {};
        const comparison = compRes.comparison || [];
        const methods = (compRes.methods_available || []).filter(m => DASHBOARD_METHODS.includes(m));
        if (methods.length === 0) methods.push(...DASHBOARD_METHODS);

        const byMethod = {};
        for (const m of methods) {
            const stats = reportRes.by_method && reportRes.by_method[m];
            const total = stats ? stats.total : 0;
            const found = stats ? stats.found : 0;
            const emptyGroups = [];
            for (const [gName, rows] of Object.entries(byGroup)) {
                let allEmpty = true;
                for (const row of rows) {
                    const col = row.methods && row.methods[m];
                    if (col && col.found) { allEmpty = false; break; }
                }
                if (allEmpty && rows.length) emptyGroups.push(gName);
            }
            byMethod[m] = { total, found, empty: total - found, empty_groups: emptyGroups.slice(0, 15) };
        }

        return {
            success: true,
            pdf_stem: compRes.pdf_stem,
            methods_available: compRes.methods_available || methods,
            by_method: byMethod,
            by_group: byGroup,
            comparison: comparison,
        };
    }

    // ========== Document List ==========
    async function loadDocuments() {
        const loading = document.getElementById('doc-list-loading');
        const grid = document.getElementById('doc-list-grid');
        const empty = document.getElementById('doc-list-empty');

        loading.style.display = 'flex';
        grid.style.display = 'none';
        empty.style.display = 'none';

        try {
            const res = await API.documents();
            loading.style.display = 'none';

            if (!res.success || !res.documents?.length) {
                empty.style.display = 'block';
                return;
            }

            grid.style.display = 'grid';
            grid.innerHTML = res.documents.map(doc => `
                <button
                    type="button"
                    class="doc-card group text-left p-5 rounded-xl glass border border-slate-700/50 hover:border-cyan-500/50 transition-all hover:shadow-lg hover:shadow-cyan-500/5"
                    data-doc-id="${escapeHtml(doc.pdf_stem)}"
                >
                    <div class="flex items-start justify-between gap-2">
                        <span class="font-semibold text-slate-100 group-hover:text-cyan-400 transition truncate">${escapeHtml(doc.pdf_stem)}</span>
                        <span class="text-white/0 group-hover:text-cyan-400 transition">→</span>
                    </div>
                    <div class="flex flex-wrap gap-1.5 mt-3">
                        ${doc.methods_available.map(m => `
                            <span class="method-badge ${METHOD_CLASS[m] || 'bg-slate-600/50 text-slate-400'}">${METHOD_LABELS[m] || m}</span>
                        `).join('')}
                    </div>
                    <p class="text-slate-500 text-xs mt-2">${doc.methods_available.length} method(s)</p>
                </button>
            `).join('');

            grid.querySelectorAll('.doc-card').forEach(btn => {
                btn.addEventListener('click', () => openDocument(btn.dataset.docId));
            });
        } catch (err) {
            loading.style.display = 'none';
            empty.style.display = 'block';
            empty.innerHTML = `<p class="text-rose-400">Error: ${escapeHtml(err.message)}</p>`;
        }
    }

    // ========== Document Detail ==========
    async function openDocument(docId) {
        currentDocId = docId;
        pdfDoc = null;
        pendingHighlightColumn = null;
        document.getElementById('doc-list-view').style.display = 'none';
        document.getElementById('doc-detail-view').style.display = 'block';

        document.getElementById('doc-title').textContent = docId;
        document.getElementById('doc-methods').textContent = 'Loading...';

        try {
            const res = await API.dashboard(docId);

            if (!res.success) {
                document.getElementById('doc-methods').textContent = 'No data';
                return;
            }

            dashboardData = res;
            const methods = res.methods_available || [];
            document.getElementById('doc-methods').textContent = methods.map(m => METHOD_LABELS[m] || m).join(' • ');

            renderDashboardCards(res);
            renderSidebar(res);
            showMainPlaceholder();
            initPdfViewer(docId);
        } catch (err) {
            document.getElementById('doc-methods').textContent = `Error: ${err.message}`;
        }
    }

    function backToList() {
        currentDocId = null;
        dashboardData = null;
        pdfDoc = null;
        document.getElementById('doc-list-view').style.display = 'block';
        document.getElementById('doc-detail-view').style.display = 'none';
        document.getElementById('pdf-viewer-panel').style.display = 'none';
    }

    // ========== PDF Viewer + Highlights ==========
    function initPdfViewer(docId) {
        const panel = document.getElementById('pdf-viewer-panel');
        const container = document.getElementById('pdf-viewer-container');
        const wrapper = document.getElementById('pdf-canvas-wrapper');
        const emptyEl = document.getElementById('pdf-viewer-empty');
        const unavailableEl = document.getElementById('pdf-viewer-unavailable');
        const canvas = document.getElementById('pdf-canvas');
        const layer = document.getElementById('pdf-highlight-layer');

        panel.style.display = 'block';
        container.style.display = 'flex';
        emptyEl.style.display = 'block';
        unavailableEl.style.display = 'none';
        wrapper.style.display = 'none';
        layer.innerHTML = '';

        if (typeof pdfjsLib === 'undefined') {
            emptyEl.style.display = 'none';
            unavailableEl.textContent = 'PDF.js not loaded';
            unavailableEl.style.display = 'block';
            return;
        }

        pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
        const loadingTask = pdfjsLib.getDocument({ url: API.pdf(docId) });
        loadingTask.promise.then(doc => {
            pdfDoc = doc;
            emptyEl.style.display = 'block';
            unavailableEl.style.display = 'none';
            if (pendingHighlightColumn) {
                const col = pendingHighlightColumn;
                pendingHighlightColumn = null;
                showPdfHighlights(col);
            }
        }).catch(() => {
            emptyEl.style.display = 'none';
            unavailableEl.textContent = 'PDF not found for this document';
            unavailableEl.style.display = 'block';
        });
    }

    async function showPdfHighlights(columnName) {
        if (!currentDocId) return;

        const wrapper = document.getElementById('pdf-canvas-wrapper');
        const canvas = document.getElementById('pdf-canvas');
        const layer = document.getElementById('pdf-highlight-layer');
        const pageLabel = document.getElementById('pdf-viewer-page');
        const emptyEl = document.getElementById('pdf-viewer-empty');
        const unavailableEl = document.getElementById('pdf-viewer-unavailable');

        if (!pdfDoc) {
            pendingHighlightColumn = columnName;
            emptyEl.style.display = 'none';
            unavailableEl.textContent = 'PDF still loading…';
            unavailableEl.style.display = 'block';
            wrapper.style.display = 'none';
            return;
        }

        try {
            const res = await API.highlights(currentDocId, columnName);
            if (!res.success || !res.available || !res.highlights?.length) {
                emptyEl.style.display = 'none';
                unavailableEl.textContent = res.reason || 'Highlight positions not available for this column';
                unavailableEl.style.display = 'block';
                wrapper.style.display = 'none';
                layer.innerHTML = '';
                return;
            }

            emptyEl.style.display = 'none';
            unavailableEl.style.display = 'none';
            wrapper.style.display = 'block';

            const firstPage = res.highlights[0].page;
            pageLabel.textContent = `Page ${firstPage}`;

            const page = await pdfDoc.getPage(firstPage);
            const viewport = page.getViewport({ scale: 1.5 });
            const ctx = canvas.getContext('2d');
            canvas.height = viewport.height;
            canvas.width = viewport.width;
            wrapper.style.width = viewport.width + 'px';
            wrapper.style.height = viewport.height + 'px';

            await page.render({
                canvasContext: ctx,
                viewport: viewport,
            }).promise;

            layer.style.width = viewport.width + 'px';
            layer.style.height = viewport.height + 'px';
            layer.innerHTML = '';

            const pageWidth = page.getViewport({ scale: 1 }).width;
            const pageHeight = page.getViewport({ scale: 1 }).height;

            res.highlights.filter(h => h.page === firstPage).forEach(h => {
                const box = h.box || {};
                const l = box.left || 0;
                const t = box.top || 0;
                const r = box.right || 1;
                const b = box.bottom || 1;
                const xMin = l * pageWidth;
                const xMax = r * pageWidth;
                const yMin = (1 - b) * pageHeight;
                const yMax = (1 - t) * pageHeight;

                const rect = viewport.convertToViewportRectangle([xMin, yMin, xMax, yMax]);
                const x = Math.min(rect[0], rect[2]);
                const y = Math.min(rect[1], rect[3]);
                const w = Math.abs(rect[2] - rect[0]);
                const height = Math.abs(rect[3] - rect[1]);

                const div = document.createElement('div');
                div.className = 'absolute bg-cyan-400/30 border-2 border-cyan-400/60 rounded';
                div.style.left = x + 'px';
                div.style.top = y + 'px';
                div.style.width = w + 'px';
                div.style.height = height + 'px';
                layer.appendChild(div);
            });
        } catch (err) {
            emptyEl.style.display = 'none';
            unavailableEl.textContent = 'Could not load highlights';
            unavailableEl.style.display = 'block';
            wrapper.style.display = 'none';
            layer.innerHTML = '';
        }
    }

    // ========== Top: 3 Method Cards ==========
    function renderDashboardCards(data) {
        const byMethod = data.by_method || {};
        const el = document.getElementById('dashboard-cards');
        el.style.display = 'grid';

        el.innerHTML = DASHBOARD_METHODS.filter(m => byMethod[m]).map(method => {
            const stats = byMethod[method];
            const emptyGroups = stats.empty_groups || [];
            const emptyList = emptyGroups.length > 0
                ? `<div class="mt-3 pt-3 border-t border-slate-700/50">
                    <div class="text-xs text-slate-500 uppercase tracking-wider mb-1">Empty groups</div>
                    <div class="flex flex-wrap gap-1">
                        ${emptyGroups.slice(0, 8).map(g => `<span class="text-xs px-2 py-0.5 rounded bg-slate-700/50 text-slate-400">${escapeHtml(g)}</span>`).join('')}
                        ${emptyGroups.length > 8 ? `<span class="text-xs text-slate-500">+${emptyGroups.length - 8}</span>` : ''}
                    </div>
                   </div>`
                : '';

            return `
                <div class="glass rounded-xl p-5 border border-slate-700/50">
                    <div class="flex items-center justify-between mb-4">
                        <span class="method-badge ${METHOD_CLASS[method] || ''}">${METHOD_LABELS[method] || method}</span>
                    </div>
                    <div class="grid grid-cols-3 gap-3">
                        <div>
                            <div class="text-xl font-bold text-cyan-400">${stats.total}</div>
                            <div class="text-xs text-slate-500 uppercase tracking-wider">Total</div>
                        </div>
                        <div>
                            <div class="text-xl font-bold text-emerald-400">${stats.found}</div>
                            <div class="text-xs text-slate-500 uppercase tracking-wider">Found</div>
                        </div>
                        <div>
                            <div class="text-xl font-bold text-amber-400">${stats.empty}</div>
                            <div class="text-xs text-slate-500 uppercase tracking-wider">Empty</div>
                        </div>
                    </div>
                    ${emptyList}
                </div>
            `;
        }).join('');
    }

    // ========== Sidebar: Groups + Columns ==========
    function renderSidebar(data) {
        const byGroup = data.by_group || {};
        const el = document.getElementById('sidebar-groups');
        document.getElementById('sidebar').style.display = 'block';
        document.getElementById('main-content').style.display = 'block';

        const groups = Object.entries(byGroup).sort((a, b) => a[0].localeCompare(b[0]));
        if (groups.length === 0) {
            el.innerHTML = '<p class="text-slate-500 text-sm">No groups</p>';
            return;
        }

        el.innerHTML = groups.map(([groupName, rows], idx) => {
            const collapseId = 'collapse-' + idx + '-' + groupName.replace(/[^a-zA-Z0-9]/g, '-').slice(0, 20);
            const columnsHtml = rows.map(row => `
                <button type="button" class="column-btn w-full text-left px-3 py-2 text-sm rounded-lg hover:bg-slate-700/50 text-slate-300 truncate transition"
                    data-column="${escapeHtml(row.column_name)}">
                    ${escapeHtml(row.column_name)}
                </button>
            `).join('');

            return `
                <div class="mb-2">
                    <button type="button" class="group-btn w-full flex items-center justify-between px-3 py-2 text-sm font-medium rounded-lg hover:bg-slate-700/50 text-slate-200 transition"
                        data-group="${escapeHtml(groupName)}">
                        <span class="truncate">${escapeHtml(groupName)}</span>
                        <span class="text-slate-500 text-xs ml-2 flex-shrink-0">${rows.length}</span>
                        <svg class="group-chevron w-4 h-4 flex-shrink-0 ml-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/>
                        </svg>
                    </button>
                    <div id="${collapseId}" class="sidebar-collapse hidden pl-2 mt-1 space-y-0.5 border-l-2 border-slate-700/50 ml-2">
                        ${columnsHtml}
                    </div>
                </div>
            `;
        }).join('');

        // Toggle collapse on group click
        el.querySelectorAll('.group-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const groupName = btn.dataset.group;
                const collapse = btn.nextElementSibling;
                if (collapse && collapse.classList.contains('sidebar-collapse')) {
                    collapse.classList.toggle('hidden');
                    const chevron = btn.querySelector('.group-chevron');
                    if (chevron) chevron.classList.toggle('rotate-180', !collapse.classList.contains('hidden'));
                }
                showGroupSnapshot(groupName);
            });
        });

        el.querySelectorAll('.column-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                showColumnDetail(btn.dataset.column);
            });
        });
    }

    // ========== Main Content ==========
    function showMainPlaceholder() {
        document.getElementById('main-placeholder').style.display = 'block';
        document.getElementById('main-group-snapshot').style.display = 'none';
        document.getElementById('main-column-detail').style.display = 'none';
    }

    function showGroupSnapshot(groupName) {
        if (!dashboardData) return;
        const byGroup = dashboardData.by_group || {};
        const rows = byGroup[groupName] || [];
        const methods = (dashboardData.methods_available || []).filter(m => DASHBOARD_METHODS.includes(m));
        if (methods.length === 0) methods.push(...DASHBOARD_METHODS);

        document.getElementById('main-placeholder').style.display = 'none';
        document.getElementById('main-column-detail').style.display = 'none';
        document.getElementById('main-group-snapshot').style.display = 'block';

        document.getElementById('group-snapshot-title').textContent = groupName;

        const tableEl = document.getElementById('group-snapshot-table');
        tableEl.innerHTML = `
            <table class="w-full text-sm">
                <thead>
                    <tr class="border-b border-slate-600">
                        <th class="text-left py-3 px-3 font-semibold text-slate-300 sticky left-0 bg-slate-900/95 min-w-[200px]">Column</th>
                        ${methods.map(m => `<th class="text-left py-3 px-3 font-semibold text-slate-300 min-w-[120px]">${METHOD_LABELS[m] || m}</th>`).join('')}
                    </tr>
                </thead>
                <tbody class="divide-y divide-slate-700/50">
                    ${rows.map(row => {
                        const cells = methods.map(m => {
                            const col = row.methods?.[m];
                            if (!col) return `<td class="py-2 px-3 text-slate-600">—</td>`;
                            const v = col.value || col.primary_value || '—';
                            const found = col.found;
                            const short = String(v).length > 35 ? String(v).slice(0, 32) + '…' : v;
                            return `
                                <td class="py-2 px-3">
                                    <button type="button" class="cell-btn text-left w-full rounded px-2 py-1.5 hover:bg-slate-800/80 transition"
                                        data-column="${escapeHtml(row.column_name)}" data-method="${escapeHtml(m)}">
                                        <span class="${found ? 'text-slate-200' : 'text-slate-500'}">${escapeHtml(short)}</span>
                                    </button>
                                </td>
                            `;
                        }).join('');
                        return `
                            <tr class="hover:bg-slate-800/30">
                                <td class="py-2 px-3 font-medium text-slate-300 sticky left-0 bg-slate-900/95 min-w-[200px]">${escapeHtml(row.column_name)}</td>
                                ${cells}
                            </tr>
                        `;
                    }).join('')}
                </tbody>
            </table>
        `;

        tableEl.querySelectorAll('.cell-btn').forEach(btn => {
            btn.addEventListener('click', () => showColumnDetail(btn.dataset.column));
        });
    }

    function showColumnDetail(columnName) {
        if (!dashboardData) return;
        const comparison = dashboardData.comparison || [];
        const row = comparison.find(r => r.column_name === columnName);
        if (!row) return;

        const methods = (dashboardData.methods_available || []).filter(m => DASHBOARD_METHODS.includes(m));
        if (methods.length === 0) methods.push(...DASHBOARD_METHODS);

        document.getElementById('main-placeholder').style.display = 'none';
        document.getElementById('main-group-snapshot').style.display = 'none';
        document.getElementById('main-column-detail').style.display = 'block';

        document.getElementById('column-detail-title').textContent = columnName;

        showPdfHighlights(columnName);

        const content = document.getElementById('column-detail-content');
        content.innerHTML = methods.map(method => {
            const col = row.methods?.[method];
            if (!col) return `<div class="p-4 rounded-lg bg-slate-800/50"><span class="text-slate-500">—</span></div>`;

            const attr = col.attribution || {};
            const candidates = col.candidates || [];
            const primary = col.value || col.primary_value || '—';
            const evidence = attr.evidence || (candidates[0] && candidates[0].evidence);
            const confidence = attr.confidence || (candidates[0] && candidates[0].confidence);
            const assumptions = attr.assumptions || (candidates[0] && candidates[0].assumptions);

            return `
                <div class="p-4 rounded-lg bg-slate-800/50 border border-slate-700/50">
                    <div class="flex items-center justify-between mb-2">
                        <span class="method-badge ${METHOD_CLASS[method] || ''}">${METHOD_LABELS[method] || method}</span>
                        ${confidence ? `<span class="text-xs ${confidenceClass(confidence)}">${confidence}</span>` : ''}
                    </div>
                    <div class="text-slate-200 font-medium mb-2">${escapeHtml(primary)}</div>
                    ${evidence ? `
                        <div class="text-sm text-slate-400 mb-2">
                            <span class="text-cyan-400/80 font-medium">Evidence:</span>
                            <div class="mt-1 text-slate-300 whitespace-pre-wrap">${escapeHtml(String(evidence).slice(0, 500))}${String(evidence).length > 500 ? '…' : ''}</div>
                        </div>
                    ` : ''}
                    ${assumptions ? `<div class="text-xs text-amber-400/80"><span class="font-medium">Assumptions:</span> ${escapeHtml(assumptions)}</div>` : ''}
                    ${col.page && col.page > 0 ? `<div class="text-xs text-slate-500 mt-2">Page ${col.page} • ${col.source_type || 'text'}</div>` : ''}
                    ${candidates.length > 1 ? `
                        <div class="mt-2 pt-2 border-t border-slate-700/50">
                            <div class="text-xs text-cyan-400/80 font-medium mb-1">Other candidates</div>
                            ${candidates.slice(1, 4).map(c => `<div class="text-xs text-slate-400">${escapeHtml(String(c.value || '').slice(0, 80))}</div>`).join('')}
                        </div>
                    ` : ''}
                </div>
            `;
        }).join('');
    }

    // ========== Attribution Modal ==========
    function showAttributionModal(columnName, method) {
        if (!dashboardData) return;
        const row = dashboardData.comparison && dashboardData.comparison.find(r => r.column_name === columnName);
        if (!row) return;
        const col = row.methods && row.methods[method];
        if (!col) return;

        document.getElementById('modal-column').textContent = columnName;
        document.getElementById('modal-method').textContent = METHOD_LABELS[method] || method;

        const body = document.getElementById('modal-body');
        const attr = col.attribution || {};
        const candidates = col.candidates || [];

        body.innerHTML = `
            <div>
                <div class="text-cyan-400 font-medium mb-1">Value</div>
                <div class="text-slate-200">${escapeHtml(col.value || col.primary_value || '—')}</div>
            </div>
            ${attr.evidence ? `<div><div class="text-cyan-400 font-medium mb-1">Evidence</div><div class="text-slate-300 whitespace-pre-wrap">${escapeHtml(attr.evidence)}</div></div>` : ''}
            ${attr.assumptions ? `<div><div class="text-amber-400 font-medium mb-1">Assumptions</div><div class="text-slate-300">${escapeHtml(attr.assumptions)}</div></div>` : ''}
            ${col.page ? `<div class="text-slate-500 text-xs">Page ${col.page} • ${col.source_type || ''}</div>` : ''}
            ${candidates.length > 1 ? `
                <div><div class="text-cyan-400 font-medium mb-2">Other candidates</div>
                ${candidates.slice(1).map(c => `<div class="p-2 rounded bg-slate-800/50 text-slate-300 text-sm">${escapeHtml(c.value || '')}</div>`).join('')}
                </div>
            ` : ''}
        `;

        document.getElementById('attribution-modal').style.display = 'flex';
    }

    function closeModal() {
        document.getElementById('attribution-modal').style.display = 'none';
    }

    // ========== PDF Viewer Expand ==========
    function initPdfViewerExpand() {
        const panel = document.getElementById('pdf-viewer-panel');
        const btn = document.getElementById('pdf-viewer-expand');
        const icon = document.getElementById('pdf-expand-icon');
        if (!panel || !btn) return;

        const expandSvg = '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4"/>';
        const collapseSvg = '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/>';

        btn.addEventListener('click', () => {
            const expanded = panel.classList.toggle('pdf-viewer-expanded');
            icon.innerHTML = expanded ? collapseSvg : expandSvg;
            btn.title = expanded ? 'Collapse PDF viewer' : 'Expand PDF viewer';
        });
    }

    // ========== Init ==========
    document.addEventListener('DOMContentLoaded', () => {
        loadDocuments();
        document.getElementById('back-to-list').addEventListener('click', backToList);
        document.getElementById('modal-close').addEventListener('click', closeModal);
        document.getElementById('attribution-modal').addEventListener('click', (e) => {
            if (e.target === e.currentTarget) closeModal();
        });
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') closeModal();
        });
        initPdfViewerExpand();
    });

    const params = new URLSearchParams(window.location.search);
    const docParam = params.get('doc');
    if (docParam) {
        document.addEventListener('DOMContentLoaded', () => openDocument(docParam));
    }
})();
