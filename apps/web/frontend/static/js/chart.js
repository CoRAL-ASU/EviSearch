// One chart form, used by the Learning page: a cumulative line over time, one series per chart (two measures of very
// different size are two charts sharing a time axis, never two y-axes). Colours are the validated categorical slots 1
// and 2 on the dark surface the chart card sets.
window.EVSChart = (function () {
    const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
    const W = 560, H = 150, PAD = {l: 34, r: 46, t: 12, b: 22};

    function line(el, series, key, colour, unit) {
        if (!el) return;
        const points = (series || []).filter((p) => p && p.at);
        if (points.length < 2) {
            el.innerHTML = `<p class="text-xs" style="color:rgba(255,255,255,.55)">Not enough history yet — ${points.length ? points[0][key] + ' so far' : 'nothing recorded'}.</p>`;
            return;
        }
        const xs = points.map((p) => Date.parse(p.at));
        const ys = points.map((p) => p[key]);
        const x0 = Math.min(...xs), x1 = Math.max(...xs), yMax = Math.max(...ys, 1);
        const px = (t) => PAD.l + (W - PAD.l - PAD.r) * (x1 === x0 ? 1 : (t - x0) / (x1 - x0));
        const py = (v) => H - PAD.b - (H - PAD.t - PAD.b) * (v / yMax);
        const path = points.map((p, i) => `${i ? 'L' : 'M'}${px(xs[i]).toFixed(1)},${py(ys[i]).toFixed(1)}`).join(' ');
        const day = (t) => new Date(t).toLocaleDateString([], {month: 'short', day: 'numeric'});
        const ticks = [0, yMax];
        el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(unit)} over time, cumulative; ends at ${ys[ys.length - 1]}">
            ${ticks.map((v) => `<line class="grid" x1="${PAD.l}" x2="${W - PAD.r}" y1="${py(v).toFixed(1)}" y2="${py(v).toFixed(1)}"/>
                <text class="axis-label" x="${PAD.l - 6}" y="${(py(v) + 3).toFixed(1)}" text-anchor="end">${v}</text>`).join('')}
            <text class="axis-label" x="${PAD.l}" y="${H - 6}">${esc(day(x0))}</text>
            <text class="axis-label" x="${W - PAD.r}" y="${H - 6}" text-anchor="end">${esc(day(x1))}</text>
            <path class="line" d="${path}" stroke="${colour}"/>
            <circle class="dot" cx="${px(xs[xs.length - 1]).toFixed(1)}" cy="${py(ys[ys.length - 1]).toFixed(1)}" fill="${colour}"/>
            <text class="end-label" x="${(W - PAD.r + 8).toFixed(1)}" y="${(py(ys[ys.length - 1]) + 4).toFixed(1)}">${ys[ys.length - 1]} ${esc(unit)}</text>
            <line class="cross" x1="0" x2="0" y1="${PAD.t}" y2="${H - PAD.b}" stroke="${colour}" stroke-width="1" opacity="0"/>
            <rect x="${PAD.l}" y="${PAD.t}" width="${W - PAD.l - PAD.r}" height="${H - PAD.t - PAD.b}" fill="transparent" class="hit"/>
        </svg><div class="tip" style="display:none"></div>`;
        const svg = el.querySelector('svg'), tip = el.querySelector('.tip'), cross = el.querySelector('.cross');
        const hit = el.querySelector('.hit');
        const move = (e) => {
            const box = svg.getBoundingClientRect();
            const sx = (e.clientX - box.left) / box.width * W;
            let best = 0;
            points.forEach((p, i) => { if (Math.abs(px(xs[i]) - sx) < Math.abs(px(xs[best]) - sx)) best = i; });
            cross.setAttribute('x1', px(xs[best])); cross.setAttribute('x2', px(xs[best])); cross.setAttribute('opacity', '.5');
            tip.style.display = 'block';
            tip.style.left = (px(xs[best]) / W * box.width) + 'px';
            tip.style.top = (py(ys[best]) / H * box.height) + 'px';
            tip.textContent = `${new Date(xs[best]).toLocaleString([], {month: 'short', day: 'numeric', hour: '2-digit'})} · ${ys[best]} ${unit}`;
        };
        hit.addEventListener('mousemove', move);
        hit.addEventListener('mouseleave', () => { tip.style.display = 'none'; cross.setAttribute('opacity', '0'); });
    }

    return {line};
})();
