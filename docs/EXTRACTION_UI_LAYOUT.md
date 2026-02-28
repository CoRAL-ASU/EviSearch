# Extraction & Attribution UI Layout (DaisyUI)

Rough UI layout for the agentic extractor page and attribution viewer, using DaisyUI components. Designed for streaming column-value pairs, future verification/revise, and consistent styling.

---

## Component Mapping

| Use Case | DaisyUI Component | Notes |
|----------|-------------------|-------|
| Overall progress | **Progress** | `progress progress-primary` — filled/total columns |
| Turn/status stats | **Stat** | `stats` — Turn N, Columns filled, Groups in batch |
| Streaming table | **Table** | `table table-zebra` — Column \| Value pairs, append rows as SSE arrives |
| Status badges | **Badge** | `badge badge-success` (running), `badge badge-info` (done), `badge badge-warning` (verify) |
| Cards/sections | **Card** | `card bg-base-200 shadow-xl` — group sections |
| Collapsible groups | **Collapse** | `collapse` — expand/collapse by group for verification |
| Alerts | **Alert** | `alert alert-info` — "Extracting…", `alert alert-success` — "Done" |
| Loading state | **Loading** | `loading loading-spinner` — during extraction |
| Steps | **Steps** | `steps` — Upload → Extract → Attributions → View |
| Timeline | **Timeline** | Optional: turn-by-turn log of groups extracted |
| Modal | **Modal** | Future: verify cell, revise answer |
| Toast | **Toast** | Success/error notifications |

---

## Page 1: Extraction (Home / Extract)

### Layout Wireframe

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Navbar: CoRal | Home | Attribution | Report                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Steps: [1.Upload] ──► [2.Columns] ──► [3.Extract] ──► [4.Done]             │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─ Card: Upload PDF ─────────────────────────────────────────────────┐    │
│  │  [File input]  or  drag & drop                                       │    │
│  │  ✓ document.pdf                                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  ┌─ Card: Column Groups ──────────────────────────────────────────────┐    │
│  │  [Select all] [ ] Add-on Treatment  [ ] Control Arm  [ ] Primary EP  │    │
│  │  ... (from definitions, checkboxes)                                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  ┌─ Card: Run Extraction ──────────────────────────────────────────────┐    │
│  │  [▶ Run Agentic Extractor]                                           │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  ┌─ Card: Live Extraction (visible when running) ──────────────────────┐    │
│  │  Stats:  Turn 3  |  42/133 columns  |  Add-on Treatment, Control Arm │    │
│  │  [████████████░░░░░░░░] 32%                                          │    │
│  │                                                                      │    │
│  │  Streaming table (column | value) — newest at top:                    │    │
│  │  ┌─────────────────────────────────────┬──────────────────────────┐  │    │
│  │  │ Column                              │ Value                    │  │    │
│  │  ├─────────────────────────────────────┼──────────────────────────┤  │    │
│  │  │ Control Arm - N                     │ 654                      │  │    │
│  │  │ Add-on Treatment                    │ darolutamide + ADT + doc  │  │    │
│  │  │ Primary Endpoint(s)                 │ Overall survival          │  │    │
│  │  │ ...                                 │ ...                      │  │    │
│  │  └─────────────────────────────────────┴──────────────────────────┘  │    │
│  │                                                                      │    │
│  │  [View Attribution →]  (enabled when done)                            │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Component Structure (Extraction Page)

```html
<!-- Stats row (during extraction) -->
<div class="stats stats-vertical lg:stats-horizontal shadow bg-base-200">
  <div class="stat">
    <div class="stat-title">Turn</div>
    <div class="stat-value text-primary">3</div>
  </div>
  <div class="stat">
    <div class="stat-title">Columns filled</div>
    <div class="stat-value">42 <span class="text-base font-normal">/ 133</span></div>
  </div>
  <div class="stat">
    <div class="stat-title">Current groups</div>
    <div class="stat-desc">Add-on Treatment, Control Arm</div>
  </div>
</div>

<!-- Progress bar -->
<progress class="progress progress-primary w-full" value="42" max="133"></progress>

<!-- Streaming table -->
<div class="overflow-x-auto">
  <table class="table table-zebra">
    <thead>
      <tr><th>Column</th><th>Value</th><th>Group</th></tr>
    </thead>
    <tbody id="streaming-rows">
      <!-- Rows appended via SSE -->
    </tbody>
  </table>
</div>
```

### Future: Verification / Revise

- **Collapse** per group: expand to see rows, each row has `[Verify]` button
- **Modal**: "Verify cell" — show chunk text, value; user confirms or edits
- **Badge** `badge-warning` on cells pending verification

---

## Page 2: Attribution Viewer (Rewrite with DaisyUI)

### Layout Wireframe

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Navbar: CoRal | Home | Report                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─ Left sidebar (Card) ─────┐  ┌─ Right: PDF + details ──────────────────┐ │
│  │  Document: [Select ▼]      │  │                                        │ │
│  │  [Refresh Attribution]     │  │  ┌─ PDF viewer ──────────────────────┐ │ │
│  │  ─────────────────────    │  │  │                                    │ │ │
│  │  Columns (with chunks):    │  │  │  [PDF pages]                      │ │ │
│  │  • Add-on Treatment        │  │  │                                    │ │ │
│  │  • Control Arm             │  │  └────────────────────────────────────┘ │ │
│  │  • Primary Endpoint(s)     │  │  [← Prev] 1/3 [Next →]                 │ │
│  │  • ...                     │  │  ─────────────────────────────────────  │ │
│  │                            │  │  Column: Add-on Treatment               │ │
│  │                            │  │  Value: darolutamide + ADT + docetaxel  │ │
│  │                            │  │  Snippets: [...]                        │ │
│  └────────────────────────────┘  └────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Component Structure (Attribution Page)

```html
<!-- Drawer layout: sidebar + main -->
<div class="drawer lg:drawer-open">
  <input id="drawer" type="checkbox" class="drawer-toggle" />
  <div class="drawer-content">
    <!-- PDF area -->
    <div class="card bg-base-200 shadow-xl">
      <div class="card-body">
        <div id="pdf-pages" class="min-h-[60vh]"></div>
        <div class="join mt-2">
          <button class="btn btn-sm join-item">← Prev</button>
          <span class="btn btn-sm btn-ghost join-item">1 / 3</span>
          <button class="btn btn-sm join-item">Next →</button>
        </div>
      </div>
    </div>
  </div>
  <div class="drawer-side">
    <label for="drawer" class="drawer-overlay"></label>
    <div class="menu p-4 w-80 bg-base-200 min-h-full">
      <select class="select select-bordered w-full">...</select>
      <button class="btn btn-primary btn-sm mt-2">Refresh Attribution</button>
      <ul class="menu mt-4" id="column-list">
        <!-- Column buttons -->
      </ul>
    </div>
  </div>
</div>

<!-- Column detail card -->
<div class="card bg-base-200 shadow-xl mt-4">
  <div class="card-body">
    <h3 class="card-title text-primary" id="sel-column-name"></h3>
    <p class="text-base-content/70" id="sel-column-value"></p>
    <div class="divider"></div>
    <div id="attributed-snippets" class="space-y-2"></div>
  </div>
</div>
```

---

## Flow Summary

1. **Home/Extract**: Upload PDF → Select columns → Run extraction → See streaming table → [View Attribution]
2. **Attribution**: Select doc → Select column → See PDF + highlights + snippets
3. **Future**: Verify cells, revise answers, re-run attribution

---

## DaisyUI Setup

- Add DaisyUI to Tailwind: `tailwind.config.js` → `plugins: [require('daisyui')]`
- Or use CDN: `https://cdn.jsdelivr.net/npm/daisyui@4.x/dist/full.min.css` (after Tailwind)
- Theme: `data-theme="dark"` or `class="dark"` for dark mode

---

## SSE Events → UI Updates

| Event | UI Action |
|-------|-----------|
| `turn_start` | Update stats (turn), progress bar |
| `columns_written` | Append rows to table (prepend for newest-first), update filled count |
| `done` | Show alert success, enable "View Attribution" button |
| `error` | Show alert error |
