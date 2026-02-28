# Attribution Page UX Improvements — Discussion

Ideas for improving the Attribution Viewer UX, aligned with the extraction page and future verification features.

---

## Current State

- **Layout**: Left sidebar (doc select + column list) | Right (PDF + chunk nav + column detail)
- **Flow**: Select doc → Select column → See PDF with highlights + snippets
- **Components**: Custom Tailwind, no DaisyUI
- **Issues**: Dense column list, unclear hierarchy, no search, PDF can feel disconnected from column context

---

## 1. Visual Consistency with Extract Page

**Use DaisyUI** on the attribution page so it matches the extract page:

- `card`, `btn`, `badge`, `select`, `drawer`, `collapse`
- Same `data-theme="dark"` and color system
- Shared navbar (Home | Extract | Attribution | Report)

---

## 2. Layout Improvements

### Option A: Drawer Sidebar (Mobile-Friendly)

- Use DaisyUI **Drawer** for the left panel
- Collapsible on small screens
- Keeps doc select + column list in a slide-out panel

### Option B: Tabs or Split View

- **Tab 1**: Document + column list
- **Tab 2**: PDF viewer + column detail
- Or a persistent split with a resizable divider

### Option C: Card-Based Sections

- **Card 1**: Document selection + Refresh
- **Card 2**: Column list (searchable, grouped)
- **Card 3**: PDF viewer
- **Card 4**: Column detail + snippets

---

## 3. Column List UX

**Current**: Flat list of column names, scrollable.

**Improvements**:

- **Search/filter**: Text input to filter columns
- **Group by**: Collapse/accordion by group (e.g. "Add-on Treatment", "Control Arm")
- **Badges**: Show chunk count per column (e.g. `3 chunks`)
- **Status**: Indicate verified vs unverified (for future verification)
- **Keyboard**: Arrow keys to move between columns

---

## 4. PDF + Highlights UX

**Current**: PDF pages stacked vertically, highlights as overlays, chunk nav below.

**Improvements**:

- **Side-by-side**: Column detail panel next to PDF (or below on mobile)
- **Highlight states**: Clear active vs inactive (e.g. opacity, border)
- **Page tabs**: Quick jump to pages with highlights
- **Zoom**: Optional zoom controls
- **Full-screen**: Optional full-screen PDF mode

---

## 5. Column Detail Panel

**Current**: Column name, value, method values, attributed snippets.

**Improvements**:

- **Card layout**: Use DaisyUI card for structure
- **Method comparison**: Table or list for method values
- **Snippet cards**: Each snippet in a small card with score
- **Copy**: Copy value or snippet to clipboard
- **Future**: "Verify" button per snippet/cell

---

## 6. Empty States & Loading

- **No doc**: Clear message and link to Extract
- **No columns**: "Run attribution" or "No attributed columns"
- **Loading PDF**: Skeleton or spinner
- **Loading highlights**: Progress indicator

---

## 7. Navigation & Flow

- **Breadcrumbs**: Home > Attribution > [Doc] > [Column]
- **Back**: Easy way to return to column list from detail
- **Deep links**: `?doc=X&column=Y` for direct linking
- **Recent**: Optional "Recently viewed" columns

---

## 8. Future: Verification Integration

When verification is added:

- **Badge**: `verified` / `pending` per column or cell
- **Verify button**: In column detail or per snippet
- **Modal**: Verify flow (chunk text, confirm/edit value)
- **Bulk actions**: "Verify all in group", "Re-run attribution"

---

## 9. Recommended Priority

| Priority | Change | Effort |
|---------|--------|--------|
| 1 | Add DaisyUI for consistency | Low |
| 2 | Search/filter in column list | Low |
| 3 | Group columns by group name (collapse) | Medium |
| 4 | Drawer for mobile | Low |
| 5 | Card layout for column detail | Low |
| 6 | Deep link `?doc=&column=` | Low |
| 7 | Page jump for highlights | Medium |

---

## 10. Wireframe (Improved Layout)

```
┌─────────────────────────────────────────────────────────────────┐
│  Navbar: Home | Extract | Attribution | Report                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─ Drawer (collapsible) ─┐  ┌─ Main ─────────────────────────┐ │
│  │  Document: [Select ▼]   │  │  PDF viewer                    │ │
│  │  [Refresh Attribution] │  │  [pages with highlights]        │ │
│  │  ───────────────────    │  │                                │ │
│  │  [Search columns...]     │  │  [← Prev] 1/3 [Next →]         │ │
│  │  ▼ Add-on Treatment     │  ├────────────────────────────────┤ │
│  │    • Add-on Treatment   │  │  Column: Add-on Treatment       │ │
│  │    • Control Arm        │  │  Value: darolutamide + ADT     │ │
│  │  ▼ Control Arm          │  │  Snippets: [card] [card]       │ │
│  │    • Control Arm - N    │  │  [Verify] (future)             │ │
│  └─────────────────────────┘  └────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```
