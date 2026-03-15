# BHMS UI Guidelines

This frontend now follows a small set of reusable page patterns so new pages stay visually consistent.

## Core page structure

1. Use `PageHero` for the first screen of every page.
   - `kicker` for the English eyebrow.
   - `title` for the main Chinese headline.
   - `description` for the narrative sentence that explains the page goal.
   - `pills` for 2-3 high-signal metrics only.
2. Use `InsightCard` as the hero-side focus card.
   - Best for one key number, status, or model output.
3. Use `PanelCard` for all secondary content blocks.
   - Charts, forms, tables, timelines, and evidence sections should all live in `PanelCard`.

## Reusable UI components

- `src/components/ui/PageHero.tsx`
  - Top-level page hero shell.
- `src/components/ui/InsightCard.tsx`
  - Dark emphasis card for page focus or summary.
- `src/components/ui/MetricCard.tsx`
  - Dashboard metric card with icon, value, and caption.
- `src/components/ui/PanelCard.tsx`
  - Standard content card wrapper based on Ant Design `Card`.
- `src/components/ui/StatusTag.tsx`
  - Unified status tag tones: `good`, `warning`, `critical`, `neutral`, `info`.

## Tone rules

- Primary action and positive trend: teal / green
- Warning and pending attention: amber
- Critical faults and danger: orange-red
- Background and neutral surfaces: warm off-white + slate text

## Layout rules

- Keep page content inside `.page-shell`.
- Prefer 18px gaps between major sections to match the current rhythm.
- On dense pages, show summary first, workflow second, details third.

## Content rules

- Avoid generic card titles like "信息" or "列表".
- Prefer action-oriented titles such as "预测配置", "最新异常事件", "诊断依据".
- The hero description should explain why the page matters, not restate the title.
