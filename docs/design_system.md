# Palatine Design System

> Mission-control-grade HMI for test engineering: propellant systems, turbine
> control, and DAQ hardware discovery. The interface behaves like an
> instrument — legible at a glance, unambiguous under stress, quiet until
> something changes.

---

## 1. Ethos

1. **Data is the interface.** Chrome recedes to near-black; process state is
   the brightest thing on screen.
2. **Machine text looks like machine text.** Signal names, values, tags, IPs
   and units are set in monospace, verbatim from config (`snake_case`,
   never re-cased). Sans-serif is reserved for humans.
3. **Color is a signal, never decoration.** If a pixel carries hue, it names
   a physical service (fuel / oxidizer / purge) or a live state
   (open / running / nominal). UI chrome is strictly grayscale.

---

## 2. Visual Principles

- **Dark-only.** One theme, no light mode. Canvas is near-black so green
  state rings and service-colored lines read instantly.
- **Hairlines over shadows.** Regions are divided by 1px borders. No drop
  shadows, no glass, no blur.
- **Square corners.** 0–2px radius everywhere. The only capsules and circles
  are domain objects (tanks, valve symbols) — never chrome.
- **Density is a feature.** 4px base grid, 10–11px micro-type, tabular
  figures. Fit the whole test cell on one screen before paging.
- **Compressed footprint.** All horizontal padding and tracking run **−7%**
  of their nominal value (see §4.4), tightening every chip, card, and nav
  element without touching vertical rhythm.
- **Restraint, then urgency.** Grayscale at rest. Semantic color appears only
  where physics appears. Gradients and glow exist in exactly one place:
  blueprint-style article illustrations (turbine renders), which are treated
  as image assets, not UI.

---

## 3. Color

### 3.1 Surfaces & chrome (grayscale only)

| Token           | Hex       | Use                                              |
| --------------- | --------- | ------------------------------------------------ |
| `--void`        | `#050505` | Page backdrop outside the app frame              |
| `--bg`          | `#0A0A0A` | App canvas / schematic board                     |
| `--surface-1`   | `#111113` | Sidebar, modals, chart & readout panels          |
| `--surface-2`   | `#17171A` | Cards, inputs, selected sidebar rows, icon buttons |
| `--surface-3`   | `#232327` | Hover, active segmented chip, bus tags           |
| `--line`        | `#1F1F22` | Hairline dividers, panel borders, plot grid      |
| `--line-strong` | `#303034` | Control borders, checkbox strokes, closed valves |

| Token      | Hex       | Use                                   |
| ---------- | --------- | ------------------------------------- |
| `--text-1` | `#EDEDED` | Primary text, live values             |
| `--text-2` | `#9C9CA1` | Labels, secondary text, nav inactive  |
| `--text-3` | `#626266` | Disabled, units, device tags, axes    |
| `--text-4` | `#3C3C40` | Faint ticks, placeholder glyphs       |

### 3.2 Process & state color (the only permitted hues)

| Token            | Hex       | Meaning                                            |
| ---------------- | --------- | -------------------------------------------------- |
| `--state-live`   | `#00D26A` | Valve OPEN ring, running transport, nominal series |
| `--svc-fuel`     | `#0047FF` | Fuel service lines & tank fill (saturated blue)    |
| `--svc-ox`       | `#D6001F` | Oxidizer service lines & tank fill; alarm red      |
| `--svc-purge`    | `#00D26A` | N₂ / pneumatic service lines (shares live green)   |
| `--spark`        | `#58A6FF` | Sparkline traces only                              |
| `--series-violet`| `#7C4DFF` | Chart series 3                                     |
| `--warn`         | `#FFB020` | Warning state (reserved)                           |
| `--heat`         | `#FF7A18` | Combustion glow — illustration assets only         |

Chart series order: green → crimson → violet → blue → amber.

### 3.3 Color rules

- Chrome never takes hue. No colored buttons, links, or backgrounds.
- **Line color = service. Ring color = state.** A valve on any service shows
  a `--state-live` ring when OPEN and a `--line-strong` ring when CLOSED.
- Every colored state carries a redundant non-color cue: state word
  (`ENABLED/DISABLED`), bar orientation, or glyph change (color-vision safe).
- Tank fills use the service color at 100% saturation — the fill *is* the
  fluid. Level is geometry, not opacity.

---

## 4. Typography

**Two voices, never blended.**

- **Sans (human):** `Geologica, "Helvetica Neue", Arial, sans-serif` — nav
  links, sidebar items, dialog titles, display headings, prose. Geologica's
  built-in width axis is pinned to its **condensed end (−7%)** for all UI
  text, giving the system its signature tight footprint.
- **Mono (machine):** `"JetBrains Mono", "IBM Plex Mono", ui-monospace,
  SFMono-Regular, Menlo, monospace` — every signal name, value, unit, tag,
  address, flag, toggle word, axis tick, timestamp.

### 4.1 Casing semantics (meaning-bearing)

| Case                    | Face  | Means                          | Examples                            |
| ----------------------- | ----- | ------------------------------ | ----------------------------------- |
| `snake_case`            | mono  | Live identifier from config    | `compressor_inlet_pressure`, `hot_fire` |
| `UPPER CASE` + tracking | mono  | Physical / bus / state words   | `ATMOSPHERE`, `FUEL SUPPLY`, `ENABLED`, `MAX POWER HOLD` |
| Title Case              | sans  | Human-facing headings & lists  | "Hardware Discovery", "Propellant Logistics" |

Identifiers are rendered **verbatim** — they must match code and config
exactly. Never prettify, re-case, or wrap them.

### 4.2 Type scale

| Role          | Size/Line | Face | Weight | Tracking   | Notes                          |
| ------------- | --------- | ---- | ------ | ---------- | ------------------------------ |
| Display       | 36/40     | sans | 400    | −0.02em    | Dialog headlines ("Discovered 5 Devices") |
| Heading       | 14/20     | sans | 500    | 0          | Panel & modal titles           |
| UI            | 13/18     | sans | 400    | 0          | Nav, sidebar, body             |
| UI small      | 12/16     | sans | 400    | 0          | Dense lists, chips             |
| Micro label   | 10/14     | mono | 400    | +0.08em    | Uppercase labels, flags, tags  |
| Value SM      | 13/16     | mono | 500    | 0          | Inline readouts                |
| Value MD      | 16/20     | mono | 500    | 0          | Card readouts                  |
| Value LG      | 20/24     | mono | 500    | 0          | Hero readouts (`16,850`)       |
| Unit          | 9/12      | mono | 400    | +0.04em    | Suffix, `--text-3`, space-separated |

> All sans roles above render through Geologica at `font-stretch` / width
> axis **93%** (i.e. −7% narrower than default). Vertical metrics and line
> heights are unchanged — only horizontal density compresses.

### 4.3 Numerals

- Tabular figures always (mono guarantees this); digits must not jitter on
  live update.
- Fixed precision per channel — never fluctuating decimal counts.
- Thousands separators for human-scale magnitudes (`16,850`, `1,620.5`).
- Units are a separate typographic token: smaller, gray, never inside the
  value string (`339.9 PSI`, `12.4 kg/s`).

### 4.4 The −7% compression rule

Every horizontal spacing value in the system is multiplied by **0.93**:

| Element                       | Nominal | Applied (×0.93) |
| ----------------------------- | ------- | --------------- |
| Card horizontal padding       | 10px    | 9px             |
| Chip / segment padding-x      | 8px     | 7px             |
| Button padding-x              | 8px     | 7px             |
| Nav link gap                  | 16px    | 15px            |
| Modal horizontal padding      | 32px    | 30px            |
| Sans letter-spacing (UI)      | 0       | −0.07em equiv. via width axis |

Vertical padding, line-height