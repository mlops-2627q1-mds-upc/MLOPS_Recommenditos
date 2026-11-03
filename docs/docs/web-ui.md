The web page
============

A browser that opens the service's address, `http://<host>:8080/` locally and on the VM, gets a web page for the price API ([#269](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/269), [EDN-96](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md)).
It is a form for the car's details in a raised cell on the left and the answer on the right: the estimated asking price, the price range with the similar listings marked on it, a waterfall of what moved the estimate, and the similar listings as a table with how far each lies from the estimate.
It is made for a laptop or desktop screen and holds its layout down to a window about 1,000 px wide; a phone layout is out of scope.

## How it is built and served

| File | What it is |
|------|------------|
| `recommenditos/api/web/index.html` | The page; `{{assets}}` stands for the versioned asset path |
| `recommenditos/api/web/app.css` | Hand-written styles, light and dark |
| `recommenditos/api/web/app.js` | Plain JavaScript, one ES module, no framework and no build step |
| `recommenditos/api/web/theme.js` | Applies a theme chosen with the toggle before the page paints |
| `recommenditos/api/web/fonts/` | BDO Grotesk (all text) and Recommenditos Mono, a renamed subset of Basically A Mono (uppercase labels, table heads, ticks and IDs), built by `tools/web_fonts.py`, each family with its SIL Open Font License file |
| `recommenditos/api/webui.py` | Reads the files once at startup and serves them from memory |
| `recommenditos/api/routes/index.py`, `routes/assets.py` | `GET /` and `GET /assets/<version>/<file>` |

The files are part of the `recommenditos` package, so they ship in the API image and reach the VM with every deploy, and a rollback puts the matching page back with the API.
nginx needs no change: its `location /` already passes every path but `/feedback` and `/metrics` to the API ([Deployment](deployment.md#the-stack)).
A change to `deploy/nginx.conf` only reaches the VM by running the bootstrap again, which is why the page does not depend on one.

`GET /` answers with the page when the request's `Accept` header names `text/html` with at least the weight of `application/json`, as a browser's does.
Every other client, curl, an HTTP library or an uptime monitor, gets the JSON index as before, and both answers say `Vary: Accept`.
The assets live under `/assets/<version>/`, where the version is a hash of all files of the page, so they are sent with `Cache-Control: public, max-age=31536000, immutable`; the page itself is sent with `no-cache` and an `ETag`, so a deploy shows at the next load.
Text files are gzip-compressed once at startup.

What a first visit loads, measured through nginx on the local stack on 2026-11-03:

| File | Size | Sent (gzip) |
|------|------|-------------|
| The page | 12.4 KB | 3.0 KB |
| `app.css` | 23.3 KB | 6.1 KB |
| `app.js` | 37.4 KB | 12.0 KB |
| `theme.js`, `favicon.svg` | 0.7 KB | 0.4 KB |
| BDO Grotesk, Recommenditos Mono 400 and 500 (woff2) | 54.8 KB | 54.8 KB |
| In all | 128.5 KB | 76.3 KB |

The page also reads `/openapi.json` (52 KB, uncompressed) and `/health` on every load; the assets are cached for a year after the first visit.
Held in memory with their gzip copies, the files take about 150 KB per worker process.
Measured with the first design, which sent 98 KB: after 300 loads of the page with all its assets, 2,400 requests in 3.6 s, nginx used 7.5 MiB against 7.3 MiB idle, of its 64 MiB limit, and the API 251.3 MiB against 250.0 MiB.

## The design

The page takes its visual language from [momentic.ai](https://momentic.ai/), which Lukas chose as the reference: a ruled ledger on warm paper.
Its colours, type, rules and component shapes were measured on that site in the browser; no logo, text, image, code or font file comes from it.
The page splits into two numbered steps: on the left a raised cell, "Describe the car", holds the form; on the right "Estimate" holds the answer, which before the first estimate points back at the form.

- **Type:** BDO Grotesk, the open-licensed grotesque momentic uses, for all text, at weights 400 and 500 with -0.04em tracking from 24 px up.
  Recommenditos Mono, a subset of Basically A Mono, only for uppercase labels, the step chips, field labels, table heads, axis ticks and IDs, never for sentences.
  Sizes: 96 px for the price, 40 px for the step titles (32 px below 1,200 px), 24 px for the form's groups and the result's sections, 16 px for body text, 15 px in fields, 14 px for tables and captions, 13 px for labels and hints, 12 px for ticks and IDs.
  Figures are tabular wherever numbers line up; the price keeps the face's proportional figures.
- **Why these faces:** the first build's serif read as a generic AI look; Schibsted Grotesk, which replaced it, was dropped when Lukas chose momentic's style.
  Both of momentic's faces are open-licensed, so the genuine ones give the closest match without a substitute: BDO Grotesk has no Reserved Font Name; Basically A Mono reserves its name, so the subset is renamed.
- **Colour:** cream ground `#f5f3f2`, ink `#1c1c1c`, greys `#404040` and `#6b6b6b`, olive `#7f806f` for every rule; lime `#87f700` is the one accent, on the Estimate button, the estimate bar and the ring of the range marker.
  Lime is too pale on cream to carry a mark alone, so the estimate bar has an ink edge and the marker an ink core.
  The form cell is `#fbfbfb`, fields an olive tint with a 1 px olive border; the dark theme is ink `#1c1c1c` with a `#252525` form cell.
  Where momentic's own values fall short of WCAG AA, the page deviates: muted grey `#6b6b6b` instead of `#737373`, a visible border on fields, a blue focus ring (`#2563eb`, `#93c5fd` in dark) instead of a pale one, and a darker error red (`#a8433b`, `#ef7a6f` in dark); olive text only from 24 px.
  A test computes every text pair at 4.5:1 and borders, marks and focus at 3:1 from the tokens in `app.css`, in both themes.
- **Structure:** every division is a 1 px olive rule that runs to the cell's edge: under the header, between the form cell and the result, above each result section; hairlines inside; no cards, shadows or gradients; 4 px radius on chips and panels, 8 px on fields and buttons.
  The similar listings sit in a tinted panel with an index column, as in momentic's leaderboard.
- **The button row:** it stays at the bottom of the window while the form runs past it and comes to rest under the last field, so the buttons are in reach at any window height.
  The form's error summary sits under the row, not in it, so the row stays 75 px, inside the 112 px of `scroll-padding-bottom`, and a field the focus reaches always scrolls clear of it; a test checks both, and after an empty submit at 1,440 and 1,024 by 700 px no field the Tab key reached was covered.
- **Layout:** at 1,440 px the form is 416 px wide in a 480 px cell and the result 896 px; below 1,200 px the form is 312 px and the result 584 px, and the listings drop their index column below 1,100 px and fold registration and power into their second line below 680 px.
- **Captions:** they run the width of their chart or table, with `text-wrap: pretty` against a single word on the last line.
- **Charts:** drawn as SVG at the column's measured pixel width, so their text keeps its size at every width.
  In the waterfall a solid ink bar raises the price and an outlined bar lowers it, so the sign does not depend on colour.
- **Motion:** colour fades of 150 to 180 ms, a 0.97 press on the buttons, the range marker sliding to the new estimate in 500 ms, the waterfall's bars growing once in 800 ms, and a blinking block after the button's label while an estimate runs; all off under `prefers-reduced-motion`.
- **Similar listings:** "vs. estimate" says how far each listing's asking price lies from the estimate, in whole percent.
  No deal tiers such as "good deal": the listings are other cars, and the API gives no boundaries for tiers.
  Each listing is titled by its make and model, the data's clean fields, with fuel, body, seller and country below.
  Under them, in quotes, is `model_version`: the seller's own description of the version, free text in the seller's language that mixes trim, equipment and even mileage ("d/Virtual-Cockpit/62.000KM"), shown as written and not parsed.
  The data cuts it at 50 characters, so a description of exactly that length ends in "…".

### The fonts

| File | Source | Licence |
|------|--------|---------|
| `bdo-grotesk.woff2` | `Fonts/Variable/BDOGrotesk-VF.ttf` of [LCTipografi/BDO-Grotesk](https://github.com/LCTipografi/BDO-Grotesk), weight axis cut to 400-500 | SIL OFL 1.1, no Reserved Font Name |
| `recommenditos-mono-400.woff2`, `-500.woff2` | Basically A Mono 1.301 Regular and Medium from [Mental Design](https://experi.mentallydesigned.com/type/bam) | SIL OFL 1.1 with the Reserved Font Name "Basically A Mono" |

A subset is a Modified Version under the OFL, so the mono's subsets must not carry the reserved name: `tools/web_fonts.py --rename` calls them Recommenditos Mono in every name record but the copyright and licence ones, and `OFL-basically-a-mono.txt` keeps the licence with a line saying so.
BDO Grotesk keeps its name.
The tool lists the characters (Latin, Latin-1 and Latin Extended-A for names such as Škoda, Łódź or Citroën, the euro and minus signs and the page's punctuation; ASCII and a few signs for the mono) and writes each file:

```bash
uv run python -m tools.web_fonts BDOGrotesk-VF.ttf \
    recommenditos/api/web/fonts/bdo-grotesk.woff2 --weight 400:500 --tabular-digits-only
uv run python -m tools.web_fonts BasicallyAMono-Regular.ttf \
    recommenditos/api/web/fonts/recommenditos-mono-400.woff2 --mono --rename "Recommenditos Mono"
```

BDO Grotesk's `tnum` feature also widens the comma, full stop and percent sign, so `--tabular-digits-only` keeps it for the figures alone.
It lacks the Esperanto and breve letters, Ĳ, ĸ, ŉ, ſ, µ and the double prime; of these the listing names hold only two double primes, which fall back to the system's face.
`tests/api/test_web.py` checks that every other one of those characters, and every character of the page's own files, is in the text face, that `tnum` changes the figures and nothing else, that no mono file carries the reserved name, that each family ships with its licence, and that the fonts weigh no more than the first design's 79.7 KB.

## What it calls

- `GET /openapi.json` once, for the option lists of the selects and the limits of the number fields, so the form offers exactly what the served model accepts.
- `GET /health` once, for the model's name and training date in the header.
- For each estimate, `POST /v1/predict`, `POST /v1/price-range` and `POST /v1/comparables` (with `k` 8) in parallel, with the same body.
  A second press of the button while an estimate runs is ignored; nothing is cancelled.

The page never calls `/v1/feedback` and sends no price anywhere: it only shows estimates.
The form has the ten fields `/v1/predict` requires and the six optional fields the served `lgbm-basic` model also reads (drive, previous owners, engine size, gears, doors, seats).
The API's other optional fields, such as the equipment lists and colours, are left out, because `lgbm-basic` does not read them and they would not change the estimate.

## Errors

- A field the form can check itself, a missing value or a number outside the limits of `/openapi.json`, is marked at the field before anything is sent, and the first such field gets the focus.
- A 422 from the API is shown the same way, field by field, with the values it allows where it lists them.
- Any other problem is shown above the result, once to a screen reader (`role="alert"`), with a button to try again where that can help.
  A 503, such as the one nginx answers while the API restarts, or no answer at all says that the price service is not answering; a 500 says the estimate failed and names the request to quote, when the problem carries one (`instance`).
- When only the range or the listings fail, the estimate is shown and the failed part says why.

One case the page cannot handle: while the API is down, for example for the few seconds of a deploy or a restart, nginx answers every path with its own 503, so a browser that loads `/` in that moment sees nginx's JSON problem instead of the page.
An HTML fallback would need an `error_page` for `/` in `deploy/nginx.conf`, which reaches the VM only through the bootstrap; it is a possible follow-up, not part of this change.

## Accessibility

Every field has a label, errors are tied to their field with `aria-describedby` and `aria-invalid`, the result is announced in a live region, and the page works with the keyboard alone; the button keeps the focus while an estimate runs (`aria-disabled`).
The waterfall is drawn for the eye and repeated as a table for screen readers.
Contrast is listed under [The design](#the-design); both animations are switched off under `prefers-reduced-motion`.

## The traffic of the page, for M6

The page marks every request it sends with the header `X-Client: web-ui`.
Two other kinds of test traffic are marked the same way: the integration suite, which `make smoke` runs against the VM after every deploy, sends `X-Client: smoke`, and the load test (`tools/loadtest/locustfile.py`) sends `X-Client: loadtest`.
The prediction log stores the mark in the column `client` of `predictions`, which is NULL for every other request ([API docs](api.md#what-a-record-holds)).
M6's drift job compares real requests with the training data, so it should leave all marked requests out:

```sql
SELECT inputs FROM predictions WHERE client IS NULL;
```

Only the values the API knows (`KNOWN_CLIENTS` in `recommenditos/api/prediction_log.py`: `web-ui`, `smoke` and `loadtest`) are stored; any other value of the header is stored as NULL, because the header is the client's to choose and a free value could identify the caller (NFR-08).
The mark is a convention, not a control: a client can send it or leave it out, so it separates the page's traffic from callers that do not imitate it.

## Rate limiting

The page adds no rate limit, and nginx has none.
A limit per client address cannot be set without risk: the load test sends NFR-03's 20 requests per second from one address ([Load testing](load-testing.md)), and whether the FIB port forwarding keeps the clients' own addresses has not been checked.
A change to `deploy/nginx.conf` also reaches the VM only through the bootstrap.
The page itself sends one estimate, three requests, at a time.

## Work on it

`make api` serves the page at `http://127.0.0.1:8000/` from the checkout; the files are read at startup, so a change shows after a restart of `make api`.
`tests/api/test_web.py` checks the fonts (above), the colour contrast, the sticky button row, the negotiation of `/`, the headers, the caching, the compression, that every asset the page names is served, that every select finds its values in `/openapi.json`, and that the page never names the feedback endpoint.
`tests/api/test_web_script.py` runs the pure functions of `app.js` in Node; it is skipped on a machine without Node, but fails in CI (`CI` set) when Node is missing: reading the option lists, mapping a 422 to its fields, the waterfall's steps, the range's scale, the stacking of close circles, the signed numbers, the title and quoted description of a listing and the text of a problem without a reference.
`tests/integration/test_deployed_proxy.py` checks the page through nginx.
The Content Security Policy allows scripts, styles, fonts and requests from the page's own origin only and no inline script or style, so a new script or style goes into a file of `web/`, and a setting from JavaScript goes through the DOM (`element.style`), not a `style` attribute.
