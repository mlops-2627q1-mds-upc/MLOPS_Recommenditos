The web page
============

A browser that opens the service's address, `http://<host>:8080/` locally and on the VM, gets a web page for the price API ([#269](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/269), [EDN-96](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md)).
It is a form for the car's details on the left and the answer on the right: the estimated asking price, the price range with the similar listings marked on it, a waterfall of what moved the estimate, and the similar listings as a table with how far each lies from the estimate.
It is made for a laptop or desktop screen and holds its layout down to a window about 1,000 px wide; a phone layout is out of scope.

## How it is built and served

| File | What it is |
|------|------------|
| `recommenditos/api/web/index.html` | The page; `{{assets}}` stands for the versioned asset path |
| `recommenditos/api/web/app.css` | Hand-written styles, light and dark |
| `recommenditos/api/web/app.js` | Plain JavaScript, one ES module, no framework and no build step |
| `recommenditos/api/web/theme.js` | Applies a theme chosen with the toggle before the page paints |
| `recommenditos/api/web/fonts/` | Newsreader (headline and price), Public Sans (text) and JetBrains Mono (labels and IDs), subset to Latin, each with its SIL Open Font License file |
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
| The page | 11.9 KB | 2.9 KB |
| `app.css` | 14.8 KB | 3.8 KB |
| `app.js` | 34.3 KB | 11.0 KB |
| `theme.js`, `favicon.svg` | 0.7 KB | 0.4 KB |
| Newsreader, Public Sans, JetBrains Mono (woff2) | 79.7 KB | 79.7 KB |
| In all | 141.4 KB | 97.8 KB |

The page also reads `/openapi.json` (52 KB, uncompressed) and `/health` on every load; the assets are cached for a year after the first visit.
Held in memory with their gzip copies, the files take about 180 KB per worker process.
After 300 loads of the page with all its assets, 2,400 requests in 3.6 s, nginx used 7.5 MiB against 7.3 MiB idle, of its 64 MiB limit, and the API 251.3 MiB against 250.0 MiB.

## The design

The page reads as a printed valuation slip: the price large in a serif, everything else small, exact and ruled.

- **Type:** five sizes only, 11, 13, 15, 32 and 96 px. Newsreader for the headline (32) and the price (96, tabular lining figures); Public Sans for text, fields and tables, with tabular figures wherever numbers line up; JetBrains Mono at 11 px for section and group labels, table headers, axis ticks and the model and request IDs.
- **Colour:** white and navy, dark navy in the dark theme, three text greys of the navy hue and one flat navy-blue accent for the primary button, the range marker, the estimate bar and the focus ring. Errors use a restrained red of their own, and nothing else is red.
  Text reaches WCAG AA (4.5:1) in both themes; field borders, the range band's end ticks, the start bar and the effect bars of the waterfall reach 3:1.
- **Structure:** a 1 px ink rule above each result section and hairlines inside it instead of cards; no shadows, gradients or tinted panels; one 4 px radius, on fields and the button only.
- **Spacing:** 4, 8, 12, 16, 24, 32, 48 and 96 px. At 1,440 px the columns are 360 and 824 px with a 96 px gap; below 1,200 px they are 312 px and the rest with a 48 px gap, and the similar listings fold registration and power into their second line once the column is under 680 px.
- **Charts:** drawn as SVG at the column's measured pixel width, so their text stays 13 and 11 px at every width. In the waterfall a solid bar raises the price and an outlined bar lowers it, so the sign does not depend on colour.
- **Motion:** two animations only, both off under `prefers-reduced-motion`: the range marker slides to the new estimate in 320 ms, and the waterfall's bars grow in turn, 160 ms each.
- **Similar listings:** "vs. estimate" says how far each listing's asking price lies from the estimate, in whole percent. No deal tiers such as "good deal": the listings are other cars, and the API gives no boundaries for tiers.

The direction comes from a study of reference sites at 1,440 px (styles measured in the browser): Linear for the tool as the first screen and the mono for data, Resend for a serif display at regular weight, Vercel's Geist for hairlines without shadows, Ramp for small uppercase labels, CarGurus' price-trend table (an archived copy) and Bring a Trailer's auction result for the result layout.
Zillow and Cars & Bids could not be studied: both answer with a bot check, live and archived, so nothing is taken from them.
One deliberate difference from the brief of that study: the form column is not sticky, because at 900 px of window height the form is taller than the window and its button would scroll out of reach.

## What it calls

- `GET /openapi.json` once, for the option lists of the selects and the limits of the number fields, so the form offers exactly what the served model accepts.
- `GET /health` once, for the model's name and training date in the header.
- For each estimate, `POST /v1/predict`, `POST /v1/price-range` and `POST /v1/comparables` (with `k` 8) in parallel, with the same body.
  A second estimate cancels the first if it has not finished.

The page never calls `/v1/feedback` and sends no price anywhere: it only shows estimates.
The form has the ten fields `/v1/predict` requires and the six optional fields the served `lgbm-basic` model also reads (drive, previous owners, engine size, gears, doors, seats).
The API's other optional fields, such as the equipment lists and colours, are left out, because `lgbm-basic` does not read them and they would not change the estimate.

## Errors

- A field the form can check itself, a missing value or a number outside the limits of `/openapi.json`, is marked at the field before anything is sent, and the first such field gets the focus.
- A 422 from the API is shown the same way, field by field, with the values it allows where it lists them.
- Any other problem, such as the 503 nginx answers while the API restarts, is shown above the result with its title, detail and the request's reference (`instance`).
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
`tests/api/test_web.py` checks the negotiation of `/`, the headers, the caching, the compression, that every asset the page names is served, that every select finds its values in `/openapi.json`, and that the page never names the feedback endpoint.
`tests/api/test_web_script.py` runs the pure functions of `app.js` in Node, when Node is installed: reading the option lists, mapping a 422 to its fields, the waterfall's steps, the range's scale, the stacking of close circles, the signed numbers and the text of a problem without a reference.
`tests/integration/test_deployed_proxy.py` checks the page through nginx.
The Content Security Policy allows scripts, styles, fonts and requests from the page's own origin only and no inline script or style, so a new script or style goes into a file of `web/`, and a setting from JavaScript goes through the DOM (`element.style`), not a `style` attribute.
