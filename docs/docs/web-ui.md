The web page
============

A browser that opens the service's address, `http://<host>:8080/` locally and on the VM, gets a web page for the price API ([#269](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/269), [EDN-96](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md)).
It is a form for the car's details on the left and the answer on the right: the estimated asking price, the price range with the similar listings marked on it, a waterfall of what moved the estimate, and the similar listings as a table.
It is made for a laptop or desktop screen and holds its layout down to a window about 1,000 px wide; a phone layout is out of scope.

## How it is built and served

| File | What it is |
|------|------------|
| `recommenditos/api/web/index.html` | The page; `{{assets}}` stands for the versioned asset path |
| `recommenditos/api/web/app.css` | Hand-written styles, light and dark |
| `recommenditos/api/web/app.js` | Plain JavaScript, one ES module, no framework and no build step |
| `recommenditos/api/web/theme.js` | Applies a theme chosen with the toggle before the page paints |
| `recommenditos/api/web/fonts/` | Instrument Serif and Instrument Sans, subset to Latin, with their SIL Open Font License files |
| `recommenditos/api/webui.py` | Reads the files once at startup and serves them from memory |
| `recommenditos/api/routes/index.py`, `routes/assets.py` | `GET /` and `GET /assets/<version>/<file>` |

The files are part of the `recommenditos` package, so they ship in the API image and reach the VM with every deploy, and a rollback puts the matching page back with the API.
nginx needs no change: its `location /` already passes every path but `/feedback` and `/metrics` to the API ([Deployment](deployment.md#the-stack)).
A change to `deploy/nginx.conf` only reaches the VM by running the bootstrap again, which is why the page does not depend on one.

`GET /` answers with the page when the request's `Accept` header names `text/html` with at least the weight of `application/json`, as a browser's does.
Every other client, curl, an HTTP library or an uptime monitor, gets the JSON index as before, and both answers say `Vary: Accept`.
The assets live under `/assets/<version>/`, where the version is a hash of all files of the page, so they are sent with `Cache-Control: public, max-age=31536000, immutable`; the page itself is sent with `no-cache` and an `ETag`, so a deploy shows at the next load.
Text files are gzip-compressed once at startup.

What a first visit loads, measured through nginx on the local stack on 2026-11-02:

| File | Size | Sent (gzip) |
|------|------|-------------|
| The page | 13.0 KB | 3.2 KB |
| `app.css` | 16.8 KB | 4.1 KB |
| `app.js` | 27.8 KB | 9.0 KB |
| `theme.js`, `favicon.svg` | 0.7 KB | 0.4 KB |
| Instrument Sans, Instrument Serif (woff2) | 47.6 KB | 47.6 KB |
| In all | 105.9 KB | 64.3 KB |

The page also reads `/openapi.json` (51 KB, uncompressed) and `/health` on every load.
Held in memory with their gzip copies, the files take about 135 KB per worker process.
After 300 loads of the page with all its assets, 2,100 requests in 3.2 s, nginx used 15.0 MiB against 14.4 MiB idle, of its 64 MiB limit, and the API 253.8 MiB against 251.2 MiB.

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

## Accessibility

Every field has a label, errors are tied to their field with `aria-describedby` and `aria-invalid`, the result is announced in a live region, and the page works with the keyboard alone.
Text meets WCAG AA contrast in both themes (4.5:1 and more), and the borders of the fields and the waterfall bars of each effect 3:1; the range marker's one animation is switched off under `prefers-reduced-motion`.

## The traffic of the page, for M6

The page marks every request it sends with the header `X-Client: web-ui`.
The prediction log stores the mark in the column `client` of `predictions`, which is NULL for every other request ([API docs](api.md#what-a-record-holds)).
M6's drift job compares real requests with the training data, so it should leave the page's demo requests out:

```sql
SELECT inputs FROM predictions WHERE client IS NULL;
```

Only the values the API knows (`KNOWN_CLIENTS` in `recommenditos/api/prediction_log.py`, now only `web-ui`) are stored; any other value of the header is stored as NULL, because the header is the client's to choose and a free value could identify the caller (NFR-08).
The mark is a convention, not a control: a client can send it or leave it out, so it separates the page's traffic from callers that do not imitate it.

## Rate limiting

The page adds no rate limit, and nginx has none.
A limit per client address cannot be set without risk: the load test sends NFR-03's 20 requests per second from one address ([Load testing](load-testing.md)), and whether the FIB port forwarding keeps the clients' own addresses has not been checked.
A change to `deploy/nginx.conf` also reaches the VM only through the bootstrap.
The page itself sends one estimate, three requests, at a time.

## Work on it

`make api` serves the page at `http://127.0.0.1:8000/` from the checkout; a reload of the server picks up a changed file.
`tests/api/test_web.py` checks the negotiation of `/`, the headers, the caching, the compression, that every asset the page names is served, that every select finds its values in `/openapi.json`, and that the page never names the feedback endpoint.
`tests/integration/test_deployed_proxy.py` checks the page through nginx.
The Content Security Policy allows scripts, styles, fonts and requests from the page's own origin only and no inline script or style, so a new script or style goes into a file of `web/`, and a setting from JavaScript goes through the DOM (`element.style`), not a `style` attribute.
