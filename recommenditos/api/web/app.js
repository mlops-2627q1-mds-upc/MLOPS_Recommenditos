// The web page of the price API (#269, docs/docs/web-ui.md). Plain JavaScript, no build step.
//
// The form's option lists and limits come from the served /openapi.json, so the
// page offers what the running model accepts. One estimate calls /v1/predict,
// /v1/price-range and /v1/comparables in parallel, each with the X-Client header
// that marks the page's requests in the prediction log. The page never calls
// /v1/feedback and sends no price anywhere.

const CLIENT = { "X-Client": "web-ui" };
const COMPARABLES_K = 8;
const REQUIRED = [
  "make", "model", "registration_date", "mileage_km_raw", "power_kw",
  "fuel_category", "transmission", "body_type", "country_code", "seller_type",
];
const OPTIONAL = ["drive_train", "nr_prev_owners", "cylinders_volume_cc", "gears", "nr_doors", "nr_seats"];

const FIELD_LABELS = {
  make: "Make", model: "Model", registration_date: "Age", mileage_km_raw: "Mileage",
  power_kw: "Power", fuel_category: "Fuel", transmission: "Gearbox", body_type: "Body type",
  country_code: "Country", seller_type: "Seller", drive_train: "Drive",
  nr_prev_owners: "Previous owners", cylinders_volume_cc: "Engine size", gears: "Gears",
  nr_doors: "Doors", nr_seats: "Seats", cylinders: "Cylinders", weight_kg: "Weight",
  electric_range_km: "Electric range",
};

const COUNTRIES = {
  AT: "Austria", BE: "Belgium", DE: "Germany", ES: "Spain", FR: "France",
  IT: "Italy", LU: "Luxembourg", NL: "Netherlands",
};
const VALUE_LABELS = {
  seller_type: { Dealer: "Dealer", PrivateSeller: "Private seller" },
  drive_train: { "4WD": "Four-wheel drive", "Front Wheel Drive": "Front-wheel drive", "Rear Wheel Drive": "Rear-wheel drive" },
  country_code: COUNTRIES,
};

const EXAMPLE = {
  make: "BMW", model: "320", registration_year: "2020", registration_month: "03",
  mileage_km_raw: "62,000", power_kw: "140", fuel_category: "Diesel",
  transmission: "Automatic", body_type: "Station wagon", country_code: "DE", seller_type: "Dealer",
};

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

const euro = new Intl.NumberFormat("en-IE", { style: "currency", currency: "EUR", maximumFractionDigits: 0 });
const integer = new Intl.NumberFormat("en-IE", { maximumFractionDigits: 0 });
const percent = new Intl.NumberFormat("en-IE", { maximumFractionDigits: 1, minimumFractionDigits: 1, signDisplay: "exceptZero" });
const longDate = new Intl.DateTimeFormat("en-GB", { day: "numeric", month: "short", year: "numeric" });

const $ = (id) => document.getElementById(id);
const form = $("car-form");
let schema = {};
let inFlight = null;

// ------------------------------------------------------------------ the theme

function setupTheme() {
  const toggle = $("theme-toggle");
  const dark = () => {
    const chosen = document.documentElement.dataset.theme;
    return chosen ? chosen === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  };
  const label = () => toggle.setAttribute("aria-label", dark() ? "Switch to light mode" : "Switch to dark mode");
  toggle.addEventListener("click", () => {
    const next = dark() ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("theme", next); } catch { /* not remembered */ }
    label();
  });
  label();
}

// ------------------------------------------------------------- the form setup

/** The allowed values of a field of the valuation request, from the OpenAPI document. */
export function allowedValues(properties, field) {
  const property = properties[field];
  if (!property) return null;
  for (const branch of property.anyOf ?? [property]) {
    if (Array.isArray(branch.enum)) return branch.enum;
  }
  return null;
}

/** The numeric limits of a field, from the OpenAPI document. */
export function limits(properties, field) {
  const property = properties[field];
  if (!property) return {};
  const branch = (property.anyOf ?? [property]).find((b) => b.type === "number" || b.type === "integer") ?? {};
  return { min: branch.minimum, max: branch.maximum, integer: branch.type === "integer" };
}

function labelOf(field, value) {
  return VALUE_LABELS[field]?.[value] ?? value;
}

function fillSelects(properties) {
  for (const select of form.querySelectorAll("select[data-options]")) {
    const field = select.dataset.options;
    const values = allowedValues(properties, field);
    if (!values) throw new Error(`The service lists no values for ${field}.`);
    const options = values.map((value) => ({ value, text: labelOf(field, value) }));
    if (field === "country_code") options.sort((a, b) => a.text.localeCompare(b.text));
    for (const { value, text } of options) select.append(new Option(text, value));
  }
  const month = $("registration_month");
  MONTHS.forEach((name, index) => month.append(new Option(name, String(index + 1).padStart(2, "0"))));
  const year = $("registration_year");
  for (let y = new Date().getFullYear(); y >= 1950; y -= 1) year.append(new Option(String(y), String(y)));
  for (const select of form.querySelectorAll("select")) {
    const mark = () => select.classList.toggle("is-empty", select.value === "");
    select.addEventListener("change", mark);
    mark();
  }
}

async function loadService() {
  const alert = $("form-alert");
  try {
    const [openapi, health] = await Promise.all([getJson("openapi.json"), getJson("health").catch(() => null)]);
    schema = openapi.components.schemas.ValuationRequest.properties;
    fillSelects(schema);
    if (health) showModel(health);
    for (const fieldset of form.querySelectorAll("fieldset")) fieldset.disabled = false;
    $("submit").disabled = false;
    $("example").disabled = false;
    alert.hidden = true;
  } catch (error) {
    alert.replaceChildren(
      el("p", {}, "The service is not answering, so the form cannot load its lists. Try again in a few seconds."),
      el("button", { type: "button", class: "button-quiet", onclick: () => location.reload() }, "Reload the page"),
    );
    alert.hidden = false;
    console.error(error);
  }
}

function showModel(health) {
  const info = $("model-info");
  const trained = health.model.trained_at ? `, trained ${longDate.format(new Date(health.model.trained_at))}` : "";
  info.textContent = `Model ${health.model.variant}${trained}`;
  info.title = `Model version ${health.model.version}`;
  info.hidden = false;
  if (health.comparables) {
    $("intro-note").textContent =
      `Trained on AutoScout24 listings from Austria, Belgium, France, Germany, Italy, Luxembourg, the Netherlands and Spain. ` +
      `Similar listings come from ${integer.format(health.comparables.listings)} of them.`;
  }
}

// --------------------------------------------------------------- validation

function setError(field, message) {
  const error = $(`${field}-error`);
  const inputs = field === "registration_date"
    ? [$("registration_month"), $("registration_year")]
    : [form.elements[field]];
  for (const input of inputs) {
    if (!input) continue;
    if (message) {
      input.setAttribute("aria-invalid", "true");
      input.setAttribute("aria-describedby", [`${field}-error`, input.dataset.hint].filter(Boolean).join(" "));
    } else {
      input.removeAttribute("aria-invalid");
      if (input.dataset.hint) input.setAttribute("aria-describedby", input.dataset.hint);
      else input.removeAttribute("aria-describedby");
    }
  }
  if (error) {
    error.textContent = message ?? "";
    error.hidden = !message;
  }
}

function clearErrors() {
  for (const field of [...REQUIRED, ...OPTIONAL]) setError(field, null);
  $("form-alert").hidden = true;
}

function parseNumber(text) {
  const cleaned = text.replace(/[\s,.'’]/g, "");
  return /^\d+$/.test(cleaned) ? Number(cleaned) : NaN;
}

/** The request body from the form, or the field errors that stop it. */
function readForm() {
  const body = {};
  const errors = {};
  const data = new FormData(form);
  for (const field of [...REQUIRED, ...OPTIONAL]) {
    const required = REQUIRED.includes(field);
    const label = FIELD_LABELS[field].toLowerCase();
    if (field === "registration_date") {
      const month = data.get("registration_month");
      const year = data.get("registration_year");
      if (!month || !year) { errors[field] = "Choose the month and year of first registration."; continue; }
      const value = `${year}-${month}`;
      const now = new Date();
      const current = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, "0")}`;
      if (value > current) { errors[field] = "The first registration cannot be in the future."; continue; }
      body[field] = value;
      continue;
    }
    const raw = String(data.get(field) ?? "").trim();
    if (raw === "") {
      if (required) errors[field] = form.elements[field].tagName === "SELECT" ? `Choose the ${label}.` : `Enter the ${label}.`;
      continue;
    }
    const input = form.elements[field];
    if (input.dataset.number !== undefined) {
      const value = parseNumber(raw);
      const { min, max } = limits(schema, field);
      if (Number.isNaN(value)) { errors[field] = "Enter a whole number."; continue; }
      if ((min !== undefined && value < min) || (max !== undefined && value > max)) {
        errors[field] = `Enter ${integer.format(min)} to ${integer.format(max)}.`;
        continue;
      }
      body[field] = value;
    } else {
      const maxLength = schema[field]?.maxLength;
      if (maxLength && raw.length > maxLength) { errors[field] = `Use at most ${maxLength} characters.`; continue; }
      body[field] = raw;
    }
  }
  return { body, errors };
}

function showFieldErrors(errors, origin) {
  const fields = Object.keys(errors);
  for (const field of fields) setError(field, errors[field]);
  const alert = $("form-alert");
  const count = fields.length;
  const lead = origin === "server" ? "The service did not accept the request." : "";
  alert.replaceChildren(el("p", {}, `${lead} ${count === 1 ? "One field needs" : `${count} fields need`} attention.`.trim()));
  alert.hidden = false;
  const first = fields.map((f) => (f === "registration_date" ? $("registration_month") : form.elements[f])).find(Boolean);
  if (first) {
    if (first.closest("details")) first.closest("details").open = true;
    first.focus({ preventScroll: true });
    first.scrollIntoView({ block: "center", behavior: smoothScroll() });
  }
}

// ---------------------------------------------------------------- requests

async function getJson(path) {
  const response = await fetch(path, { headers: { Accept: "application/json", ...CLIENT } });
  if (!response.ok) throw new Error(`${path} answered ${response.status}`);
  return response.json();
}

class Problem extends Error {
  constructor(status, problem) {
    super(problem?.detail ?? `The service answered ${status}.`);
    this.status = status;
    this.problem = problem;
  }
}

async function post(path, body, signal) {
  let response;
  try {
    response = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json", ...CLIENT },
      body: JSON.stringify(body),
      signal,
    });
  } catch (error) {
    if (error.name === "AbortError") throw error;
    throw new Problem(0, { title: "No connection", detail: "The service could not be reached. Check the connection and try again." });
  }
  const payload = await response.json().catch(() => null);
  if (!response.ok) throw new Problem(response.status, payload);
  return payload;
}

/** Field errors of a 422 problem, by top-level field. */
export function fieldErrors(problem) {
  const errors = {};
  for (const error of problem?.errors ?? []) {
    const field = error.pointer.replace(/^#\/?/, "").split("/")[0];
    if (!field || errors[field]) continue;
    errors[field] = error.allowed?.length
      ? `${error.detail} Choose one of: ${error.allowed.join(", ")}.`
      : error.detail;
  }
  return errors;
}

let lastResult = null;

async function estimate(event) {
  event.preventDefault();
  clearErrors();
  const { body, errors } = readForm();
  if (Object.keys(errors).length) {
    showFieldErrors(errors, "client");
    return;
  }
  inFlight?.abort();
  const controller = new AbortController();
  inFlight = controller;
  setBusy(true);
  const settled = await Promise.allSettled([
    post("v1/predict", body, controller.signal),
    post("v1/price-range", body, controller.signal),
    post("v1/comparables", { ...body, k: COMPARABLES_K }, controller.signal),
  ]);
  if (controller.signal.aborted) return;
  inFlight = null;
  setBusy(false);
  const [valuation, range, comparables] = settled;
  if (valuation.status === "rejected") {
    const problem = valuation.reason;
    if (problem.status === 422) {
      const byField = fieldErrors(problem.problem);
      const known = Object.keys(byField).filter((field) => [...REQUIRED, ...OPTIONAL].includes(field));
      if (known.length) {
        showFieldErrors(Object.fromEntries(known.map((f) => [f, byField[f]])), "server");
        return;
      }
    }
    showProblem(problem);
    return;
  }
  lastResult = { body, valuation: valuation.value };
  render(body, valuation.value, range, comparables);
}

function setBusy(busy) {
  $("result").setAttribute("aria-busy", String(busy));
  const submit = $("submit");
  submit.disabled = busy;
  submit.textContent = busy ? "Estimating…" : "Estimate price";
  if (busy) $("announcer").textContent = "Estimating the price.";
}

// --------------------------------------------------------------- rendering

function el(tag, attributes = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attributes)) {
    if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else if (value !== undefined && value !== null) node.setAttribute(key, value);
  }
  node.append(...children.filter((child) => child !== null && child !== undefined));
  return node;
}

const SVG = "http://www.w3.org/2000/svg";
function svg(tag, attributes = {}, text) {
  const node = document.createElementNS(SVG, tag);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  if (text !== undefined) node.textContent = text;
  return node;
}

function showProblem(error) {
  const problem = error.problem ?? {};
  const box = $("problem");
  const title = error.status === 503
    ? "The service is busy or restarting"
    : problem.title ?? "Something went wrong";
  const detail = problem.detail ?? error.message;
  box.replaceChildren(
    el("p", { class: "problem-title" }, title),
    el("p", { class: "problem-detail" }, detail),
    problem.instance ? el("p", { class: "problem-ref" }, `Reference ${problem.instance.replace("urn:uuid:", "")}`) : null,
  );
  box.hidden = false;
  // An older estimate below the problem would read as the answer to this request.
  $("estimate").hidden = true;
  $("intro").hidden = false;
  lastResult = null;
  $("announcer").textContent = `${title}. ${detail}`;
  revealResult();
}

function smoothScroll() {
  return matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth";
}

/** Scrolls the result column into view when its top is above the window. */
function revealResult() {
  const result = $("result");
  if (result.getBoundingClientRect().top < 0) result.scrollIntoView({ block: "start", behavior: smoothScroll() });
}

function describeCar(body) {
  const [year, month] = body.registration_date.split("-");
  return [
    `${body.make} ${body.model}`,
    `${MONTHS[Number(month) - 1]} ${year}`,
    `${integer.format(body.mileage_km_raw)} km`,
    `${integer.format(body.power_kw)} kW`,
    body.fuel_category,
    body.transmission,
    COUNTRIES[body.country_code] ?? body.country_code,
  ].join(" · ");
}

function render(body, valuation, range, comparables) {
  $("intro").hidden = true;
  $("problem").hidden = true;
  $("estimate").hidden = false;
  $("price").textContent = euro.format(valuation.price_eur);
  $("car-line").textContent = describeCar(body);

  const warnings = [...valuation.warnings];
  const list = $("warnings");
  list.replaceChildren(...warnings.map((warning) => el("li", {}, warning.detail)));
  list.hidden = warnings.length === 0;

  const listings = comparables.status === "fulfilled" ? comparables.value.listings : null;
  renderRange(range, valuation.price_eur, listings);
  renderWaterfall(valuation);
  renderListings(comparables, body);

  const model = valuation.model;
  $("estimate-foot").textContent = `Model ${model.variant}, version ${model.version.slice(0, 8)}. Request ${valuation.request_id}.`;
  $("announcer").textContent = `Estimated asking price ${euro.format(valuation.price_eur)}.`;
  revealResult();
}

/** The scale of the range bar: the range and every comparable price, with some room. */
export function rangeScale(lower, upper, prices) {
  const low = Math.min(lower, ...prices);
  const high = Math.max(upper, ...prices);
  const pad = (high - low) * 0.12 || high * 0.1;
  return { min: Math.max(0, low - pad), max: high + pad };
}

function renderRange(range, price, listings) {
  const box = $("range");
  const error = $("range-error");
  if (range.status === "rejected") {
    box.hidden = true;
    $("range-level").textContent = "";
    $("range-note").textContent = "";
    error.textContent = `No range for this car: ${range.reason.message}`;
    error.hidden = false;
    return;
  }
  error.hidden = true;
  box.hidden = false;
  const { lower_eur: lower, upper_eur: upper, confidence_level: level } = range.value;
  const prices = (listings ?? []).map((listing) => listing.price_eur);
  const scale = rangeScale(lower, upper, prices);
  const at = (value) => `${(((value - scale.min) / (scale.max - scale.min)) * 100).toFixed(2)}%`;
  box.style.setProperty("--low", at(lower));
  box.style.setProperty("--high", at(upper));
  // The marker settles on the estimate from the middle of the range, the one motion that explains something.
  box.style.setProperty("--at", at((lower + upper) / 2));
  requestAnimationFrame(() => requestAnimationFrame(() => box.style.setProperty("--at", at(price))));
  $("range-marker-label").textContent = euro.format(price);
  $("range-min").textContent = euro.format(lower);
  $("range-max").textContent = euro.format(upper);
  $("range-ticks").replaceChildren(
    ...prices.map((value) => {
      const tick = el("span", { class: "range-tick", title: `A similar listing at ${euro.format(value)}` });
      tick.style.left = at(value);
      return tick;
    }),
  );
  const share = Math.round(level * 10);
  $("range-level").textContent = `${Math.round(level * 100)} % prediction interval`;
  box.setAttribute("aria-label",
    `Range from ${euro.format(lower)} to ${euro.format(upper)}, estimate ${euro.format(price)}` +
    (prices.length ? `, with ${prices.length} similar listings marked` : ""));
  const note = $("range-note");
  note.replaceChildren(
    `The model expects ${share} in 10 cars like this to be listed between ${euro.format(lower)} and ${euro.format(upper)}.`,
    prices.length ? " The circles " : "",
    prices.length ? el("span", { class: "range-legend", "aria-hidden": "true" }) : "",
    prices.length ? " mark the asking prices of the similar listings below." : "",
  );
}

function fieldValue(field, body) {
  const value = body[field];
  if (value === undefined) return "not given";
  switch (field) {
    case "registration_date": {
      const [year, month] = value.split("-").map(Number);
      const now = new Date();
      const years = (now.getFullYear() - year) + (now.getMonth() + 1 - month) / 12;
      return `${years.toFixed(1)} years`;
    }
    case "mileage_km_raw": return `${integer.format(value)} km`;
    case "power_kw": return `${integer.format(value)} kW`;
    case "cylinders_volume_cc": return `${integer.format(value)} cc`;
    default: return labelOf(field, String(value));
  }
}

/** The steps of the waterfall: from the base price, each effect multiplies the running price. */
export function waterfallSteps(explanation) {
  const steps = [];
  let running = explanation.base_price_eur;
  const effects = [
    ...explanation.top_features.map((feature) => ({ field: feature.field, pct: feature.effect_pct })),
    { field: null, pct: explanation.other_features_effect_pct },
  ];
  for (const { field, pct } of effects) {
    const next = running * (1 + pct / 100);
    steps.push({ field, pct, from: running, to: next });
    running = next;
  }
  return { base: explanation.base_price_eur, steps, total: running };
}

function niceStep(max) {
  const raw = max / 4;
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  return [1, 2, 2.5, 5, 10].map((m) => m * magnitude).find((step) => step >= raw);
}

function compactEuro(value) {
  if (value === 0) return "€0";
  return value >= 1000 ? `€${integer.format(value / 1000)}k` : euro.format(value);
}

function renderWaterfall(valuation) {
  const container = $("waterfall");
  const body = lastResult?.body ?? {};
  const { base, steps, total } = waterfallSteps(valuation.explanation);
  const width = Math.max(480, container.clientWidth || 640);
  const labelWidth = Math.min(210, width * 0.3);
  const effectWidth = 112;
  const plotLeft = labelWidth + 12;
  const plotRight = width - effectWidth - 12;
  const rowHeight = 40;
  const barHeight = 18;
  const rows = [
    { kind: "base", label: "Starting point", value: "average listing", from: 0, to: base },
    ...steps.map((step) => ({
      kind: step.to >= step.from ? "up" : "down",
      label: step.field ? FIELD_LABELS[step.field] ?? step.field : "All other details",
      value: step.field ? fieldValue(step.field, body) : "combined",
      from: step.from, to: step.to, pct: step.pct,
    })),
    { kind: "total", label: "Estimate", value: "", from: 0, to: total },
  ];
  const top = 6;
  const axisY = top + rows.length * rowHeight + 6;
  const height = axisY + 22;
  const maxValue = Math.max(...rows.map((row) => Math.max(row.from, row.to))) * 1.04;
  const step = niceStep(maxValue);
  const domainMax = Math.ceil(maxValue / step) * step;
  const x = (value) => plotLeft + (value / domainMax) * (plotRight - plotLeft);

  const root = svg("svg", { viewBox: `0 0 ${width} ${height}`, width, height, role: "img" });
  root.append(svg("title", {}, `Waterfall from the starting price ${euro.format(base)} to the estimate ${euro.format(total)}`));
  for (let tick = 0; tick <= domainMax + 1e-6; tick += step) {
    root.append(svg("line", { class: "wf-grid", x1: x(tick), x2: x(tick), y1: top, y2: axisY }));
    root.append(svg("text", { class: "wf-axis", x: x(tick), y: axisY + 16, "text-anchor": "middle" }, compactEuro(tick)));
  }
  rows.forEach((row, index) => {
    const y = top + index * rowHeight;
    const barY = y + (rowHeight - barHeight) / 2;
    const left = x(Math.min(row.from, row.to));
    const barWidth = Math.max(2, Math.abs(x(row.to) - x(row.from)));
    if (row.kind === "total") {
      root.append(svg("line", { class: "wf-rule", x1: 0, x2: width, y1: y + 0.5, y2: y + 0.5 }));
    }
    root.append(svg("text", { class: "wf-label", x: 0, y: y + rowHeight / 2 - 1, "dominant-baseline": "middle" }, row.label));
    if (row.value) {
      const label = root.lastChild;
      label.setAttribute("y", y + rowHeight / 2 - 8);
      root.append(svg("text", { class: "wf-value", x: 0, y: y + rowHeight / 2 + 9, "dominant-baseline": "middle" }, row.value));
    }
    root.append(svg("rect", { class: `wf-bar-${row.kind}`, x: left, y: barY, width: barWidth, height: barHeight, rx: 2 }));
    if (index < rows.length - 1 && row.kind !== "total") {
      const end = x(row.to);
      root.append(svg("line", { class: "wf-link", x1: end, x2: end, y1: barY + barHeight, y2: barY + rowHeight }));
    }
    const effect = row.kind === "base" || row.kind === "total"
      ? euro.format(row.to)
      : `${percent.format(row.pct)} %`;
    const delta = row.kind === "up" || row.kind === "down"
      ? `${row.to >= row.from ? "+" : "−"}${euro.format(Math.abs(row.to - row.from))}`
      : "";
    root.append(svg("text", {
      class: "wf-effect", x: width, y: delta ? y + rowHeight / 2 - 8 : y + rowHeight / 2 - 1,
      "text-anchor": "end", "dominant-baseline": "middle",
    }, effect.replace("-", "−")));
    if (delta) {
      root.append(svg("text", { class: "wf-delta", x: width, y: y + rowHeight / 2 + 9, "text-anchor": "end", "dominant-baseline": "middle" }, delta));
    }
  });
  container.replaceChildren(root);
}

function renderListings(comparables, body) {
  const wrap = $("listings-wrap");
  const empty = $("listings-empty");
  const error = $("listings-error");
  const aside = $("listings-aside");
  empty.hidden = true;
  error.hidden = true;
  wrap.hidden = true;
  aside.textContent = "";
  if (comparables.status === "rejected") {
    error.textContent = `No listings for this car: ${comparables.reason.message}`;
    error.hidden = false;
    return;
  }
  const { listings, index } = comparables.value;
  if (!listings.length) {
    empty.textContent =
      `None of the ${integer.format(index.listings)} listings in the data is a ${body.make} ${body.model} ` +
      "within two years of this car's age and 25 % of its mileage.";
    empty.hidden = false;
    return;
  }
  aside.textContent = `${listings.length} closest by mileage and age, of ${integer.format(index.listings)}`;
  $("listings-body").replaceChildren(...listings.map((listing) => {
    const name = [listing.model, listing.model_version].filter(Boolean).join(" · ");
    const meta = [
      listing.fuel_category, listing.body_type, labelOf("seller_type", listing.seller_type),
      COUNTRIES[listing.country_code] ?? listing.country_code,
    ].filter(Boolean).join(" · ");
    const [year, month] = (listing.registration_date ?? "").split("-");
    return el("tr", {},
      el("td", {}, el("span", { class: "listing-name", title: name }, name), el("span", { class: "listing-meta" }, meta)),
      el("td", {}, year ? `${MONTHS[Number(month) - 1]} ${year}` : "–"),
      el("td", { class: "num" }, listing.mileage_km_raw === null ? "–" : `${integer.format(listing.mileage_km_raw)} km`),
      el("td", { class: "num" }, listing.power_kw === null ? "–" : `${integer.format(listing.power_kw)} kW`),
      el("td", { class: "num listing-price" }, euro.format(listing.price_eur)),
    );
  }));
  wrap.hidden = false;
}

// ------------------------------------------------------------------- start

function fillExample() {
  clearErrors();
  for (const [field, value] of Object.entries(EXAMPLE)) {
    const input = form.elements[field];
    input.value = value;
    input.dispatchEvent(new Event("change"));
  }
  $("submit").focus();
}

function setupNumbers() {
  for (const input of form.querySelectorAll("[data-number]")) {
    input.addEventListener("blur", () => {
      const value = parseNumber(input.value.trim());
      if (!Number.isNaN(value)) input.value = integer.format(value);
    });
  }
}

function setupPowerHint() {
  const input = $("power_kw");
  const hint = $("power-hint");
  input.dataset.hint = "power-hint";
  input.addEventListener("input", () => {
    const value = parseNumber(input.value.trim());
    hint.textContent = Number.isNaN(value) || value === 0 ? "kW, as the listing states it" : `About ${integer.format(value * 1.36)} hp`;
  });
}

if (typeof document !== "undefined" && form) {
  setupTheme();
  setupPowerHint();
  setupNumbers();
  form.addEventListener("submit", estimate);
  $("example").addEventListener("click", fillExample);
  let resizeFrame = 0;
  new ResizeObserver(() => {
    cancelAnimationFrame(resizeFrame);
    resizeFrame = requestAnimationFrame(() => { if (lastResult) renderWaterfall(lastResult.valuation); });
  }).observe($("waterfall"));
  loadService();
}
