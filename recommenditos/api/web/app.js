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

/** What a field's error asks for when it is left empty. */
const EMPTY_MESSAGES = {
  make: "Choose a make.",
  model: "Enter the model, for example 320 or Golf.",
  mileage_km_raw: "Enter the mileage in km, for example 62000.",
  power_kw: "Enter the power in kW, for example 140.",
  fuel_category: "Choose a fuel.",
  transmission: "Choose a gearbox.",
  body_type: "Choose a body type.",
  country_code: "Choose a country.",
  seller_type: "Choose a seller type.",
};

/** What a number field's range error names: its unit and an example. */
const NUMBER_HINTS = {
  mileage_km_raw: { what: "the mileage in km", example: "62000" },
  power_kw: { what: "the power in kW", example: "140" },
  cylinders_volume_cc: { what: "the engine size in cc", example: "1998" },
  nr_prev_owners: { what: "the number of previous owners", example: "1" },
  gears: { what: "the number of gears", example: "6" },
  nr_doors: { what: "the number of doors", example: "5" },
  nr_seats: { what: "the number of seats", example: "5" },
};

/** A warning names its field as the person reads it, which for `model` is the model name. */
const WARNING_LABELS = { model: "Model name", registration_date: "First registration", country_code: "Country" };

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
const MINUS = "−";

const euro = new Intl.NumberFormat("en-IE", { style: "currency", currency: "EUR", maximumFractionDigits: 0 });
const integer = new Intl.NumberFormat("en-IE", { maximumFractionDigits: 0 });
const longDate = new Intl.DateTimeFormat("en-GB", { day: "numeric", month: "short", year: "numeric" });

/** A signed percentage with a real minus sign: "+9.2%", "−21.7%". */
export function signedPercent(value, digits = 1) {
  const rounded = Number(value.toFixed(digits));
  if (rounded === 0) return `${(0).toFixed(digits)}%`;
  return `${rounded > 0 ? "+" : MINUS}${Math.abs(rounded).toFixed(digits)}%`;
}

/** A signed amount in euros: "+€2,410", "−€8,343". */
function signedEuro(value) {
  return `${value >= 0 ? "+" : MINUS}${euro.format(Math.abs(value))}`;
}

const $ = (id) => document.getElementById(id);
// Null outside a browser, where the tests import the pure functions below.
const form = typeof document === "undefined" ? null : $("car-form");
let schema = {};
let busy = false;
let lastResult = null;
let lastMarkerX = null;

// ------------------------------------------------------------------ the theme

function setupTheme() {
  const toggle = $("theme-toggle");
  const dark = () => {
    const chosen = document.documentElement.dataset.theme;
    return chosen ? chosen === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  };
  const label = () => {
    toggle.textContent = dark() ? "Light" : "Dark";
    toggle.setAttribute("aria-label", dark() ? "Switch to the light theme" : "Switch to the dark theme");
  };
  toggle.addEventListener("click", () => {
    const next = dark() ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("theme", next); } catch { /* not remembered */ }
    label();
  });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", label);
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
      el("p", {}, "The price service is not answering, so the form cannot load its lists. Try again in a minute."),
      el("button", { type: "button", class: "button-quiet", onclick: () => location.reload() }, "Reload the page"),
    );
    alert.hidden = false;
    console.error(error);
  }
}

function showModel(health) {
  const info = $("model-info");
  const trained = health.model.trained_at ? ` · trained ${longDate.format(new Date(health.model.trained_at))}` : "";
  info.textContent = `Model ${health.model.variant}${trained}`;
  info.title = `Model version ${health.model.version}`;
  info.hidden = false;
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
      if (required) errors[field] = EMPTY_MESSAGES[field] ?? `Enter the ${FIELD_LABELS[field].toLowerCase()}.`;
      continue;
    }
    const input = form.elements[field];
    if (input.dataset.number !== undefined) {
      const value = parseNumber(raw);
      const { min, max } = limits(schema, field);
      if (Number.isNaN(value)) { errors[field] = "Enter a whole number."; continue; }
      if ((min !== undefined && value < min) || (max !== undefined && value > max)) {
        const hint = NUMBER_HINTS[field];
        errors[field] = hint
          ? `Enter ${hint.what} between ${integer.format(min)} and ${integer.format(max)}, for example ${hint.example}.`
          : `Enter a number between ${integer.format(min)} and ${integer.format(max)}.`;
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

function showFieldErrors(errors) {
  const fields = Object.keys(errors);
  for (const field of fields) setError(field, errors[field]);
  const alert = $("form-alert");
  const labels = fields.map((field) => (field === "registration_date" ? "First registration" : FIELD_LABELS[field] ?? field));
  const count = fields.length;
  alert.replaceChildren(el("p", {}, `${count === 1 ? "One field needs" : `${count} fields need`} attention: ${labels.join(", ")}.`));
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

async function post(path, body) {
  let response;
  try {
    response = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json", ...CLIENT },
      body: JSON.stringify(body),
    });
  } catch {
    throw new Problem(0, null);
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

/**
 * What the page says about a failed estimate: a title, a detail, and whether a retry may help.
 *
 * A 503 or no answer at all is the service restarting or unreachable; nginx's own
 * 503 names no instance, so a reference appears only when the problem has one.
 */
export function describeProblem(status, problem) {
  const reference = typeof problem?.instance === "string" ? problem.instance.replace("urn:uuid:", "") : null;
  if (status === 0 || status === 502 || status === 503 || status === 504) {
    return { title: "The price service is not answering.", detail: "Your inputs are kept; try again in a minute.", retry: true };
  }
  if (status >= 500) {
    const quote = reference ? `quote request ${reference}` : "say when it happened";
    return {
      title: `The service failed while estimating this car (error ${status}).`,
      detail: `Try again; if it repeats, ${quote}.`,
      retry: true,
    };
  }
  if (status === 422) {
    const [field, message] = Object.entries(fieldErrors(problem))[0] ?? [];
    const detail = field
      ? `The service did not accept ${FIELD_LABELS[field]?.toLowerCase() ?? field}: ${message}`
      : problem?.detail ?? "The service did not accept the request.";
    return { title: "The request was not accepted.", detail, retry: false };
  }
  return {
    title: problem?.title ?? `The service answered ${status}.`,
    detail: problem?.detail ?? "Try again in a minute.",
    retry: false,
  };
}

async function estimate(event) {
  event?.preventDefault();
  if (busy) return;
  clearErrors();
  const { body, errors } = readForm();
  if (Object.keys(errors).length) {
    showFieldErrors(errors);
    return;
  }
  setBusy(true);
  const settled = await Promise.allSettled([
    post("v1/predict", body),
    post("v1/price-range", body),
    post("v1/comparables", { ...body, k: COMPARABLES_K }),
  ]);
  setBusy(false);
  const [valuation, range, comparables] = settled;
  if (valuation.status === "rejected") {
    const problem = valuation.reason;
    if (problem.status === 422) {
      const byField = fieldErrors(problem.problem);
      const known = Object.keys(byField).filter((field) => [...REQUIRED, ...OPTIONAL].includes(field));
      if (known.length) {
        showFieldErrors(Object.fromEntries(known.map((f) => [f, byField[f]])));
        return;
      }
    }
    showProblem(problem);
    return;
  }
  lastResult = { body, valuation: valuation.value, range, comparables };
  render(lastResult, true);
}

/** Marks the estimate as running. The button stays focusable (aria-disabled), so focus stays on it. */
function setBusy(running) {
  busy = running;
  $("result").setAttribute("aria-busy", String(running));
  const submit = $("submit");
  submit.setAttribute("aria-disabled", String(running));
  submit.textContent = running ? "Estimating…" : "Estimate price";
  $("price-label").textContent = running && lastResult ? "Estimated asking price · updating" : "Estimated asking price";
  if (running) $("announcer").textContent = "Estimating the price.";
}

// --------------------------------------------------------------- rendering

function el(tag, attributes = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attributes)) {
    if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else if (value !== undefined && value !== null && value !== false) node.setAttribute(key, value);
  }
  node.append(...children.filter((child) => child !== null && child !== undefined && child !== false && child !== ""));
  return node;
}

const SVG = "http://www.w3.org/2000/svg";
function svg(tag, attributes = {}, text) {
  const node = document.createElementNS(SVG, tag);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  if (text !== undefined) node.textContent = text;
  return node;
}

function smoothScroll() {
  return matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth";
}

/** Scrolls the result column into view when its top is above the window. */
function revealResult() {
  const result = $("result");
  if (result.getBoundingClientRect().top < 0) result.scrollIntoView({ block: "start", behavior: smoothScroll() });
}

/** Puts the result back to its empty state, so that nothing stale stays on screen. */
function resetResult() {
  $("estimate").classList.add("is-empty");
  $("price").textContent = "No car yet.";
  $("car-line").textContent = "";
  $("warnings").hidden = true;
  for (const id of ["range", "waterfall", "range-note", "range-level", "listings-aside", "estimate-foot"]) $(id).replaceChildren();
  $("listings-wrap").hidden = true;
  for (const id of ["listings-empty", "listings-error", "range-error"]) $(id).hidden = true;
  lastResult = null;
  lastMarkerX = null;
}

function showProblem(error) {
  const box = $("problem");
  const { title, detail, retry } = describeProblem(error.status, error.problem);
  const parts = [el("p", { class: "problem-title" }, title), el("p", { class: "problem-detail" }, detail)];
  if (retry) parts.push(el("button", { type: "button", class: "button-quiet", onclick: () => estimate() }, "Try again"));
  box.replaceChildren(...parts);
  box.hidden = false;
  // An older estimate below the problem would read as the answer to this request.
  resetResult();
  // The box is role="alert", which announces it once; the live region stays quiet.
  $("announcer").textContent = "";
  revealResult();
}

/** The price, with the euro sign in a span of its own so it can be kerned against the figures. */
function setPrice(value) {
  const text = euro.format(value);
  const price = $("price");
  if (text.startsWith("€")) {
    price.replaceChildren(el("span", { class: "price-currency" }, "€"), text.slice(1));
  } else {
    price.textContent = text;
  }
}

/** The longest `model_version` the data keeps; a version of exactly this length was cut. */
export const VERSION_MAX_LENGTH = 50;

/** A listing's name: its version line, led by the model unless the line starts with it. */
export function listingName(listing) {
  const version = (listing.model_version ?? "").trim();
  const model = (listing.model ?? "").trim();
  if (!version) return model || listing.make;
  const cut = version.length >= VERSION_MAX_LENGTH ? `${version}…` : version;
  return model && !version.toLowerCase().startsWith(model.toLowerCase()) ? `${model} ${cut}` : cut;
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
    labelOf("seller_type", body.seller_type),
  ].join(" · ");
}

function render(result, fresh) {
  const { body, valuation, range, comparables } = result;
  $("problem").hidden = true;
  $("estimate").classList.remove("is-empty");
  setPrice(valuation.price_eur);
  $("car-line").textContent = describeCar(body);

  const list = $("warnings-list");
  list.replaceChildren(...valuation.warnings.map((warning) => {
    const field = warning.pointer.replace(/^#\/?/, "").split("/")[0];
    const label = WARNING_LABELS[field] ?? FIELD_LABELS[field];
    return el("li", {}, label ? `${label}: ${warning.detail}` : warning.detail);
  }));
  $("warnings").hidden = valuation.warnings.length === 0;

  const listings = comparables.status === "fulfilled" ? comparables.value.listings : [];
  renderRange(range, valuation.price_eur, listings, fresh);
  renderWaterfall(valuation, body, fresh);
  renderListings(comparables, body, valuation.price_eur);

  const model = valuation.model;
  $("estimate-foot").textContent =
    `Model ${model.variant} · version ${model.version.slice(0, 8)} · request ${valuation.request_id}`;
  if (fresh) {
    $("announcer").textContent = `Estimated asking price ${euro.format(valuation.price_eur)}.`;
    revealResult();
  }
}

/** The scale of the range bar: the range and every comparable price, padded 4 % on each side. */
export function rangeScale(lower, upper, prices) {
  const low = Math.min(lower, ...prices);
  const high = Math.max(upper, ...prices);
  const pad = (high - low) * 0.04 || high * 0.04;
  return { min: Math.max(0, low - pad), max: high + pad };
}

/** Rows for circles that would overlap: each circle goes one row up from a neighbour closer than `gap` px. */
export function stackRows(xs, gap) {
  const rows = [];
  const placed = [];
  xs.forEach((x, index) => {
    let row = 0;
    while (placed.some((p) => p.row === row && Math.abs(p.x - x) < gap)) row += 1;
    placed.push({ x, row });
    rows[index] = row;
  });
  return rows;
}

function renderRange(range, price, listings, fresh) {
  const box = $("range");
  const error = $("range-error");
  const note = $("range-note");
  if (range.status === "rejected") {
    box.replaceChildren();
    note.textContent = "";
    $("range-level").textContent = "";
    error.textContent = "The range could not be loaded; the estimate above is not affected.";
    error.hidden = false;
    return;
  }
  error.hidden = true;
  const { lower_eur: lower, upper_eur: upper, confidence_level: level } = range.value;
  const prices = listings.map((listing) => listing.price_eur);
  const scale = rangeScale(lower, upper, prices);
  const width = Math.max(320, Math.round(box.clientWidth));
  const x = (value) => ((value - scale.min) / (scale.max - scale.min)) * width;
  const axisY = 36;
  // Circles closer than 10 px would touch (r 3.5 plus the stroke), so they stack.
  const rows = stackRows(prices.map(x), 10);
  const root = svg("svg", { width, height: 72, "aria-hidden": "true", class: "range-svg" });
  root.append(svg("line", { class: "range-axis", x1: 0, x2: width, y1: axisY + 0.5, y2: axisY + 0.5 }));
  const bandStart = x(lower);
  const bandEnd = x(upper);
  root.append(svg("rect", { class: "range-band", x: bandStart, y: 31, width: Math.max(1, bandEnd - bandStart), height: 11 }));
  root.append(svg("line", { class: "range-end", x1: bandStart + 0.5, x2: bandStart + 0.5, y1: 31, y2: 42 }));
  root.append(svg("line", { class: "range-end", x1: bandEnd - 0.5, x2: bandEnd - 0.5, y1: 31, y2: 42 }));
  prices.forEach((value, index) => {
    root.append(svg("circle", {
      class: "range-listing", "data-listing": index, cx: x(value), cy: axisY + 0.5 - rows[index] * 8, r: 3.5,
    }));
  });
  // The marker and its value, which slide from where they were (the band centre at first).
  const target = x(price);
  const marker = svg("g", { class: "range-marker" });
  marker.append(svg("rect", { x: -1, y: 22, width: 2, height: 27 }));
  const label = svg("text", { class: "range-value", x: 0, y: 14, "text-anchor": "middle" }, euro.format(price));
  marker.append(label);
  root.append(marker);
  if (bandEnd - bandStart >= 140) {
    root.append(svg("text", { class: "range-bound", x: bandStart, y: 64, "text-anchor": "start" }, euro.format(lower)));
    root.append(svg("text", { class: "range-bound", x: bandEnd, y: 64, "text-anchor": "end" }, euro.format(upper)));
  } else {
    root.append(svg("text", { class: "range-bound", x: (bandStart + bandEnd) / 2, y: 64, "text-anchor": "middle" },
      `${euro.format(lower)} to ${euro.format(upper)}`));
  }
  box.replaceChildren(root);
  // Keep the value inside the column.
  const half = label.getComputedTextLength() / 2 || 32;
  const clamp = Math.min(Math.max(target, half), width - half) - target;
  label.setAttribute("x", String(clamp));
  const from = fresh ? (lastMarkerX ?? (bandStart + bandEnd) / 2) : target;
  marker.style.transform = `translateX(${from}px)`;
  if (from !== target) {
    requestAnimationFrame(() => requestAnimationFrame(() => {
      marker.classList.add("is-moving");
      marker.style.transform = `translateX(${target}px)`;
    }));
  }
  lastMarkerX = target;

  $("range-level").textContent = ` · ${Math.round(level * 100)}% prediction interval`;
  const share = Math.round(level * 10);
  note.textContent =
    `${share} in 10 cars like this are listed between ${euro.format(lower)} and ${euro.format(upper)}.` +
    (prices.length ? ` The circles are the ${prices.length === 1 ? "similar listing" : `${prices.length} similar listings`} below.` : "");
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

/** A text clipped with an ellipsis to `width` px, as an SVG text measures it. */
function fitText(node, width) {
  if (node.getComputedTextLength() <= width) return;
  const text = node.textContent;
  let low = 0;
  let high = text.length;
  while (low < high) {
    const middle = Math.ceil((low + high) / 2);
    node.textContent = `${text.slice(0, middle)}…`;
    if (node.getComputedTextLength() <= width) low = middle; else high = middle - 1;
  }
  node.textContent = `${text.slice(0, low)}…`;
}

function renderWaterfall(valuation, body, fresh) {
  const container = $("waterfall");
  const { base, steps, total } = waterfallSteps(valuation.explanation);
  const width = Math.max(480, Math.round(container.clientWidth));
  const labelWidth = width >= 760 ? 208 : 168;
  const euroWidth = 88;
  const pctWidth = 64;
  const plotLeft = labelWidth + 16;
  const plotRight = width - euroWidth - 16 - pctWidth - 16;
  const rowHeight = 32;
  const barHeight = 14;
  const rows = [
    { kind: "base", label: "Average listing", value: "", from: 0, to: base },
    ...steps.map((step) => ({
      kind: step.to >= step.from ? "up" : "down",
      label: step.field ? FIELD_LABELS[step.field] ?? step.field : "All other details",
      value: step.field ? fieldValue(step.field, body) : "",
      from: step.from, to: step.to, pct: step.pct,
    })),
    { kind: "total", label: "Estimate", value: "", from: 0, to: total },
  ];
  const maxValue = Math.max(...rows.map((row) => Math.max(row.from, row.to)));
  const domainMax = Math.ceil(maxValue / 10000) * 10000;
  const tick = domainMax < 20000 ? 5000 : 10000;
  const x = (value) => plotLeft + (value / domainMax) * (plotRight - plotLeft);
  const plotBottom = rows.length * rowHeight;
  const height = plotBottom + 28;

  const root = svg("svg", { width, height, "aria-hidden": "true", class: fresh ? "waterfall-svg is-fresh" : "waterfall-svg" });
  for (let t = 0; t <= domainMax + 1e-6; t += tick) {
    const tx = Math.round(x(t)) + 0.5;
    root.append(svg("line", { class: "wf-grid", x1: tx, x2: tx, y1: 0, y2: plotBottom }));
    root.append(svg("text", { class: "wf-tick", x: tx, y: plotBottom + 20, "text-anchor": "middle" }, t === 0 ? "€0" : `€${t / 1000}k`));
  }
  const labels = [];
  rows.forEach((row, index) => {
    const top = index * rowHeight;
    const middle = top + rowHeight / 2;
    const barY = middle - barHeight / 2;
    if (row.kind === "total") {
      root.append(svg("line", { class: "wf-rule", x1: 0, x2: width, y1: top + 0.5, y2: top + 0.5 }));
    }
    const text = svg("text", { class: row.kind === "total" ? "wf-label is-total" : "wf-label", x: 0, y: middle, "dominant-baseline": "central" });
    text.append(svg("tspan", {}, row.label));
    if (row.value) text.append(svg("tspan", { class: "wf-value", dx: 6 }, row.value));
    root.append(text);
    labels.push(text);
    const start = Math.min(row.from, row.to);
    const left = x(start);
    const barWidth = Math.max(1, Math.abs(x(row.to) - x(row.from)));
    const inset = row.kind === "down" ? 0.5 : 0;
    const bar = svg("rect", {
      class: `wf-bar wf-bar-${row.kind}`,
      x: left + inset, y: barY + inset, width: Math.max(1, barWidth - 2 * inset), height: barHeight - 2 * inset,
    });
    // Grows from its start edge: the running total before it, which for a lowering effect is its right edge.
    bar.style.transformOrigin = `${row.kind === "down" ? left + barWidth : left}px ${middle}px`;
    bar.style.animationDelay = `${index * 40}ms`;
    root.append(bar);
    if (index < rows.length - 2) {
      const end = Math.round(x(row.to)) + 0.5;
      root.append(svg("line", { class: "wf-link", x1: end, x2: end, y1: barY + barHeight, y2: barY + rowHeight }));
    }
    if (row.pct !== undefined) {
      root.append(svg("text", { class: "wf-pct", x: width - euroWidth - 16, y: middle, "text-anchor": "end", "dominant-baseline": "central" }, signedPercent(row.pct)));
    }
    const amount = row.kind === "up" || row.kind === "down" ? signedEuro(row.to - row.from) : euro.format(row.to);
    root.append(svg("text", {
      class: row.kind === "total" ? "wf-euro is-total" : "wf-euro", x: width, y: middle,
      "text-anchor": "end", "dominant-baseline": "central",
    }, amount));
  });
  const table = el("table", { class: "visually-hidden" },
    el("caption", {}, "What moved the estimate, from the average listing to the estimate"),
    el("thead", {}, el("tr", {},
      el("th", { scope: "col" }, "Detail"), el("th", { scope: "col" }, "Value"),
      el("th", { scope: "col" }, "Effect"), el("th", { scope: "col" }, "Price after it"))),
    el("tbody", {}, ...rows.map((row) => el("tr", {},
      el("th", { scope: "row" }, row.label),
      el("td", {}, row.value || "–"),
      el("td", {}, row.pct === undefined ? "–" : signedPercent(row.pct)),
      el("td", {}, euro.format(row.to)),
    ))),
  );
  container.replaceChildren(root, table);
  for (const label of labels) fitText(label, labelWidth);
}

function renderListings(comparables, body, price) {
  const wrap = $("listings-wrap");
  const empty = $("listings-empty");
  const error = $("listings-error");
  const aside = $("listings-aside");
  empty.hidden = true;
  error.hidden = true;
  wrap.hidden = true;
  aside.textContent = "";
  if (comparables.status === "rejected") {
    error.textContent = "Similar listings could not be loaded; the estimate above is not affected.";
    error.hidden = false;
    return;
  }
  const { listings, index } = comparables.value;
  if (!listings.length) {
    empty.textContent =
      `None of the ${integer.format(index.listings)} listings in the data is a ${body.make} ${body.model} ` +
      "within two years of this car's age and 25% of its mileage.";
    empty.hidden = false;
    return;
  }
  aside.textContent = ` · ${listings.length} closest by mileage and age, of ${integer.format(index.listings)}`;
  $("listings-body").replaceChildren(...listings.map((listing, position) => {
    const name = listingName(listing);
    const [year, month] = (listing.registration_date ?? "").split("-");
    const registered = year ? `${MONTHS[Number(month) - 1]} ${year}` : "–";
    const power = listing.power_kw === null ? "–" : `${integer.format(listing.power_kw)} kW`;
    // In a narrow column the seller drops out and the country moves to the front as its code, so it stays visible.
    const meta = [
      el("span", {}, [listing.fuel_category, listing.body_type].filter(Boolean).join(" · ")),
      listing.seller_type ? el("span", { class: "meta-wide" }, ` · ${labelOf("seller_type", listing.seller_type)}`) : null,
      listing.country_code ? el("span", { class: "meta-wide" }, ` · ${COUNTRIES[listing.country_code] ?? listing.country_code}`) : null,
    ];
    const narrow = [registered, power, listing.country_code].filter(Boolean).join(" · ");
    const row = el("tr", { "data-listing": position },
      el("td", {},
        el("span", { class: "listing-name", title: name }, name),
        el("span", { class: "listing-meta" },
          el("span", { class: "listing-narrow" }, `${narrow} · `), ...meta)),
      el("td", { class: "col-registered" }, registered),
      el("td", { class: "num" }, listing.mileage_km_raw === null ? "–" : `${integer.format(listing.mileage_km_raw)} km`),
      el("td", { class: "num col-power" }, power),
      el("td", { class: "num listing-price" }, euro.format(listing.price_eur)),
      el("td", { class: "num listing-versus" }, signedPercent(((listing.price_eur - price) / price) * 100, 0)),
    );
    const circle = () => document.querySelector(`.range-listing[data-listing="${position}"]`);
    row.addEventListener("mouseenter", () => circle()?.classList.add("is-active"));
    row.addEventListener("mouseleave", () => circle()?.classList.remove("is-active"));
    return row;
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
    input.dispatchEvent(new Event("input"));
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
  // The charts are drawn at the column's pixel width, so a new width draws them again.
  let resizeFrame = 0;
  let drawnWidth = 0;
  new ResizeObserver(([entry]) => {
    const width = Math.round(entry.contentRect.width);
    if (width === drawnWidth) return;
    drawnWidth = width;
    cancelAnimationFrame(resizeFrame);
    resizeFrame = requestAnimationFrame(() => { if (lastResult) render(lastResult, false); });
  }).observe($("result"));
  loadService();
}
