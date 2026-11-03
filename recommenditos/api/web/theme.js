// Applies a theme chosen with the toggle before the page paints; without one,
// the page follows the system (prefers-color-scheme in app.css).
try {
  const theme = localStorage.getItem("theme");
  if (theme === "light" || theme === "dark") document.documentElement.dataset.theme = theme;
} catch {
  // Storage blocked: the system theme applies.
}
