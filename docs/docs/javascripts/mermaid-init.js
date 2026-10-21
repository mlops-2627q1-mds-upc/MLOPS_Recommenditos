// Draws the ```mermaid blocks of a page as diagrams (architecture.md).
// MkDocs renders such a block as <pre><code class="language-mermaid">, so each
// one is replaced by the element Mermaid draws into. Loaded as a module
// (mkdocs.yml), which runs once the page is parsed; Mermaid itself is fetched
// only on a page that has a diagram.
const blocks = document.querySelectorAll("pre > code.language-mermaid");

if (blocks.length > 0) {
  const { default: mermaid } = await import(
    "https://cdn.jsdelivr.net/npm/mermaid@12.2.0/dist/mermaid.esm.min.mjs"
  );
  mermaid.initialize({ startOnLoad: false });
  // One at a time, each with its own ID: `mermaid.run` derives IDs from the
  // clock, and two diagrams drawn in the same millisecond shared one, so the
  // second was drawn over the first.
  let index = 0;
  for (const code of blocks) {
    const { svg } = await mermaid.render(`mermaid-diagram-${index++}`, code.textContent);
    const diagram = document.createElement("div");
    diagram.className = "mermaid";
    diagram.innerHTML = svg;
    code.parentElement.replaceWith(diagram);
  }
}
