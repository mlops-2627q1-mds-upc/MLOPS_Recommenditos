// Draws the ```mermaid blocks of a page as diagrams (architecture.md).
// MkDocs renders such a block as <pre><code class="language-mermaid">, so each
// one is replaced by the element Mermaid draws into. Loaded as a module
// (mkdocs.yml), which runs once the page is parsed.
import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@12.2.0/dist/mermaid.esm.min.mjs";

for (const code of document.querySelectorAll("pre > code.language-mermaid")) {
  const diagram = document.createElement("div");
  diagram.className = "mermaid";
  diagram.textContent = code.textContent;
  code.parentElement.replaceWith(diagram);
}

mermaid.initialize({ startOnLoad: false });
await mermaid.run({ querySelector: "div.mermaid" });
