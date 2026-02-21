Mermaid source files for SRS diagrams.

To render PNG/SVG images locally, install `@mermaid-js/mermaid-cli` (Node.js) and run:

```bash
# install once
npm install -g @mermaid-js/mermaid-cli

# render all .mmd files to PNG
mmdc -i usecase.mmd -o usecase.png
mmdc -i seq_fed.mmd -o seq_fed.png
mmdc -i seq_signal.mmd -o seq_signal.png
mmdc -i dfd.mmd -o dfd.png
mmdc -i class.mmd -o class.png
```

Alternatively, use the Mermaid Live Editor (https://mermaid.live/) to paste each `.mmd` file and export images.