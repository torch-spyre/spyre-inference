# RFC figures

Diagram **sources** for `../upstream-connector-port.md`. The rendered `.svg`
outputs were removed because they were stale relative to these sources (they
still showed a superseded M2 design); regenerate them from the sources below.

| Source | Renders to | Used by |
|---|---|---|
| `spyre-offloading-arch.mmd` | `spyre-offloading-arch.svg` | §5 (overall architecture) |
| `spyre-shared-pool-m2.mmd` | `spyre-shared-pool-m2.svg` | §6.8 (M2 shared host pool) |

`.d2` variants (`*.d2` → `*.d2.svg`) are alternate sources for the same diagrams.

## Regenerate (needs a machine with a headless Chromium — e.g. a Mac, not the CI container)

```bash
# Mermaid → SVG
npx -y -p @mermaid-js/mermaid-cli@10 mmdc \
  -i spyre-offloading-arch.mmd -o spyre-offloading-arch.svg -b transparent
npx -y -p @mermaid-js/mermaid-cli@10 mmdc \
  -i spyre-shared-pool-m2.mmd -o spyre-shared-pool-m2.svg -b transparent

# D2 → SVG (optional alternate)
d2 spyre-offloading-arch.d2   spyre-offloading-arch.d2.svg
d2 spyre-shared-pool-m2.d2    spyre-shared-pool-m2.d2.svg
```
