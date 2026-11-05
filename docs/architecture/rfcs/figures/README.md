# RFC figures

These diagrams accompany `../upstream-connector-port.md`. The committed SVGs are rendered from
D2 with **v0.7.1**. Mermaid files express the same nodes and relationships for readers who prefer
that format; they are not a second source for the committed SVG output.

| D2 source | Committed output | Used by |
|---|---|---|
| `spyre-offloading-arch.d2` | `spyre-offloading-arch.svg` | §5, connector integration and transfer ownership |
| `spyre-shared-pool-m2.d2` | `spyre-shared-pool-m2.svg` | §6.8, target shared pools and persistent lifecycle |

## Regenerate

From this directory:

```bash
d2 --pad 50 spyre-offloading-arch.d2 spyre-offloading-arch.svg
d2 --pad 50 spyre-shared-pool-m2.d2 spyre-shared-pool-m2.svg
```

Keep D2 and Mermaid labels/edges aligned, render after changes, and inspect the generated layout.
The shared-pool diagram shows the target architecture, not a claim that the separate sharing and
secondary-memory prototypes already implement persistent shared secondary-memory serving.
