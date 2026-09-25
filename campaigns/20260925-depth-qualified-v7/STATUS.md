# Depth-qualified supervision v7 — live status

<!-- driver:status:begin -->
| | |
|---|---|
| **Active job** | `21423233` search.collect |
| **Current result** | smoke pass |
| **Limitation / blocker** | none |
| **Next automatic action** | driver advances the next stage when a waited job ends |

_Updated 2026-09-25T21:59:16Z by scripts/driver.py._
<!-- driver:status:end -->

## Record

append-only; newest last

### 2026-09-25 — opened

From the verified v6 diagnosis: the fine-tune reaches early but lands shallow.
v7 records landing telemetry in a fresh search, keeps only branches that
settled deep and early, gates on a preregistered check that landing depth
predicts settling, adds large-garment P_B rows, and trains v6's recipe
unchanged; evaluation is v5's matched protocol verbatim.
