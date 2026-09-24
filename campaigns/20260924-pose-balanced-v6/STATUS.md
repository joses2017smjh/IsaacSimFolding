# Pose-balanced supervision v6 — live status

<!-- driver:status:begin -->
| | |
|---|---|
| **Active job** | `21408329` search.collect |
| **Current result** | targeted search pending |
| **Limitation / blocker** | none |
| **Next automatic action** | driver advances the next stage when a waited job ends |

_Updated 2026-09-24T16:11:28Z by scripts/driver.py._
<!-- driver:status:end -->

## Record

append-only; newest last

### 2026-09-24 — opened

The per-pose transfer analysis over v4/v5 artifacts shows a2's change tracks
per-pose supervision density (README). v6 supplies dense validated
supervision at P_B and P_A, gated per pose, and trains a2's recipe unchanged
on a pose-balanced corpus; evaluation is v5's matched protocol verbatim.
