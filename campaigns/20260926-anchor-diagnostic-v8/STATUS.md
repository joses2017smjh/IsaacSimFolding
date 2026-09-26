# Anchor diagnostic v8 — live status

<!-- driver:status:begin -->
| | |
|---|---|
| **Active job** | none submitted yet |
| **Current result** | none — campaign being frozen |
| **Limitation / blocker** | none |
| **Next automatic action** | verify the reused corpus, then train with the pants anchor |
<!-- driver:status:end -->

## Record

append-only; newest last

### 2026-09-26 — opened

From the verified v7 diagnosis: the H50 deficit (P_B) comes from an early-H50
shift every fine-tune shares; the byte-identical BC anchor (12/16 tops) is
the most concrete shared suspect. v8 changes only the anchor (16 pants),
reuses v7's corpus and recipe, judges the anchor hypothesis by preregistered
mechanistic endpoints first, and requires a confirmatory final-set gate for
any positive claim.
