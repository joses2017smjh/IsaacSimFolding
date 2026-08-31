# Rollout results

Every verdict is produced by LeHome's own `success_checker_garment_fold`,
the same function the challenge scores with. Nothing here is self-assessed.

| garment | driver | episode | conditions | verdict |
|---|---|---|---|---|
| Pant_Long_Seen_0 | policy | - | 2/5 | failure |
| Pant_Long_Seen_1 | policy | - | 2/5 | failure |
| Pant_Short_Seen_0 | policy | - | 3/5 | failure |
| Pant_Short_Seen_1 | policy | - | 3/5 | failure |
| Top_Long_Seen_0 | policy | - | 2/5 | failure |
| Top_Long_Seen_1 | policy | - | 2/5 | failure |
| Top_Short_Seen_0 | policy | - | 2/5 | failure |
| Top_Short_Seen_1 | policy | - | 2/5 | failure |
| Top_Short_Seen_0 | replay | - | 4/5 | failure |
| Top_Short_Seen_0 | replay | ep1 | 4/5 | failure |
| Top_Short_Seen_0 | replay | ep2 | 5/5 | **SUCCESS** |
| Top_Short_Seen_0 | replay | ep3 | 3/5 | failure |
| Top_Short_Seen_0 | replay | ep4 | 5/5 | **SUCCESS** |
| Top_Short_Seen_1 | replay | ep25 | 5/5 | **SUCCESS** |
| Top_Short_Seen_1 | replay | ep26 | 5/5 | **SUCCESS** |
| Top_Short_Seen_1 | replay | ep27 | 5/5 | **SUCCESS** |
| Top_Short_Seen_1 | replay | ep28 | 5/5 | **SUCCESS** |

- demonstration replay: **6/9 succeeded**
- trained BC policy:    **0/8 succeeded**
