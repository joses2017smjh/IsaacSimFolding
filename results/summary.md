# Rollout results

Every verdict is produced by LeHome's own `success_checker_garment_fold`, the same
function the challenge scores with. Nothing here is self-assessed.

Runs shorter than 250 steps are camera/plumbing tests, not fold attempts, and are excluded.

| garment | class | driver | episode | conditions | verdict |
|---|---|---|---|---|---|
| Pant_Long_Seen_0 | Pant_Long | policy | - | 2/5 | failure |
| Pant_Long_Seen_1 | Pant_Long | policy | - | 2/5 | failure |
| Pant_Short_Seen_0 | Pant_Short | policy | - | 3/5 | failure |
| Pant_Short_Seen_1 | Pant_Short | policy | - | 3/5 | failure |
| Top_Long_Seen_0 | Top_Long | policy | - | 2/5 | failure |
| Top_Long_Seen_1 | Top_Long | policy | - | 2/5 | failure |
| Top_Short_Seen_0 | Top_Short | policy | - | 2/5 | failure |
| Top_Short_Seen_1 | Top_Short | policy | - | 2/5 | failure |
| Pant_Long_Seen_0 | Pant_Long | replay | ep750 | 4/5 | **SUCCESS** |
| Pant_Long_Seen_0 | Pant_Long | replay | ep751 | 4/5 | failure |
| Pant_Long_Seen_0 | Pant_Long | replay | ep752 | 4/5 | **SUCCESS** |
| Pant_Short_Seen_0 | Pant_Short | replay | ep500 | 5/5 | **SUCCESS** |
| Pant_Short_Seen_0 | Pant_Short | replay | ep501 | 5/5 | **SUCCESS** |
| Pant_Short_Seen_0 | Pant_Short | replay | ep502 | 4/5 | failure |
| Top_Long_Seen_0 | Top_Long | replay | ep250 | 3/5 | failure |
| Top_Long_Seen_0 | Top_Long | replay | ep251 | 5/5 | **SUCCESS** |
| Top_Long_Seen_0 | Top_Long | replay | ep252 | 5/5 | **SUCCESS** |
| Top_Long_Seen_1 | Top_Long | replay | ep275 | 5/5 | **SUCCESS** |
| Top_Short_Seen_0 | Top_Short | replay | - | 4/5 | failure |
| Top_Short_Seen_0 | Top_Short | replay | ep1 | 4/5 | failure |
| Top_Short_Seen_0 | Top_Short | replay | ep2 | 5/5 | **SUCCESS** |
| Top_Short_Seen_0 | Top_Short | replay | ep3 | 3/5 | failure |
| Top_Short_Seen_0 | Top_Short | replay | ep4 | 5/5 | **SUCCESS** |
| Top_Short_Seen_0 | Top_Short | replay | ep5 | 5/5 | **SUCCESS** |
| Top_Short_Seen_0 | Top_Short | replay | ep6 | 5/5 | **SUCCESS** |
| Top_Short_Seen_1 | Top_Short | replay | ep25 | 5/5 | **SUCCESS** |
| Top_Short_Seen_1 | Top_Short | replay | ep26 | 5/5 | **SUCCESS** |
| Top_Short_Seen_1 | Top_Short | replay | ep27 | 5/5 | **SUCCESS** |
| Top_Short_Seen_1 | Top_Short | replay | ep28 | 5/5 | **SUCCESS** |

| driver | episodes | succeeded |
|---|---|---|
| demonstration replay | 21 | **15** |
| trained BC policy | 8 | **0** |

### Replay, by garment class

| class | episodes | succeeded |
|---|---|---|
| Top_Short | 11 | **8** |
| Top_Long | 4 | **3** |
| Pant_Short | 3 | **2** |
| Pant_Long | 3 | **2** |
