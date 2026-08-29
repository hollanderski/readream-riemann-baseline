# DREAM database screen: experience vs no-experience as a cross-subject classification task
# source: DREAM DATABASE/{Datasets.csv, Data records.csv}, latest amendment per row
# 20 deposits after dedup (23 raw rows), 2669 awakening records (3191 raw)

deposit                            n_subj n_awak   exp norec noexp  stage              ch      Hz acc
----------------------------------------------------------------------------------------------------------------------
Multiple awakenings                   19    456   314     0   142  N2/REM             32     250 Open
DATA1                                 10    324   107    63   152  N3/NREM3/NREM4/N2     25    2000 Open
Zhang & Wamsley 2019                  28    308   238     0    70  N2/W               58     400 Open
Tononi Serial Awakenings              39    287   124    56   107  N2                257     500 Open
Oudiette_N1Data                       63    246   214    14    18  W/N2                3     256 Open
LODE                                  28    190   118    14    58  N2/REM              7     250 Open
Aamodt_evening_sleep                  27    158   116    27    15  N2/N3/NREM3/NREM4     62    1000 Open
REM_Turku                             18    134   123     8     3  REM                24     500 Open
Aamodt_morning_sleep                  16     97    60    21    14  N2/N1              62    1000 Open
SCANDataset                           18     85    63     7    15  REM/N3/NREM3/NREM4      4     200 Private
Kumral et al., 2023                   19     66    42     0    24  N2/N3/NREM3/NREM4    128    1000 Open
Dream_YoungAdults                     65     65    40     0    25  N2/REM             21     128 Open
Noreika_Motor_tDCS                    10     49    49     0     0  REM                16     500 Open
Brain Institute - Federal Univers     41     41    25     0    16  REM/N2             32    1000 Open
Older adults                          40     40    18     0    22  REM/N2             19     256 Open
TWC_USA                               19     33    27     1     5  REM/N2             21    1000 Open
MEG Kyushu                             1     31    24     4     0  N1/W                3    1000 Open
ChildrenDreaming                      30     30    13     0    17  N2/REM             32     250 Open
Sleep Talking                         12     22    11     1    10  N2/REM             32     250 Open
Dream Database from Donders            5      7     7     0     0  N2/REM             59     500 Open

## 1. Pooled
subjects 508 (sum over deposits, not unique people), awakenings 2669
binary experience vs no-experience: 1733 vs 713, balance 70.9% / 29.1% (n=2446, drops 216 without-recall)

## 2. Deposits that can carry a LOSO claim alone (>=10 subj, >=20 per class, Open)
deposit                            n_subj n_awak   exp noexp     ch
Dream_YoungAdults                     65     65    40    25     21
Tononi Serial Awakenings              39    287   124   107    257
Zhang & Wamsley 2019                  28    308   238    70     58
LODE                                  28    190   118    58      7
Multiple awakenings                   19    456   314   142     32
Kumral et al., 2023                   19     66    42    24    128
DATA1                                 10    324   107   152     25

7 deposits qualify, 208 subjects, 1696 awakenings

## 3. Channel counts across the qualifying subset
per-deposit median channel counts: [7, 21, 25, 32, 58, 128, 257]
minimum common count = 7
requiring >=19 ch drops 1 deposit(s): LODE

CAVEAT: the CSVs give channel COUNTS, not channel NAMES, so a true montage
intersection cannot be computed from metadata alone. It needs the deposit files.
