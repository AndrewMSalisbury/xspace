# Data audit

Generated 2026-10-01 by `scripts/data_audit.py` (summary statistics only; no raw data). Definitions are in [methodology.md](methodology.md).

How to read this:

- *Actor–ball* is the distance between the acting player and the tracking ball at the synced event frame. For PFF it is ~0 m by construction (the smoothed ball is pulled onto the player at tagged events), so it confirms timing, not ball accuracy.
- *Clock offset* is the per-period shift applied to event times. PFF frames are tagged with event ids, so its offsets are ~0; IDSSE's vary by match and half.
- IDSSE's lower *actor within 3 m* reflects per-event timing noise in DFL events (candidate fix: per-event refinement).
- Matches with **no tracking for periods 3–4** lose all extra-time frames; their extra-time events are kept but unsynced.

## IDSSE (7 matches)

|                         |    min |     5% |    50% |    95% |    max |
|:------------------------|-------:|-------:|-------:|-------:|-------:|
| Ball in play (min)      |  45.58 |  46.3  |  54.31 |  56.9  |  56.91 |
| Clean frames %          |  99.87 |  99.89 |  99.95 |  99.97 |  99.97 |
| Ball missing %          |   0    |   0    |   0    |   0    |   0    |
| On-ball events synced % |  97.16 |  97.24 |  97.98 |  98.42 |  98.43 |
| Actor–ball median (m)   |   1.39 |   1.43 |   1.59 |   2.14 |   2.23 |
| Actor within 3 m %      |  54.25 |  55.16 |  60.26 |  64.66 |  65.22 |
| Max |clock offset| (s)  |   0.12 |   0.25 |   1    |   1.27 |   1.32 |
| Open play %             |  82.83 |  83.12 |  86.41 |  87.84 |  88.16 |
| Transition %            |  27.24 |  27.32 |  27.9  |  35.01 |  35.88 |
| Possessions             | 212    | 214.1  | 227    | 254.3  | 257    |

Match notes:

- J03WN1 (VfL Bochum 1848 v Bayer 04 Leverkusen): 1 sent off
- J03WQQ (Fortuna Düsseldorf v FC St. Pauli): 1 sent off

## PFF (64 matches)

|                         |    min |     5% |    50% |    95% |    max |
|:------------------------|-------:|-------:|-------:|-------:|-------:|
| Ball in play (min)      |  50.08 |  53.31 |  59.96 |  71    |  79.98 |
| Clean frames %          |  88.27 |  90.01 |  93.78 |  96.09 |  97.14 |
| Ball missing %          |   2.55 |   3.47 |   5.91 |   9.74 |  11.21 |
| On-ball events synced % |  76.23 |  99.68 | 100    | 100    | 100    |
| Actor–ball median (m)   |   0    |   0    |   0    |   0    |   0    |
| Actor within 3 m %      |  89.17 |  91.1  |  94.17 |  96.04 |  96.38 |
| Max |clock offset| (s)  |   0    |   0    |   0.04 |   0.04 |   0.04 |
| Open play %             |  79.65 |  83.55 |  86.97 |  90.07 |  92.48 |
| Transition %            |  19.37 |  22.03 |  27.65 |  36.98 |  39.57 |
| Possessions             | 194    | 207.9  | 247.5  | 322    | 372    |

Match notes:

- 10507 (Brazil v South Korea): 1 GK change(s)
- 10510 (Croatia v Brazil): **no tracking for period(s) 3,4**
- 10511 (Netherlands v Argentina): 1 sent off; **no tracking for period(s) 3,4**
- 10512 (Morocco v Portugal): 1 sent off
- 3813 (England v Iran): 1 GK change(s)
- 3828 (Wales v Iran): 1 GK change(s); 1 sent off
- 3859 (Cameroon v Brazil): 1 sent off

## Per match

| source   | match_id   | home               | away                 |   in_play_min |   clean_pct |   synced_pct |   actor_within_3m_pct |   max_abs_offset_s |   open_play_pct |   home_possession_pct |
|:---------|:-----------|:-------------------|:---------------------|--------------:|------------:|-------------:|----------------------:|-------------------:|----------------:|----------------------:|
| idsse    | J03WMX     | 1. FC Köln         | FC Bayern München    |          56.9 |        99.9 |         98.4 |                  60.3 |                1   |            86.4 |                  46.6 |
| idsse    | J03WN1     | VfL Bochum 1848    | Bayer 04 Leverkusen  |          45.6 |        99.9 |         98.3 |                  65.2 |                0.6 |            82.8 |                  49.3 |
| idsse    | J03WOH     | Fortuna Düsseldorf | SSV Jahn Regensburg  |          48   |        99.9 |         98   |                  59   |                1.3 |            83.8 |                  52.9 |
| idsse    | J03WOY     | Fortuna Düsseldorf | F.C. Hansa Rostock   |          50.9 |       100   |         98.4 |                  62.6 |                0.8 |            83.9 |                  46.3 |
| idsse    | J03WPY     | Fortuna Düsseldorf | 1. FC Nürnberg       |          54.3 |        99.9 |         97.5 |                  54.3 |                1.2 |            86.5 |                  65   |
| idsse    | J03WQQ     | Fortuna Düsseldorf | FC St. Pauli         |          56.2 |       100   |         97.2 |                  63.4 |                0.1 |            87.1 |                  43.4 |
| idsse    | J03WR9     | Fortuna Düsseldorf | 1. FC Kaiserslautern |          56.9 |       100   |         97.4 |                  57.3 |                1.1 |            88.2 |                  66.8 |
| pff      | 10502      | Netherlands        | United States        |          63.8 |        94.8 |        100   |                  94.5 |                0   |            90   |                  39.1 |
| pff      | 10503      | Argentina          | Australia            |          66.3 |        92.6 |        100   |                  93.3 |                0   |            88.1 |                  59.6 |
| pff      | 10504      | France             | Poland               |          64   |        93.9 |        100   |                  95.1 |                0   |            90.1 |                  52.4 |
| pff      | 10505      | England            | Senegal              |          59.5 |        95.8 |        100   |                  95.2 |                0   |            87.2 |                  59.2 |
| pff      | 10506      | Japan              | Croatia              |          80   |        92.7 |         99.7 |                  94.1 |                0   |            87.5 |                  42.2 |
| pff      | 10507      | Brazil             | South Korea          |          62.9 |        93.2 |        100   |                  94.2 |                0   |            88.9 |                  52.4 |
| pff      | 10508      | Morocco            | Spain                |          79.8 |        95.2 |         99.7 |                  95.6 |                0   |            86.5 |                  27.8 |
| pff      | 10509      | Portugal           | Switzerland          |          56.3 |        93.9 |        100   |                  92.9 |                0   |            85.7 |                  46.7 |
| pff      | 10510      | Croatia            | Brazil               |          67.1 |        90   |         78   |                  91   |                0   |            89   |                  49.5 |
| pff      | 10511      | Netherlands        | Argentina            |          61.1 |        91.7 |         76.2 |                  93.9 |                0   |            86.5 |                  58.7 |
| pff      | 10512      | Morocco            | Portugal             |          58.1 |        92.7 |        100   |                  92.8 |                0   |            86.2 |                  28.6 |
| pff      | 10513      | England            | France               |          61.1 |        90.8 |        100   |                  94.2 |                0   |            89.2 |                  57.6 |
| pff      | 10514      | Argentina          | Croatia              |          59.9 |        91.2 |        100   |                  93.8 |                0   |            88.8 |                  41.2 |
| pff      | 10515      | France             | Morocco              |          62.8 |        92.3 |        100   |                  94.5 |                0   |            88.4 |                  40.6 |
| pff      | 10516      | Croatia            | Morocco              |          59.9 |        92.4 |        100   |                  93.7 |                0   |            86.8 |                  50.8 |
| pff      | 10517      | Argentina          | France               |          72.5 |        92.3 |         99.6 |                  92.6 |                0   |            84.1 |                  51.6 |
| pff      | 3812       | Senegal            | Netherlands          |          56.6 |        93.3 |        100   |                  93.1 |                0   |            84   |                  46.7 |
| pff      | 3813       | England            | Iran                 |          56.9 |        97.1 |        100   |                  95.4 |                0   |            87.4 |                  75.3 |
| pff      | 3814       | Qatar              | Ecuador              |          53.3 |        96.1 |        100   |                  96.2 |                0   |            85.4 |                  49   |
| pff      | 3815       | United States      | Wales                |          60.3 |        95.4 |        100   |                  94.9 |                0   |            86.6 |                  58.6 |
| pff      | 3816       | Argentina          | Saudi Arabia         |          55.2 |        93.4 |        100   |                  93.8 |                0   |            84   |                  68.9 |
| pff      | 3817       | Denmark            | Tunisia              |          60   |        94.2 |        100   |                  93.5 |                0   |            86.7 |                  59.7 |
| pff      | 3818       | Mexico             | Poland               |          56.9 |        93.7 |        100   |                  93.4 |                0   |            84.7 |                  58.4 |
| pff      | 3819       | France             | Australia            |          67.5 |        94.6 |        100   |                  95   |                0   |            89.4 |                  60   |
| pff      | 3820       | Morocco            | Croatia              |          59.6 |        96   |        100   |                  96   |                0   |            87.2 |                  36.8 |
| pff      | 3821       | Germany            | Japan                |          60.6 |        94   |        100   |                  94.2 |                0   |            86.1 |                  72.2 |
| pff      | 3822       | Spain              | Costa Rica           |          68.5 |        95.6 |        100   |                  96.1 |                0   |            88.6 |                  77.6 |
| pff      | 3823       | Belgium            | Canada               |          60.5 |        94.4 |        100   |                  94.7 |                0   |            86.8 |                  50.2 |
| pff      | 3824       | Switzerland        | Cameroon             |          64   |        96.4 |        100   |                  95.9 |                0   |            88.7 |                  48.9 |
| pff      | 3825       | Uruguay            | South Korea          |          58.7 |        96   |        100   |                  95.3 |                0   |            88.1 |                  53.8 |
| pff      | 3826       | Portugal           | Ghana                |          59.2 |        95.3 |        100   |                  94.7 |                0   |            87.2 |                  61.5 |
| pff      | 3827       | Brazil             | Serbia               |          60.1 |        93.5 |        100   |                  94.4 |                0   |            88.5 |                  59.2 |
| pff      | 3828       | Wales              | Iran                 |          55.8 |        96   |        100   |                  94.2 |                0   |            86.2 |                  59.3 |
| pff      | 3829       | Qatar              | Senegal              |          59.8 |        93.8 |        100   |                  92.9 |                0   |            86.2 |                  46.3 |
| pff      | 3830       | Netherlands        | Ecuador              |          58.6 |        93.1 |        100   |                  93   |                0   |            84.9 |                  53.3 |
| pff      | 3831       | England            | United States        |          62.1 |        96.9 |        100   |                  95.8 |                0   |            88.9 |                  54.1 |
| pff      | 3832       | Tunisia            | Australia            |          57.8 |        90.1 |        100   |                  92.1 |                0   |            83.6 |                  57.5 |
| pff      | 3833       | Poland             | Saudi Arabia         |          53.3 |        93.3 |        100   |                  93.8 |                0   |            82.4 |                  37.7 |
| pff      | 3834       | France             | Denmark              |          62.9 |        93.8 |        100   |                  93.8 |                0   |            88.7 |                  49.6 |
| pff      | 3835       | Argentina          | Mexico               |          54.7 |        92.4 |        100   |                  91.5 |                0   |            84.1 |                  58.5 |
| pff      | 3836       | Japan              | Costa Rica           |          61.5 |        95.2 |        100   |                  95.2 |                0   |            87.7 |                  55.3 |
| pff      | 3837       | Belgium            | Morocco              |          58.8 |        95   |        100   |                  94.1 |                0   |            87   |                  63.7 |
| pff      | 3838       | Croatia            | Canada               |          61.7 |        93.4 |        100   |                  93.6 |                0   |            87.3 |                  47.3 |
| pff      | 3839       | Spain              | Germany              |          57.8 |        95.7 |        100   |                  96.4 |                0   |            86   |                  60.8 |
| pff      | 3840       | Cameroon           | Serbia               |          55.8 |        90.8 |        100   |                  91   |                0   |            87.4 |                  43.9 |
| pff      | 3841       | South Korea        | Ghana                |          55.9 |        96.1 |        100   |                  94.2 |                0   |            85.1 |                  61.4 |
| pff      | 3842       | Brazil             | Switzerland          |          61.9 |        94.6 |        100   |                  94   |                0   |            86.5 |                  55.3 |
| pff      | 3843       | Portugal           | Uruguay              |          58.4 |        92.6 |        100   |                  93.8 |                0   |            86.4 |                  59.1 |
| pff      | 3844       | Ecuador            | Senegal              |          50.5 |        89.1 |        100   |                  89.2 |                0   |            83.5 |                  58   |
| pff      | 3845       | Netherlands        | Qatar                |          67.8 |        93.6 |        100   |                  95.1 |                0   |            89.4 |                  58.1 |
| pff      | 3846       | Wales              | England              |          53.3 |        94.6 |        100   |                  95.3 |                0   |            87.2 |                  36   |
| pff      | 3847       | Iran               | United States        |          62.7 |        94   |        100   |                  94.2 |                0   |            88.1 |                  49.3 |
| pff      | 3848       | Australia          | Denmark              |          62   |        92.6 |        100   |                  93.4 |                0   |            86.9 |                  34.4 |
| pff      | 3849       | Tunisia            | France               |          58.7 |        93.3 |        100   |                  93.7 |                0   |            84.8 |                  36.9 |
| pff      | 3850       | Poland             | Argentina            |          68.4 |        95.2 |        100   |                  96   |                0   |            90.5 |                  28.5 |
| pff      | 3851       | Saudi Arabia       | Mexico               |          50.1 |        89.8 |        100   |                  90.2 |                0   |            79.7 |                  38.7 |
| pff      | 3852       | Croatia            | Belgium              |          70.2 |        94.7 |        100   |                  95.2 |                0   |            90.1 |                  50.5 |
| pff      | 3853       | Canada             | Morocco              |          54.7 |        95.5 |        100   |                  94.2 |                0   |            84.7 |                  58.5 |
| pff      | 3854       | Japan              | Spain                |          71.1 |        92.6 |        100   |                  94.7 |                0   |            92.5 |                  18.1 |
| pff      | 3855       | Costa Rica         | Germany              |          65   |        93.8 |        100   |                  94.2 |                0   |            88.5 |                  32.6 |
| pff      | 3856       | Ghana              | Uruguay              |          56   |        90.9 |        100   |                  92.7 |                0   |            83.3 |                  53.2 |
| pff      | 3857       | South Korea        | Portugal             |          61.2 |        93.8 |        100   |                  94.8 |                0   |            88.2 |                  41.1 |
| pff      | 3858       | Serbia             | Switzerland          |          57   |        92.1 |        100   |                  93.3 |                0   |            86   |                  52.9 |
| pff      | 3859       | Cameroon           | Brazil               |          58.8 |        88.3 |        100   |                  92.9 |                0   |            86.4 |                  41.6 |
