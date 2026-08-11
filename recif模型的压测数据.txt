recif 模型压测结果分析
一、目的
对 recif 模型在两种配置下的压测结果进行整理和分析（有 custom 优化 vs 没有 custom 优化），给出结论与优化建议。原始日志附在文档末尾。

二、数据来源
文件：recif模型的压测数据.txt
仓库路径：/ZhiYunpenghuawei/p1/blob/main/recif%E6%A8%A1%E5%9E%8B%E7%9A%84%E5%8E%8B%E6%B5%8B%E6%95%B0%E6%8D%AE.txt
（原始文本已完整保留在文末）
三、关键指标汇总
启用 custom 优化（第一段）
Model load time : 33103.0 ms
Avg latency : 38.70 ms
Min latency : 34.59 ms
Max latency : 46.37 ms
P50 latency : 38.11 ms
P95 latency : 45.52 ms
P99 latency : 46.37 ms
Std dev : 3.04 ms
未启用 custom 优化（第二段）
Model load time : 63331.8 ms
Avg latency : 348.95 ms
Min latency : 248.76 ms
Max latency : 1060.37 ms
P50 latency : 343.82 ms
P95 latency : 359.03 ms
P99 latency : 1060.37 ms
Std dev : 110.98 ms
四、观察与说明
warmup 阶段（两次日志中）输出一致：top1_sid=5154174410510，logprob 大约 -2.95 — 表示模型预测结果在各次运行间稳定。
启用 custom 优化时：
平均延迟约 38.7 ms，延迟分布集中（Std dev = 3.04 ms），P99 仅 46.37 ms，说明延迟稳定且延迟非常低。
模型加载时间约 33.1 s。
未启用 custom 优化时：
平均延迟约 349 ms，延迟波动大（Std dev = 110.98 ms），存在明显异常点（例如 iter 7 = 1060.37 ms，iter 41 = 517.91 ms），导致 P99 飙高到 1060 ms。
模型加载时间约 63.3 s，明显比启用优化时更长。
性能对比（粗略）：
平均延迟：未优化 / 优化 ≈ 348.95 / 38.70 ≈ 9.0×（约 9 倍）
模型加载时间约减少 ~1.9×（63s → 33s）
稳定性：启用优化后 Std dev 大幅下降，异常延迟（spike）消失。
五、结论
启用 custom 优化能显著提升延迟性能并提高稳定性：平均延迟下降约 9 倍，P99 从 ~1s 降到 ~46 ms，标准差显著降低。
未启用优化时存在少数严重的延迟峰值，需排查导致峰值的根因（例如加载、内存交换、GC、线程调度或资源竞争等）。
warmup 的提示（"consider extending warmup to cover this shape/config."）建议采纳：更充分的 warmup 能减少首次或特殊输入形状导致的延迟波动。
consider extending warmup to cover this shape/config.
  warmup 1: top1_sid=5154174410510, logprob=-2.9549
  warmup 2: top1_sid=5154174410510, logprob=-2.9549
  warmup 3: top1_sid=5154174410510, logprob=-2.9549
  warmup 4: top1_sid=5154174410510, logprob=-2.9549
  warmup 5: top1_sid=5154174410510, logprob=-2.9549
[Benchmark] Running 50 benchmark iterations...
  iter   1/50: latency=  34.86 ms | top1_sid=       5154174410510 | logprob=-2.9549
  iter   2/50: latency=  35.02 ms | top1_sid=       5154174410510 | logprob=-2.9549
  iter   3/50: latency=  36.18 ms | top1_sid=       5154174410510 | logprob=-2.9549
  iter   4/50: latency=  35.62 ms | top1_sid=       5154174410510 | logprob=-2.9549
  iter   5/50: latency=  36.66 ms | top1_sid=       5154174410510 | logprob=-2.9549
  iter   6/50: latency=  36.38 ms | top1_sid=       5154174410510 | logprob=-2.9549
  iter   7/50: latency=  35.52 ms | top1_sid=       5154174410510 | logprob=-2.9549
  iter   8/50: latency=  37.18 ms | top1_sid=       5154174410510 | logprob=-2.9549
  iter   9/50: latency=  35.78 ms | top1_sid=       5154174410510 | logprob=-2.9549
  iter  10/50: latency=  35.68 ms | top1_sid=       5154174410510 | logprob=-2.9549
  iter  11/50: latency=  37.68 ms | top1_sid=       5154174410510 | logprob=-2.9549
  iter  12/50: latency=  35.26 ms | top1_sid=       5154174410510 | logprob=-2.9549
  iter  13/50: latency=  35.87 ms | top1_sid=       5154174410510 | logprob=-2.9549
  iter  14/50: latency=  36.89 ms | top1_sid=       5154174410510 | logprob=-2.9549
  iter  15/50: latency=  34.59 ms | top1_sid=       5154174410510 | logprob=-2.9549
  iter  16/50: latency=  36.42 ms | top1_sid=       5154174410510 | logprob=-2.9549
  iter  17/50: latency=  36.99 ms | top1_sid=       5154174410510 | logprob=-2.9549
  iter  18/50: latency=  39.28 ms | top1_sid=       5154174410510 | logprob=-2.9549
  iter  19/50: latency=  36.81 ms | top1_sid=       5154174410510 | logprob=-2.9549
  iter  20/50: latency=  37.90 ms | top1_sid=       5154174410510 | logprob=-2.9549
  iter  21/50: latency=  46.20 ms | top1_sid=       5154174410510 | logprob=-2.9549
  iter  22/50: latency=  37.46 ms | top1_sid=       5154174410510 | logprob=-2.9549
  iter  23/50: latency=  36.45 ms | top1_sid=       5154174410510 | logprob=-2.9549
  iter  24/50: latency=  39.56 ms | top1_sid=       5154174410510 | logprob=-2.9549
  iter  25/50: latency=  45.52 ms | top1_sid=       5154174410510 | logprob=-2.9549
  iter  26/50: latency=  43.45 ms | top1_sid=       5154174410510 | logprob=-2.9549
  iter  27/50: latency=  44.90 ms | top1_sid=       5154174410510 | logprob=-2.9549
  iter  28/50: latency=  40.37 ms | top1_sid=       5154174410510 | logprob=-2.9549
  iter  29/50: latency=  39.53 ms | top1_sid=       5154174410510 | logprob=-2.9549
  iter  30/50: latency=  39.77 ms | top1_sid=       5154174410510 | logprob=-2.9549
  iter  31/50: latency=  38.11 ms | top1_sid=       5154174410510 | logprob=-2.9549
  iter  32/50: latency=  38.11 ms | top1_sid=       5154174410510 | logprob=-2.9549
  iter  33/50: latency=  39.21 ms | top1_sid=       5154174410510 | logprob=-2.9549
  iter  34/50: latency=  39.68 ms | top1_sid=       5154174410510 | logprob=-2.9549
  iter  35/50: latency=  39.43 ms | top1_sid=       5154174410510 | logprob=-2.9549
  iter  36/50: latency=  39.41 ms | top1_sid=       5154174410510 | logprob=-2.9549
  iter  37/50: latency=  38.98 ms | top1_sid=       5154174410510 | logprob=-2.9549
  iter  38/50: latency=  42.52 ms | top1_sid=       5154174410510 | logprob=-2.9549
  iter  39/50: latency=  43.55 ms | top1_sid=       5154174410510 | logprob=-2.9549
  iter  40/50: latency=  40.70 ms | top1_sid=       5154174410510 | logprob=-2.9549
  iter  41/50: latency=  38.97 ms | top1_sid=       5154174410510 | logprob=-2.9549
  iter  42/50: latency=  37.70 ms | top1_sid=       5154174410510 | logprob=-2.9549
  iter  43/50: latency=  42.20 ms | top1_sid=       5154174410510 | logprob=-2.9549
  iter  44/50: latency=  35.65 ms | top1_sid=       5154174410510 | logprob=-2.9549
  iter  45/50: latency=  37.48 ms | top1_sid=       5154174410510 | logprob=-2.9549
  iter  46/50: latency=  46.37 ms | top1_sid=       5154174410510 | logprob=-2.9549
  iter  47/50: latency=  38.73 ms | top1_sid=       5154174410510 | logprob=-2.9549
  iter  48/50: latency=  37.93 ms | top1_sid=       5154174410510 | logprob=-2.9549
  iter  49/50: latency=  38.51 ms | top1_sid=       5154174410510 | logprob=-2.9549
  iter  50/50: latency=  42.24 ms | top1_sid=       5154174410510 | logprob=-2.9549

============================================================
[Benchmark] Results Summary
============================================================
  Model load time :    33103.0 ms
  Avg latency     :      38.70 ms
  Min latency     :      34.59 ms
  Max latency     :      46.37 ms
  P50 latency     :      38.11 ms
  P95 latency     :      45.52 ms
  P99 latency     :      46.37 ms
  Std dev         :       3.04 ms
============================================================
============================================================
[Benchmark] Results Summary
============================================================
  Model load time :    32848.2 ms
  Avg latency     :      36.46 ms
  Min latency     :      34.03 ms
  Max latency     :      46.30 ms
  P50 latency     :      35.74 ms
  P95 latency     :      41.55 ms
  P99 latency     :      46.30 ms
  Std dev         :       2.42 ms
============================================================

beam=128
============================================================
[Benchmark] Results Summary
============================================================
  Model load time :    73279.7 ms
  Avg latency     :      33.62 ms
  Min latency     :      31.63 ms
  Max latency     :      41.57 ms
  P50 latency     :      33.21 ms
  P95 latency     :      35.98 ms
  P99 latency     :      41.57 ms
  Std dev         :       1.63 ms
============================================================
============================================================
[Benchmark] Results Summary
============================================================
  Model load time :    42204.5 ms
  Avg latency     :      34.26 ms
  Min latency     :      31.55 ms
  Max latency     :      43.45 ms
  P50 latency     :      33.34 ms
  P95 latency     :      42.80 ms
  P99 latency     :      43.45 ms
  Std dev         :       2.71 ms
============================================================

beam=512
============================================================
[Benchmark] Results Summary
============================================================
  Model load time :    81696.5 ms
  Avg latency     :      45.47 ms
  Min latency     :      33.35 ms
  Max latency     :     395.78 ms
  P50 latency     :      36.71 ms
  P95 latency     :      50.98 ms
  P99 latency     :     395.78 ms
  Std dev         :      50.78 ms
============================================================


================================================================================
[Benchmark] Comparison Table
================================================================================
  Beam |  Samples/s |   Tokens/s |  PeakMem(MB) |   Avg(ms) |   P95(ms) |   P99(ms)
--------------------------------------------------------------------------------
    32 |      31.36 |      94.08 |          0.0 |     31.88 |     41.74 |     52.37
    64 |      32.20 |      96.59 |          0.0 |     31.05 |     32.68 |     33.62
   128 |      31.06 |      93.19 |          0.0 |     32.18 |     33.93 |     36.70
   256 |      30.22 |      90.67 |          0.0 |     33.08 |     35.51 |     41.11
   512 |      17.20 |      51.59 |          0.0 |     58.14 |     64.69 |    524.36
================================================================================
================================================================================
[Benchmark] Comparison Table
================================================================================
  Beam |  Samples/s |   Tokens/s |  PeakMem(MB) |   Avg(ms) |   P95(ms) |   P99(ms)
--------------------------------------------------------------------------------
    32 |      32.38 |      97.15 |          0.0 |     30.87 |     35.17 |     40.57
    64 |      30.81 |      92.43 |          0.0 |     32.45 |     34.59 |     36.10
   128 |      29.26 |      87.79 |          0.0 |     34.16 |     37.91 |     42.03
   256 |      27.57 |      82.72 |          0.0 |     36.25 |     43.41 |     48.36
   512 |      15.82 |      47.46 |          0.0 |     63.20 |     65.03 |    497.13
================================================================================

没有开custom优化：
================================================================================
[Benchmark] Comparison Table
================================================================================
  Beam |  Samples/s |   Tokens/s |  PeakMem(MB) |   Avg(ms) |   P95(ms) |   P99(ms)
--------------------------------------------------------------------------------
    32 |       4.93 |      14.79 |          0.0 |    202.77 |    213.42 |    216.66
    64 |       3.04 |       9.11 |          0.0 |    329.26 |    352.54 |    352.90
   128 |       3.49 |      10.47 |          0.0 |    286.49 |    344.78 |    348.87
   256 |       3.31 |       9.92 |          0.0 |    302.39 |    348.62 |    726.73
   512 |       3.35 |      10.05 |          0.0 |    298.60 |    347.21 |    669.50
================================================================================
consider extending warmup to cover this shape/config.
  warmup 1: top1_sid=5154174410510, logprob=-2.9562
  warmup 2: top1_sid=5154174410510, logprob=-2.9562
  warmup 3: top1_sid=5154174410510, logprob=-2.9572
  warmup 4: top1_sid=5154174410510, logprob=-2.9562
  warmup 5: top1_sid=5154174410510, logprob=-2.9562
[Benchmark] Running 50 benchmark iterations...
  iter   1/50: latency= 248.76 ms | top1_sid=       5154174410510 | logprob=-2.9573
  iter   2/50: latency= 253.62 ms | top1_sid=       5154174410510 | logprob=-2.9562
  iter   3/50: latency= 253.98 ms | top1_sid=       5154174410510 | logprob=-2.9562
  iter   4/50: latency= 254.74 ms | top1_sid=       5154174410510 | logprob=-2.9572
  iter   5/50: latency= 257.39 ms | top1_sid=       5154174410510 | logprob=-2.9562
  iter   6/50: latency= 254.79 ms | top1_sid=       5154174410510 | logprob=-2.9562
  iter   7/50: latency=1060.37 ms | top1_sid=       5154174410510 | logprob=-2.9572
  iter   8/50: latency= 275.28 ms | top1_sid=       5154174410510 | logprob=-2.9562
  iter   9/50: latency= 279.01 ms | top1_sid=       5154174410510 | logprob=-2.9562
  iter  10/50: latency= 312.04 ms | top1_sid=       5154174410510 | logprob=-2.9572
  iter  11/50: latency= 333.48 ms | top1_sid=       5154174410510 | logprob=-2.9572
  iter  12/50: latency= 352.39 ms | top1_sid=       5154174410510 | logprob=-2.9573
  iter  13/50: latency= 339.19 ms | top1_sid=       5154174410510 | logprob=-2.9562
  iter  14/50: latency= 343.57 ms | top1_sid=       5154174410510 | logprob=-2.9573
  iter  15/50: latency= 340.70 ms | top1_sid=       5154174410510 | logprob=-2.9573
  iter  16/50: latency= 351.22 ms | top1_sid=       5154174410510 | logprob=-2.9562
  iter  17/50: latency= 340.21 ms | top1_sid=       5154174410510 | logprob=-2.9573
  iter  18/50: latency= 359.03 ms | top1_sid=       5154174410510 | logprob=-2.9573
  iter  19/50: latency= 348.70 ms | top1_sid=       5154174410510 | logprob=-2.9562
  iter  20/50: latency= 344.79 ms | top1_sid=       5154174410510 | logprob=-2.9562
  iter  21/50: latency= 347.41 ms | top1_sid=       5154174410510 | logprob=-2.9572
  iter  22/50: latency= 356.27 ms | top1_sid=       5154174410510 | logprob=-2.9572
  iter  23/50: latency= 345.86 ms | top1_sid=       5154174410510 | logprob=-2.9573
  iter  24/50: latency= 341.77 ms | top1_sid=       5154174410510 | logprob=-2.9562
  iter  25/50: latency= 343.82 ms | top1_sid=       5154174410510 | logprob=-2.9572
  iter  26/50: latency= 343.26 ms | top1_sid=       5154174410510 | logprob=-2.9562
  iter  27/50: latency= 339.55 ms | top1_sid=       5154174410510 | logprob=-2.9572
  iter  28/50: latency= 322.89 ms | top1_sid=       5154174410510 | logprob=-2.9572
  iter  29/50: latency= 351.35 ms | top1_sid=       5154174410510 | logprob=-2.9562
  iter  30/50: latency= 355.00 ms | top1_sid=       5154174410510 | logprob=-2.9562
  iter  31/50: latency= 340.17 ms | top1_sid=       5154174410510 | logprob=-2.9562
  iter  32/50: latency= 339.39 ms | top1_sid=       5154174410510 | logprob=-2.9562
  iter  33/50: latency= 346.15 ms | top1_sid=       5154174410510 | logprob=-2.9562
  iter  34/50: latency= 343.82 ms | top1_sid=       5154174410510 | logprob=-2.9562
  iter  35/50: latency= 343.52 ms | top1_sid=       5154174410510 | logprob=-2.9562
  iter  36/50: latency= 353.69 ms | top1_sid=       5154174410510 | logprob=-2.9562
  iter  37/50: latency= 350.46 ms | top1_sid=       5154174410510 | logprob=-2.9572
  iter  38/50: latency= 334.96 ms | top1_sid=       5154174410510 | logprob=-2.9562
  iter  39/50: latency= 352.00 ms | top1_sid=       5154174410510 | logprob=-2.9562
  iter  40/50: latency= 357.37 ms | top1_sid=       5154174410510 | logprob=-2.9562
  iter  41/50: latency= 517.91 ms | top1_sid=       5154174410510 | logprob=-2.9573
  iter  42/50: latency= 351.67 ms | top1_sid=       5154174410510 | logprob=-2.9562
  iter  43/50: latency= 347.62 ms | top1_sid=       5154174410510 | logprob=-2.9562
  iter  44/50: latency= 331.16 ms | top1_sid=       5154174410510 | logprob=-2.9562
  iter  45/50: latency= 356.69 ms | top1_sid=       5154174410510 | logprob=-2.9562
  iter  46/50: latency= 352.05 ms | top1_sid=       5154174410510 | logprob=-2.9562
  iter  47/50: latency= 345.71 ms | top1_sid=       5154174410510 | logprob=-2.9572
  iter  48/50: latency= 331.43 ms | top1_sid=       5154174410510 | logprob=-2.9573
  iter  49/50: latency= 348.53 ms | top1_sid=       5154174410510 | logprob=-2.9573
  iter  50/50: latency= 352.56 ms | top1_sid=       5154174410510 | logprob=-2.9573

============================================================
[Benchmark] Results Summary
============================================================
  Model load time :    63331.8 ms
  Avg latency     :     348.95 ms
  Min latency     :     248.76 ms
  Max latency     :    1060.37 ms
  P50 latency     :     343.82 ms
  P95 latency     :     359.03 ms
  P99 latency     :    1060.37 ms
  Std dev         :     110.98 ms
============================================================
