---
name: fix-loop
description: 在 EmbodiedHarness 中运行 RoboDojo 的 ASPIRE 风格开发修复、冻结评估与技能独立验证，并使用 System 1 数值执行和 System 2 规划接力。
---

# RoboDojo fix-loop

蓝本：ASPIRE tabletop iterative-debugging；RoboDojo 接口参照 EmbodiedDualLoop。
这是项目内实验 skill；不自动授权新设备部署或模型付费调用。构建/修改代码时不要启动实验。
无 Docker。流程由 [campaign.py](../../../src/embodied_harness/campaign.py) 执行，不能用自然语言省略协议步骤。

运行前读 [部署说明](../../../docs/SERVER_DEBUG.zh-CN.md) 与 [重建验收清单](../../../docs/REBUILD_PLAN.zh-CN.md)。
模型角色读 [打包的角色契约](../../../src/embodied_harness/instructions/fix_loop.md)。

1. 从真实 RoboDojo 布局清单创建互斥 dev/unseen/heldout 分区，保留文件 hash、bank/ordinal。
   不复制 LIBERO seeds；不把 manifest、私有原生目录或评分入口交给 agent。
2. 核对 profile 的实际标定；运行真实 SAM3 快照检查，固定源码/权重指纹。fixture 不能充当 SAM3。
3. 从公开初始观测和当前 verified 库生成程序；每一版本跑全部 dev，按共同失败模式修复。
   System 1 仅选有界数值参数；System 2 管理有序子任务与开发修复。遵守公开 API 和动作限额。
4. 仅依据 dev 选定程序并记录 hash。选定后 unseen 每场景一次，heldout 固定执行，无 System 2 修复。
   失败保留为失败；中断/未知提交不自动重跑。native termination+success 才是原生成功。
5. verifier 获得公开目标/观测/库及技能提议，看不到提议者程序/轨迹；最多两次独立重构并在 dev 执行。
   只有实际成功且 provenance 通过的提议进入 promoted 库。保存 skill_report 的使用和效果证据。
6. 多任务使用 curriculum：按显式课程顺序传递 verified 库；最终库变化后，对低于阈值的任务只做一次
   rerun_02 独立条件，不覆盖 attempt_01，也不把两个条件合并算成功率。

使用 `embodied-harness campaign --config ...` 或 `curriculum --config ...`。
恢复使用同一配置/目录；如果 ledger 标为 failed/interrupted，保留证据并用新实验目录，不抹掉后重跑。
报告程序/库/SAM3版本、全分母、primary/recovered、干预次数、失败原因，并明确模型与物理是否真实运行。
