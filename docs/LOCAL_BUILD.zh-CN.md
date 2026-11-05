# 本地构建与测试交付

2026-10-29：128项测试通过；Python wheel 构建成功，已检查包含运行时角色提示词、campaign/native 模块和
ASPIRE LICENSE/NOTICE。fix-loop skill 的官方 quick_validate 校验通过。
未启动真实 Isaac、SAM3 checkpoint、System 2 模型或 Jev；无 Docker。

## 完成模块

| 模块 | 已实现内容 |
| --- | --- |
| contracts/artifacts/geometry | immutable观测/动作/回执，快照与hash，同帧RGB-D，frame/TCP变换，点云与可见表面PCA |
| providers/services/perception | SAM3文本/点HTTP，严格原图mask/score/box，代码与权重pin、strict加载，候选消歧与深度质量 |
| API/programs | ToolRegistry/CapApi、公共工具schema、ASPIRE源码policy、逐调用provenance、真实持久Python进程 |
| System 1 | 数值scope/单位/范围验证、宿主测量状态、Jev/fixed、选值注入、实际动作幅度再仲裁 |
| System 2 | Plan/有序子任务、generate/review/repair接力，目标/前缀/版本/干预限制，native成功独立于review |
| RoboDojo | 独立native bootstrap、bank/ordinal布局清单、公开RGB-D、原生step_lim、单reset、故障锁定与生命周期 |
| campaign | 全dev版本矩阵、失败模式修复、仅dev选定、程序hash、一次unseen、heldout固定执行禁S2 |
| skills/curriculum | 使用效果skill_report、独立verifier两次重构、verified晋升、库冻结/传递、最终库单次rerun_02 |
| deployment/CLI | 原生子进程provisioner、fresh worker/session、模型驱动角色服务、持久ledger、doctor/check-sam3、配置与skill |

## 测试实际覆盖

| 测试组 | 内容 |
| --- | --- |
| contracts/geometry/snapshots/perception/evidence | frame/四元数/不可变数组、快照防覆盖/损坏、RGB hash、深度投影/覆盖、候选歧义、部分证据、身份与随动边界 |
| SAM3 wire/backend | 真PNG/回环HTTP、身份/尺寸/框mask一致性、空预测与缺字段区分、文本/单正点分支、分数/二值mask、代码/权重hash、禁下载、严格state_dict加载；模型是mock |
| API/policy/programs | registry参数校验/事件，禁止私有/native/文件/网络/reflection路径，允许Result.data；真实Python子进程持久状态、凭据环境、动作限额、超时/输出/管道边界 |
| runtime/parameters/relay | single-submit/native计数区分、重复动作去重、profile/预算/终止、低置信/停止、数值菜单拒策略文字、fine档实际动作超限拒绝、S2修复保留前缀/目标、review不等于成功、frozen禁S2 |
| native/provisioner | 数字ordinal排序与布局hash、fake native一次reset/实际预算/关闭、hash不符不启动、fresh allocation、原生启动失败不重试、进程组清理；没有启动Isaac |
| campaign | 每版本全dev、dev选定、unseen失败不改程序、heldout固定、provenance拒绝、重复worker/路径逃逸/中断拒重跑、verifier无解题代码、失败不晋升、恢复不重放 |
| curriculum | 新库向后传递、仅库改变且首轮低分时rerun_02、保留首轮分母/结果、配置改动拒恢复、完成后不重执行 |
| diagnostics | 文本+点提示+反投影验收、错误权重pin拒绝、doctor只读/空闲/commissioning条件；服务仍是fixture |
| 跨服务集成 | 实际agent RPC→provisioner RPC→Harness→真实技能子进程→SAM3 HTTP/深度投影→worker RPC；开发失败→repair→unseen/heldout；模型/物理是fixture |

完整命令：`python -m unittest discover -s tests -q`（NumPy/Pillow，允许本机loopback）。
不把测试方法数量当实际场景数，也不把 fixture 的native_success当真实任务成功率。

## 服务器仍需验收

1. 固定RoboDojo/Isaac/资产版本，真实 native-worker 启动/单reset、渲染/状态稳定、step_lim/终止计数与关闭。
2. 实测相机光学frame、米制depth、K/E、关节顺序/限位、夹爪极性与EE/TCP；完成commissioned profile。
3. 合法取得官方SAM3源码/权重，真实GPU strict加载、文本/点mask目视检查、RGB-D投影、显存与延迟。
4. 提供真实System 2模型驱动（fresh conversation＋多模态输入）及固定Jev模型/凭据；验证实际API响应与handoff。
5. 小预算单场景动作与停止验收，再跑正式互斥分区。检查完整分母、原生成功、失败证据与库冻结。

通用抓取规划/接触关系真值并未凭空实现：任务代码利用SAM3几何，宿主只接受可信物理关系检查器或原生成功，
未知就handoff。CPU测试不能证明真实识别、标定、安全性或物理成功；AST也不是OS权限隔离。
