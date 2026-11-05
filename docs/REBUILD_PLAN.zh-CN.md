# ASPIRE Fix Loop + RoboDojo DualLoop 重建清单

目标基线：ASPIRE fa00f0ef2bc1d91d594b9aa502c22133c9f538f7，
EmbodiedDualLoop 01210eedc8b9424e15b0d1b0e1e115b026dc89c8。本清单替代“已有底座即完整实现”的判断。
只构建/测试本地代码；Isaac Sim、System 2、真实模型在目标服务器验收。无 Docker。

## 起始审计发现（2026-10-29，实现前）

| 优先级 | 当前缺口 | 依据/影响 |
| --- | --- | --- |
| P0 | 没有 ASPIRE fix-loop skill、worker/coordinator/verifier 提示词及 campaign 状态机 | 只有 Session 接受 revision，没有自动生成→按失败模式修复→选定→unseen→held-out→技能验证/晋升 |
| P0 | RoboDojo 仅接口桥，启动、布局枚举/划分、原生预算来源仍交给外部 factory | 无法用本项目独立构造实际环境，现有 native_action_limit 可被人工填写，与官方限额未核对 |
| P0 | SAM3 只验证了 fixture；backend 关键分支未有本地契约测试 | 缺失 masks/scores 被当空成功；缺代码版本、checkpoint约束、响应框/掩码一致性、真实服务验收工具 |
| P0 | Python 技能允许任意 import/文件/网络，不符合 ASPIRE 的 non-privileged API 程序契约 | 无 Docker 仍应有公共工具白名单、源码检查和逐调用 provenance；不能把检查称为 OS 沙箱 |
| P0 | System 1 仅两档 servo 或任意文字菜单，未完整承接原 DualLoop 的数值参数空间 | 需宿主验证 parameter_scope、数值选择、当前测量来源，拒绝策略 ID 冒充参数与过期状态 |
| P0 | System 2 仅 RPC 调用者，没有有序计划/任务状态/自动接力协议 | 需远端 agent bridge、失败上下文、修订与完成标准、开发修复与 held-out 修复条件分开 |
| P1 | 一般预算/容差硬编码在 Session，且报告/错误/原生结果缺统一持久索引 | 配置一次生效，报告带预算/原生终止/程序哈希，失败保留并可核验恢复 |
| P1 | 技能库只有文件冻结，无提议/独立验证/晋升 | 不允许未经 verifier 重做的知识进入下一任务库；验证不等于泛化 |
| P1 | BatchRunner 只接受外部 driver，不能完成完整 fix-loop 分区与防泄漏 | 开发、unseen、held-out 必须按 RoboDojo 有限布局显式不相交；选择后冻结，评估禁 System 2 调参 |
| P1 | 缺 doctor、真实 RGB-D/SAM3 校验、可运行的目标设备启动配置 | 测试结果不能代替模型/相机/引擎验收 |
| P2 | 配置/说明多处出现过时状态、角色与代码耦合在大 Session | 收敛到 API/registry、执行循环、campaign 三个职责；不加无用服务和空目录 |

## 实施顺序与验收（必须依次执行）

- [x] 0. 审计并写入本清单，锁定来源和协议边界。
- [x] 1. SAM3：补齐严格输入/输出与模型 provenance、backend 无权重测试、真实验收命令。
      验收：空结果≠缺字段，形状/分数/框一致，文本和点提示分支都覆盖；无需真实权重的 CPU 测试通过。
- [x] 2. ASPIRE 风格 registry/API 与生成程序契约：统一工具入口、只允许公共工具/构造器和 math，记录 provenance。
      验收：禁止 simulator/奖励/文件/网络/反射访问，所有动作仍经宿主；旧示例/测试同时迁移。
- [x] 3. 完整 System 1 参数循环与 System 2 接力：合法数值参数范围、宿主注入测量、计划/修订/失败交接。
      验收：参数不是策略 ID；每tick≤1提交，低置信/过期/未知交回；自动 repair 保留目标与前缀。
- [x] 4. 独立 RoboDojo backend：显式 runtime root、原生启动/一次reset/公开RGB-D/官方限额/终止/关闭，布局清单。
      验收：不导入相邻项目，不改vendor，不暴露私有场景/评分；CPU fake native 覆盖生命周期，GPU步骤单列。
- [x] 5. ASPIRE fix-loop campaign：coordinator/worker/verifier、development→unseen→held-out、选定程序/库冻结、技能晋升。
      验收：自动修复只看开发证据；完整开发版本矩阵；一次unseen；held-out不修复/不选模型；独立 verifier 不见答案。
- [x] 6. 串联部署CLI、服务检查、RoboDojo fix-loop skill与文档；清理过时配置、跑完整测试与打包。
      验收：模拟远端 agent + native + SAM3 的端到端 campaign；本地证据和真实服务器待验收项分开列出。

## 协议选择

采用 ASPIRE tabletop fix-loop 的“开发修复→选定程序→一次unseen→held-out→独立技能验证”流程，
以 RoboDojo 实际 bank/ordinal 布局替代 LIBERO seeds。System 1 执行有界参数化 Python 技能，
System 2 负责总体计划与开发阶段修复。held-out 默认冻结程序、禁止 System 2 调参；若将来启用
Test局部修复，必须另设实验条件、分别报告primary/recovered，不能混入严格fix-loop。
BEHAVIOR 的逐scene增长policy/只冻结库是另一协议，本轮不混用。

ASPIRE 的许可、提示词和 API 设计在来源说明中记录；复制/改写其内容须保留 Apache-2.0 与 NOTICE。
SAM3 使用官方代码/授权权重，不提供真值mask、不读取私有物体位姿，不把颜色阈值或fixture伪称SAM3。
即使本地所有验收通过，仍须在服务器验证真实 checkpoint、API、相机标定和物理任务成功。

## 执行记录

后续每步在此记录改动、测试内容/数量、未完成边界。只有对应验收通过才勾选；
不得以“有接口”代替实现，也不得以 CPU fixture 代替 GPU/机器人实测。

1. SAM3 本地验收：98项测试通过（新增5项backend测试及wire框一致性负例）。
   覆盖文本/点分支、原分辨率、空与缺字段区分、异常mask/score、checkpoint hash与代码hash、
   禁止隐式下载；增加check-sam3真实快照验收命令。实际权重与GPU未在本机运行。
2. API/程序契约：101项测试通过。新增registry、CapApi、ASPIRE风格源码policy与工具调用记录；
   示例与真实子进程测试迁移到注入构造器/公共工具，拒绝os/numpy/文件/反射/私有状态访问。
   保留无Docker约定；这是非特权API契约，不宣称OS安全沙箱。Apache来源许可/notice已记录。
3. 参数/接力本地验收：108项测试通过（7项新测试）。数值菜单与测量由宿主规范化，
   赋值进入runtime_parameters后tick只收到advance；动作幅度由宿主再次校验，接触模式须有新鲜SAM3几何。
   Relay支持远端generate/review/repair、有序前缀与目标保留、strict Test禁System2。
   S2视觉review不等于原生成功。真实Jev与远端agent部署仍须验收。

4. Native/参数效果验收：112项通过。内置独立bootstrap、RGB-D、numeric布局与hash、原生step_lim、
   单reset/故障锁定/关闭，fake native验证预算和生命周期；增加“选fine实际大步”拒绝测试。
   相邻项目不导入，vendor不改；真实Isaac/GPU启动仍需设备验收。
5. Campaign验收：开发矩阵、确定性选定、程序hash、一次unseen、heldout禁S2、技能使用报告、
   两次独立verifier/verified晋升、fresh会话与失败/中断不重跑，8项专项测试通过。
   多任务课程、verified库传递和最终库单次rerun_02有独立测试；不覆盖首轮结果。
6. 最终128项测试通过。真实HTTP+Python子进程的跨服务fixture campaign覆盖分割/投影、开发失败修复、
   unseen/heldout。另覆盖provisioner hash/启动失败/关闭、doctor只读与SAM3文本+点验收。
   补齐agent-service/provisioner/campaign/curriculum/partition/doctor/check-sam3 CLI及部署配置，
   项目fix-loop skill通过quick_validate。wheel构建成功，确认含角色提示词和Apache许可/NOTICE。
   真实SAM3严格加载、Isaac物理、模型驱动多模态输入/Jev和标定验收没有在本机完成，详见LOCAL_BUILD。

## 最终边界

清单勾选表示本地代码验收完成，不表示真实环境实验完成。采用ASPIRE协议与工程职责划分，
没有把原框架整包复制，也不宣称原始benchmark等价。System2实际模型驱动为外部模型SDK接入点，
服务/角色/Plan协议已内置；环境provisioner与native bootstrap已内置，不再要求用户另写环境factory。
本轮未推送GitHub（此前连接写权限/本机Git凭据未就绪），工作保留在本地目录。

最终审阅：低分候选即使显式选中也返回low_confidence，不生成可执行几何；补充对应负例后128项全量复跑通过，wheel重新构建成功。
