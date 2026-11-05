# 远程运行时

当前部署步骤统一维护在 [SERVER_DEBUG](SERVER_DEBUG.zh-CN.md)，模块与测试状态见
[LOCAL_BUILD](LOCAL_BUILD.zh-CN.md)，协议缺口和验收记录见 [REBUILD_PLAN](REBUILD_PLAN.zh-CN.md)。

设备 A：RoboDojo/Isaac、native provisioner、worker、Harness/campaign、System 1 client。
设备 B：SAM3 权重与服务、System 2 模型驱动与 agent-service。默认 loopback；跨设备显式 tunnel/地址。
provisioner 动态分配的 worker 为 A 的 loopback 地址，coordinator 也在 A 运行。

RPC 具备 bearer token、字节/并发上限、显式类型 wire、不用 pickle。动作 intent/receipt ID 关联、去重，
不自动重试未知提交。stop 是独立并发入口，可打断等待 selector 的执行；关闭 worker 清理原生 App。
令牌仅取环境变量；Python 技能子进程不继承服务凭据。RPC 不是 TLS 替代品，远端网络由部署者配置。

`native-worker` 使用内置 robodojo_native bootstrap；`worker --factory` 仅是保留的可信自定义 backend 入口。
`serve` 暴露公开工具与交互 Relay；`agent-service` 通过 fresh model driver 执行生成/修复/提议/独立验证角色。
`campaign` 内置严格 fix-loop；`curriculum` 内置任务顺序、verified 库传递与最终库单次 rerun sweep。
`batch --factory` 是兼容的低层 job runner，新的正式 fix-loop 使用 campaign/curriculum。

profile 必须来自实际标定。模板无有效 camera convention/commissioning 信息，不能直接运行。
通用接触关系检测、物理轨迹规划和技能正确性不由简单 EE 到位替代；未知即交回。
真实 native/model API、相机标定、延迟/显存与任务成功率须在服务器测量，不能用本地 fixture 代替。
