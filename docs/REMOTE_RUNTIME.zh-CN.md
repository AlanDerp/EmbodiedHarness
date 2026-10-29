# 本地代码构建与远程 DualLoop 接入

## 现有可执行边界

本轮已实现 worker 原生接口桥、机器人 profile、单步控制、System 1 selector、
有界阶段运行时、远端 System 2 工具 RPC、事件记录、几何身份关联/随动证据、知识冻结。
没有启动 Isaac Sim、真实 System 2 或真实 Jev/SAM3 模型调用。

```text
System 2 设备 --认证 RPC--> Harness 宿主 --认证 RPC--> RoboDojo worker / Isaac Sim 设备
                              |
                              +--> SAM3 HTTP 服务
                              +--> Jev SystemOne API（可选）
                              +--> 快照、分割、动作与交接记录
```

Harness 可部署在本机或任意宿主，不要求与 System 2、仿真在同一设备。当前 RPC 服务采用
同步调用；必须使用可信网络、SSH 转发或受保护的 TLS 入口，HTTP 本身不加密 bearer token。
两个 RPC 服务分别配置 token，token 仅从环境变量读取；不写入 run 配置和事件日志。

## 1. 仿真设备的接入契约

启动入口：

```sh
embodied-harness worker --factory my_station:create_backend \
  --profile /path/to/commissioned-arx.json --episode-id unique-episode-id \
  --native-action-limit 500 --phase playground --port 8120
```

`my_station:create_backend` 是部署设备上可信、已安装的 Python factory，返回具有
`capture()`、`step(native_action)`、`close()` 的对象。前两者返回 `NativeFrame`。
factory 不由 System 2 上传或通过 RPC 执行，它负责目标设备特定的 Isaac/RoboDojo 启动。
已有 provisioned task 如具备 public_observation/control_step/control_steps/terminated/
truncated/success/close，可直接包装为 `ExistingTaskBackend(task)`。

本项目现在提供接口桥和启动插件入口，尚未内置一套已验证的独立 RoboDojo/Isaac 引擎
bootstrap、资产安装或虚拟环境。因此部署时仍须实现/配置这个设备 factory，不能仅凭
本地测试宣称新框架已经原生启动 RoboDojo。

NativeFrame.public 只包含当前 instruction、vision、state。vision 使用 RoboDojo 的
color/depth/intrinsic_matrix/extrinsic_matrix；state 使用 left/right_ee_pose（XYZ+WXYZ）、
left/right_arm_joint_state、left/right_ee_joint_state（一项归一化值）。
原生计数和终止状态从宿主获取，不能由代理编造。

worker 独占一个 episode；不允许外部控制器绕过它运动。动作模式为完整双臂 EE 或关节
字典，同时携带两个夹爪字段。step 是一次原生提交，返回的计数差可以大于 1。
**原生 driver 自身仍须执行原生动作上限**，worker 在提交前检查剩余额度；它无法中断
一个原生 take_action 内部的物理步，不把提交数冒充原生预算。

已知兼容点：上游 success 初始可能为 true，只有 terminated + success 才映射任务成功。
worker 重启生成新 session_id，旧宿主的动作拒绝；同一 action_id 和同一命令返回原 receipt，
不同命令复用 ID 拒绝。去重记录仅存于当前 worker 内存，重启后不自动恢复未确认动作。
动作分发后的异常会锁住 worker；必须检查实际状态后重新建立会话，不能自动重试。

## 2. Profile 与标定

`configs/arx_x5.profile.template.json` 是故意未填完的模板，不能直接用于操作。
填入实际关节名顺序、限位、workspace、EE→TCP 变换和标定记录。模板里的数值只是占位符，
不是测得的 ARX 机械参数。commissioned=false 拒绝动作。

camera_convention 必须显式指定为 optical_verified 或 opengl_verified；后者执行
T_world_from_native @ diag(1,-1,-1,1)，前提是已核对原生父节点与相机轴，不能用字符串
代替标定。未知 convention 拒绝生成观测，避免把未核对的矩阵伪装成标准光学外参。
夹爪归一化值沿用原生观测/命令约定，部署时必须核对开合极性；不代表力或实际夹距。

`tcp_target_to_ee()` 提供显式 TCP→EE 目标换算；servo 接收的是 EE target，不能直接
把 TCP 目标当 EE 目标。workspace/joint bounds 校验不等于碰撞规划、IK 或接触安全证明。

## 3. Harness 宿主与 System 2

根据 configs/harness.example.json 创建私有配置，设置 worker URL、两个 token 环境变量名、
新的 run_dir、SAM3 URL 与 selector，然后启动：

```sh
embodied-harness serve --config /path/to/harness.json
```

run_dir 必须不存在，失败的运行也保留目录；不要通过复用目录覆盖证据。
远端 System 2 安装同一版包后，通过 `RpcClient.call()` 调用统一工具：

| 工具 | 用途 |
| --- | --- |
| capabilities | 查看支持的阶段、selector、关系 evaluator、phase |
| observe | 零动作获取并保存当前公开观测 |
| localize_object | 对当前 episode 的已保存快照调用 SAM3 并记录定位结果 |
| delegate | 提交声明式有序阶段计划，阻塞至完成、交接或失败 |
| delegate_python | 提交持久Python子进程技能hooks，可选skill_python解释器 |
| get_report | 读取最近交接报告 |
| stop | 锁住会话，阻止下一次动作；已提交动作仍需返回真实结果 |

stop 可并发进入，不等待 delegate 的串行锁；不能硬中断正在运行的模型请求或原生动作。
当前没有自动 resume，stop 后须核对现场再建立新宿主会话，原生动作预算不因此恢复。

示例 clients 在 examples/system2_client.py；program.template.json 只展示字段，零坐标
与关系名必须用当前观测和工具能力替换，不能直接当操作程序。

manifest 支持 servo（arm、world EE target、可选归一化 opening）和 relation（关系名）。
无需在代理设备安装 simulator，也不允许从 RPC 发送 Python 代码、shell 或修改 simulator。
这是声明式任务DSL；另已提供Python技能路径，见PYTHON_SKILLS.zh-CN.md。
Python技能在独立子进程运行，不需要Docker；动作门控不等同于操作系统权限隔离。

## 4. 双循环与证据规则

System 2 负责有序计划、目标与修复。System 1 当前支持：

- fixed：确定性 normal 步长基线，用于检验执行工程。
- jev：显式配置 pinned jev_model，使用宿主 TYPESAFE_API_KEY；请求形状参考旧项目
  SystemOne choice 接口。当前只选择两档数值步长，输入是测得误差数值，不带 RGB。
  此条件为 servo-step-size-v1，不能说成已经迁移了旧项目全部 Jev 视觉模式。

normal：平移≤10mm、旋转≤0.1rad；fine：平移≤1mm、旋转≤0.01rad；夹爪每次变化≤0.02。
另一只手保持实测 EE/夹爪状态。每次决定最多发一次原生提交。到位根据动作后实测
位置、旋转和夹爪读数判断；没有额外“保持不动”调用冒充物理持续时间。

宿主限制每程序最多100提交/150决定/180秒/10秒本地观测年龄，置信度≥0.5、
位置容差≤3mm、旋转容差≤0.01rad、最多5次无进展。它们是当前显式工程策略，
尚未按任务调参。观测年龄使用宿主接收的单调时钟，不拿两台机器的 monotonic 值相减；
worker 同时验证来源状态。远端实际传感延迟尚需部署测量。

低置信度、超时、无进展、动作预算或未知关系返回 needs_system2。修复最多3次尝试，
revision递增、目标不变，不得扩大原预算/容差或降低置信度，不得改写已完成前缀。
只改任务元数据/阶段名称或预算不能重启同一个物理程序。

relation 由可信宿主 evaluator 返回同一 observation source 的 true/false/unknown；
缺 evaluator 或来源陈旧不能继续。配置 extensions 可列出可信 module:install_function，
在启动时注册 session.evaluators；没有内置对所有任务都有效的 grasp/placement 检查。

几何关联 associate() 只在唯一候选同时通过中心距离和尺寸门槛时关联，跨 episode/reset
拒绝；相似近邻仍返回 ambiguous。following_evidence() 区分物体随 EE 抬起、未随动、
身份/时序不明；随动不是抓取接触或长期无滑动认证。复杂任务仍须注册现场适用的检查器，
并在最终状态重新验证可能被后续动作改变的关系。

program_completed 只证明声明的阶段谓词已满足，不能当 native_task_succeeded。
原生终止成功才返回 native_task_succeeded。

## 5. 记录与冻结

宿主保存 config（只含 token 的环境变量名）、输入快照、完整定位结果、动作 intent/receipt、
selector请求状态/候选、Jev原始回答、阶段证据、时延与交接。记录失败使运行停止/不可验收，
动作结果不确定则禁止继续分发。日志包含用户任务数据，应由宿主权限保护。

```sh
embodied-harness freeze /path/to/playground/workspace /path/to/new-frozen-knowledge
embodied-harness verify /path/to/new-frozen-knowledge
```

冻结只传 skills/lessons，拒绝符号链接，保存逐文件哈希，原子提交；stage_knowledge()
要求新的目标知识目录并把文件设为只读。owner 可修改文件权限，因此仍须由部署提供
只读挂载/代理工作区权限。worker 不暴露 reset，Test 工具无法调用 reset。
当前不提供 Playground reset API、固定任务程序封存/自动 Test 局部修复、场景池抽样。
批量ledger和失败/中断恢复已实现，见LOCAL_BUILD.zh-CN.md；设备driver仍负责跨场景初始化和隔离。

## 本轮测试测了什么

当前总计93项测试（本轮新增41项），CPU、本机loopback及可信测试子进程运行：

- 原生观测转换、明确的OpenGL轴变换、关节顺序、TCP→EE与180度四元数变换。
- action_id去重、session改变/终止拒绝、不确定动作锁住、未commissioned/越界拒绝。
- 单步位移/角度上限、另一只手保持、q/-q等价、本地观测年龄与提交预算。
- 真实HTTP两跳：模拟远端System2→宿主→模拟RoboDojo；token错误与未注册reset拒绝。
- 阶段顺序、动作后实测到位、原生成功判定、低置信度零动作、无进展交接。
- 修复不得扩大预算/改变已完成前缀/只改名称；stop后不再发动作。
- 模型选择等待期间，远端stop可通过真实HTTP并发进入并阻止下一动作。
- relation缺失或陈旧不能验收；几何身份唯一/歧义/丢失及跨episode检查。
- 物体随动、原位、来源不明区分；证据写失败阻断；freeze只传知识并检测篡改。
- Jev分布解析使用模拟回答，没有真实API调用；RPC类型仅允许显式注册contract。
- NPY伪造超大尺寸头不会触发巨量分配，URL不允许内嵌凭据，知识符号链接拒绝。

测试中的NativeFixture是运动学假对象，无碰撞、接触、重力或真实图像，不能据此报告操作成功率。
