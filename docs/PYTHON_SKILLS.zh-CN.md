# Python 技能与双循环契约

技能在普通持久 Python 子进程运行，无 Docker。默认宿主 sys.executable，可用 skill_python 指定
安装 NumPy 的解释器；SDK 源码路径由宿主设置。临时 cwd、无服务 token 环境、进程组关闭、有界输出和管道期限。
允许 math、注入的 Pose/ArmCommand/RobotAction 及公共工具；AST 拒绝文件/网络/反射/私有/native访问，
每条 API 由宿主注册表仲裁和记录。这是程序 provenance 契约，不是 OS 安全沙箱。

公共工具：get_observation、get_instruction、get_robot_state、localize_object、segment_text、segment_point、
mask_to_point_cloud、estimate_geometry、servo_action；动作只用 step。通过工具 schema 发现参数；
Result.ok 为状态，Result.data 为数据，失败保留 error/部分证据；不能吞掉 step 失败后声称完成。

五个 hook，完整示例见 examples/ee_skill.py：

- skill_init(parameters)：初始化 JSON/typed 状态和 runtime_parameters，无动作。
- skill_observe(state)：刷新公开观测/感知，返回 feedback（包含 source），无动作。
- skill_context(state)：返回 status、typed target_pose/active_arm（可选）和 parameter_scope，无动作。
- skill_options(state)：active 时返回2..12个数值选项；done 时可为空，无动作。
- skill_tick(state, 'advance')：读取宿主写入 runtime_parameters 的选值，最多一次 step。

parameter_scope 必须包含 stage、local_mode、contact_mode、evidence_valid、parameters、adjustments。
parameters 每项含 unit/min/max/value，范围不得越过宿主 LIMITS；adjustments 把 option 映射到 {parameter,value}。
参数支持平移/旋转步长、目标 xyz 小偏移、夹爪步长。System 1 只看宿主规范化的机器人当前测量、目标误差、
新鲜 SAM3 几何与数值菜单；不把 System 2 的策略文字、任意 context 或 task instruction 塞给选择器。
代码持有阶段/策略。Jev 回答须合法选项/概率/置信度；FixedSelector 是独立基线条件。

宿主注入参数后 tick 只收到 advance；宿主再次校验实际平移/旋转/夹爪和关节动作幅度，拒绝选fine却提交大步。
任何 tick ≤1 native 提交；提交数与 native action count 分开记录。联系物体的 pregrasp/contact/release
要求当前 state_version 的 SAM3 几何；证据不足/未知/低置信度/预算到限返回 needs_system2。
这是有限参数与动作上限检查，不是对任意代码语义与物理安全的形式化证明。

代码 done 仅为 awaiting_verification；可信宿主 relation evaluator 在当前观测上为真才可 verified_subtask。
S2 fresh-image review 可确认开发期子任务，只有原生终止成功才是 native_task_succeeded。
有序 Relay 保留完成前缀、当前任务 ID/目标，最多2次干预；3次版本预算不得扩大，AST+参数重复被拒。
已提交动作的传输异常不自动重试。每次修复启动新 Python 子进程，不继承旧 globals。

正式 fix-loop：单个 Python 技能可内部管理完整任务阶段；多个 subtask 的 frozen 执行必须有可信宿主关系检查器。
所有 dev 版本跑完整 dev 矩阵，S2 修复只看 dev；选定后 unseen/heldout 禁 S2 generation/review/repair。
测试范围与服务器待验收项见 LOCAL_BUILD.zh-CN.md；实际模型生成、GPU/接触物理尚未在本机验证。
