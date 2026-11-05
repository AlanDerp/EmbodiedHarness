# 服务器调试与实验入口

目标：以 ASPIRE tabletop fix-loop 为协议，以 RoboDojo 为环境，System 2 生成/修复 Python 技能，
System 1 在宿主限定的数值空间内选择。所有服务使用普通进程，无 Docker。
本地完成的是代码与 CPU/HTTP/子进程测试；下面的真实设备验收仍需在服务器执行。

## 设备与安装

建议 coordinator、provisioner、Harness 在 Isaac/RoboDojo 设备 A；SAM3、System 2 可在设备 B。
设备 A 到 B 使用 SSH tunnel 或显式受控服务地址；默认所有端口仅绑定 loopback。
provisioner 返回的是设备 A 的 loopback worker 地址，所以 coordinator 必须也在 A。
System 2 不得到 RoboDojo runtime root、分区清单、原生评分或 SSH 访问。

```sh
python -m pip install -e '.[sam3-client]'
python -m unittest discover -s tests -q
```

Isaac 的 Python 环境也要安装本项目及 NumPy/Pillow；provisioner 的 python_command 必须是实际可用的
Isaac Python 启动 argv，例如 `["/srv/IsaacLab/isaaclab.sh", "-p"]`，不能直接假定系统 Python 可启动 Isaac。
原生适配集中在 adapters/robodojo_native.py；显式 runtime_root 指向已部署 RoboDojo，不导入旧项目，
不修改 vendor。启动一次 reset，按原生 step_lim 预算，每场景新进程，关闭时清理原生环境与 App。
部署前记录 RoboDojo/Isaac/资产版本；native.json 会记录布局 hash 和 eval_env.py hash。
本地测试没有证明目标 vendor 版本兼容，首次启动必须核对实际字段、渲染/物理稳定性。

## 标定与 SAM3

复制并填好 configs/arx_x5.profile.template.json。确认实际有序关节名/限位、workspace、
夹爪开度方向、T_ee_from_tcp；光学系必须实测确认。原生 OpenGL camera-parent 外参仅在确认后
使用 opengl_verified，转换为 `E @ diag(1,-1,-1,1)`；depth 是米制 distance_to_image_plane。
commissioned=true 是操作员的标定声明，不是程序自动测量的证明。不要把占位模板改个布尔值就执行动作。

按 [SAM3说明](SAM3.zh-CN.md)安装获授权的官方源码和 checkpoint，完成真实文本/点提示与 RGB-D 验收，
把 /ready 的完整 model_version 固定进 campaign.harness.sam3_model_version。
不得使用真值 mask、颜色阈值或测试 fixture 代替模型推理。

## 启动服务

令牌只通过进程环境传入，不写配置。设备 A 的 coordinator/provisioner/worker 需共享 worker token；
agent、provisioner RPC 分别用独立 token；Jev 使用 TYPESAFE_API_KEY。这里不填任何凭据样例。

```sh
embodied-harness provisioner --config configs/provisioner.example.json
embodied-harness agent-service --factory my_model_driver:create --port 8123
```

可直接使用内置 Codex CLI 驱动：`--factory embodied_harness.codex_driver:create`，设置
`EMBODIED_CODEX_MODEL` 并完成 Codex 登录，见 [类别测试入口](CODEX_CATEGORY_TEST.zh-CN.md)。

第二条在 System 2 设备运行。my_model_driver 是实际模型提供方接入点：`create()` 每次返回新对话对象，
其 `respond(role, instructions, context)` 返回角色契约 JSON，可选 `close()`。
服务内置完整角色提示词、Plan 源码校验、独立 verifier 上下文白名单；没有把某一家 LLM SDK 写死。
模型驱动要支持 context 中的 typed Observation：图像在 observation.cameras[camera].rgb，按实际模型
接口编码为图像输入，不要只把 ndarray 的字符串发给模型。工具调用和模型会话不应共享 proposer 历史给 verifier。
当前角色协议与示例见 src/embodied_harness/instructions/fix_loop.md。

## 分区、冻结与运行

在 A 的 RoboDojo 目录枚举真实有限布局。scene_ref=`bank*1000000+ordinal`，ordinal 是数字文件名排序的
下标，不是文件名后缀。refs.example.json 只演示格式，必须按实际存在的布局及实验规模填写。
partition 命令会拒绝跨组重叠及内容相同的布局，并锁定 hash；私有 manifest 不传给 agent。

```sh
embodied-harness partition --runtime-root /srv/RoboDojo --task ACTUAL_TASK \
  --refs configs/partition.refs.example.json --output /srv/config/partition.json
mkdir -p /srv/knowledge/empty/skills /srv/knowledge/empty/lessons
embodied-harness freeze /srv/knowledge/empty /srv/knowledge/initial
embodied-harness doctor --config configs/campaign.example.json
embodied-harness campaign --config configs/campaign.example.json
```

先替换配置中的绝对路径、任务指令、服务地址、模型 pin。doctor 只检查版本、服务/标定声明和空闲资源，
不调用付费生成、不分割、不执行物理；不能代替真实快照、相机标定与小预算物理验收。
先单场景、小动作额度确认网络、图像、预算、夹爪与 TCP，再扩大正式分区。
多任务用 `embodied-harness curriculum --config configs/curriculum.example.json`。
项目 skill 在 `.agents/skills/fix-loop/SKILL.md`；安装到其他 agent 时保留仓库引用。

## 执行与证据

每个候选版本重测全部 dev，修复只看 dev 证据；按 dev 成功数选定，平分保留早期版本。
selected.json 记录程序/初始库 hash；unseen 每场景一次，heldout 固定程序、关闭 System 2 修复。
固定评估的完整任务可用一个 Python 技能内部管理全部阶段；多 subtask 必须有可信宿主 relation evaluator，
否则 code done 会 handoff，不能假设 frozen 模式仍可请求 S2 review。

Session/Relay 可单独用于开发期有序子任务接力和 S2 review/repair；review 不能替代原生任务成功。
严格 campaign 将修复放在 dev 版本之间，不在评估 episode 内调参。Jev 是 System 1 正式条件；
fixed 仅作协议/基线调试，报告时分开。

技能提议保存 skill_report.json；独立 verifier 只看公开目标/观测/候选库，最多两次在前两个 dev 重构。
仅实测 native 成功且源码/固定计划审计通过才产生 promoted 库；这是复现检查，不证明跨任务泛化或语义无泄漏。
多任务按显式顺序传递 verified 库；最终库改变后，低于阈值任务最多一次 rerun_02，结果与首轮分开。

campaign.json 为单宿主控制器的持久 ledger；不要并发启动同一目录。
每次外部调用/动作前记录 running，完成后保存证据。恢复不重跑 failed/interrupted/未知提交；
保留失败目录，若需重新实验使用新目录。没有隐藏物理重试、场景换号或失败剔除。
每 trial 都有 worker/agent 会话身份、快照、events.jsonl、程序 provenance；native 日志只在私有目录。

AST/公开工具白名单不是 OS 文件/网络沙箱，也不替代专用运行账号。技能子进程不继承服务 token，
宿主仲裁动作与安全限额；不宣称“无 Docker”具有容器级权限隔离。
