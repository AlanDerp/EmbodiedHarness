# EmbodiedHarness

面向 RoboDojo / ARX X5 的新具身执行 harness。工程组织参考本地 ASPIRE，保留
EmbodiedDualLoop 的 System 2 规划、System 1 有界执行和证据交接思路。

当前已实现基础 contracts 与最小 Python 打包，包括观测快照、相机、位姿、机器人状态、
分割、点云、双臂动作、结构化结果，以及观测/分割 provider 协议。
另已实现同帧 RGB-D 投影、显式坐标变换及可见表面 PCA 包围盒。
SAM3 客户端、HTTP 服务和模型 backend 入口已实现。
本地还已实现 RoboDojo worker/client 接口桥、机器人 profile、单步控制、有界阶段运行时、
System 2 工具 RPC、fixed/Jev selector、事件记录、基础身份关联和知识冻结。
另有持久Python子进程技能hooks与批量评估ledger；本轮共93项本地测试通过。
没有运行真实 SAM3 模型调用、GPU 实验或机器人动作。

Python 3.11+ 安装与 CPU 测试：

```sh
python -m pip install -e .
python -m unittest discover -s tests -v
```

也可在已安装 NumPy 的环境使用 `PYTHONPATH=src python -m unittest discover -s tests -v`。
类型约定与最小示例参见 [contracts 使用说明](docs/CONTRACTS.zh-CN.md)。
几何接口、测试范围与局限参见 [geometry 使用与验证](docs/GEOMETRY.zh-CN.md)。
已接通[快照存储与定位流程](docs/PERCEPTION.zh-CN.md)：保存/读取公开观测、
注入分割 provider、候选消歧、深度覆盖检查和三维定位，保留失败时的部分证据。
[SAM3 服务说明](docs/SAM3.zh-CN.md)包括安装、部署、协议和 HTTP 测试范围。
[远程运行时与构建状态](docs/REMOTE_RUNTIME.zh-CN.md)包括设备分工、原生 backend 接口、
System 2 接入示例、配置模板与当前功能边界。现有 CLI 支持 worker、serve、batch、freeze、verify、validate-profile。
[本轮构建与测试总览](docs/LOCAL_BUILD.zh-CN.md)明确本地完成项和部署侧待验收项；
[Python技能说明](docs/PYTHON_SKILLS.zh-CN.md)包含解释器配置与hooks接口。
[服务器调试步骤](docs/SERVER_DEBUG.zh-CN.md)提供安装、自检及远端设备接入顺序；无需Docker。
本地最小链路已用模拟机器人验证；真实 Isaac 启动 factory、标定和模型联调仍在部署设备完成。
运行含 SAM3 协议测试的完整 CPU 测试集前安装 `.[sam3-client]`，并允许本机 loopback 端口。

参见 [架构与模块重建方案](docs/ARCHITECTURE.zh-CN.md)：模块边界、计划目录、
SAM3 服务分层、迁移来源、开发顺序与验收条件。

新 API / SAM3 / 双循环实验应使用独立的协议和实验配置，不沿用
`workspace-session-v1` 的四原语复现标签。旧项目和已有实验记录保持独立。
