# 远端服务器调试入口（无需 Docker）

## 安装与本地自检

在服务器克隆本仓库后进入仓库目录，使用 Python 3.11/3.12：

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[sam3-client]'
python -m unittest discover -s tests -v
embodied-harness --help
```

93项测试只需要CPU、Pillow/NumPy和本机loopback。它们启动测试HTTP服务和Python子进程，
不依赖Isaac Sim、模型权重、API凭据或Docker。完整SAM3推理环境请单独按照SAM3.zh-CN.md准备。
当前进程管道使用POSIX接口，目标为Linux/macOS。

## 先接仿真，后接模型和System 2

1. 在Isaac设备配置已安装的RoboDojo启动factory，返回NativeFrame backend；接口见
   REMOTE_RUNTIME.zh-CN.md。也可将已有task包装为ExistingTaskBackend(task)。
2. 按实际标定填写configs/arx_x5.profile.template.json。占位关节、限位和外参约定不能
   直接运行；先使用validate-profile检查格式，再在设备核对真实观测。commissioned=false禁止动作。
3. 在仿真设备启动worker；在Harness设备配置worker URL和token环境变量。
   默认绑定127.0.0.1，跨设备可使用SSH端口转发，将示例URL保留为localhost。
4. 创建私有harness.local.json，参考configs/harness.example.json。首次将sam3_url设为null、
   selector设为fixed，run_dir选择一个不存在的时间前缀目录。skill_python默认null，使用当前解释器。
5. 先调用capabilities与observe，检查快照和位姿，再提交小预算的EE目标。
   之后接SAM3定位、Python技能、Jev，最后接完整System 2循环。

```sh
embodied-harness validate-profile /path/to/commissioned-profile.json
embodied-harness serve --config /path/to/harness.local.json
```

凭据放在EMBODIED_WORKER_TOKEN、EMBODIED_SYSTEM2_TOKEN及可选TYPESAFE_API_KEY环境变量。
System 2使用examples/system2_client.py或RpcClient；SDK可在System 2设备独立安装。
RPC地址与端口、factory和API示例见REMOTE_RUNTIME.zh-CN.md。首次调试不要直接使用示例零坐标。

## Python技能

delegate_python直接启动持久Python子进程，每次委托结束清理。默认环境已包含核心SDK和NumPy。
如配置其他skill_python虚拟环境，至少在该环境安装NumPy。无需镜像构建或任何容器运行时。
动作次数、状态版本、超时、原生终止和物理证据要求仍由宿主仲裁。
子进程保留运行账号的系统权限，不构成操作系统安全沙箱。

## 发布范围与当前局限

仓库只包含源码、测试、示例配置与说明；run记录、模型权重、凭据、虚拟环境及构建缓存均忽略。
实际RoboDojo/Isaac启动factory、profile标定、SAM3权重与System 2部署需要在目标设备配置。
遇到失败请保留run_dir里的config、events和snapshots；不要删除失败记录后反复重跑。
