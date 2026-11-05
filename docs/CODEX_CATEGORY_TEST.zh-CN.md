# 让 Codex 按类别启动 RoboDojo 测试

仓库根目录的 [AGENTS.md](../AGENTS.md) 是 Codex 的自动阅读入口。
配置一次真实部署后，可以直接说：“读取 AGENTS.md，运行 RoboDojo 的 X 类别测试”。
这里的类别是部署者登记的 native task 对应项，不猜测 RoboDojo 的分类，也不自动选布局。
当前聊天负责启动与监督；campaign 中每个 System 2 角色使用独立模型会话。

## 一次性部署

先完成 [服务器部署](SERVER_DEBUG.zh-CN.md) 中的 Isaac、profile 标定、真实 SAM3
快照验收、partition 和 freeze。provisioner 与 coordinator 在同一模拟器设备运行。
这些真实设备前提不能通过新增说明文件或 fixture 补齐。

System 2 已提供 Codex CLI 驱动，无需再编写 `my_model_driver`：

```sh
python -m pip install -e '.[sam3-client]'
codex --version
codex login status
export EMBODIED_CODEX_MODEL=YOUR_PINNED_MULTIMODAL_MODEL
embodied-harness agent-service --factory embodied_harness.codex_driver:create --port 8123
```

沿用部署说明配置 `EMBODIED_AGENT_TOKEN`，不要在仓库写入凭据。
Codex CLI 必须支持 `exec --ephemeral --ignore-user-config --image --output-last-message`；
按 [官方非交互模式说明](https://learn.chatgpt.com/docs/non-interactive-mode)安装兼容版本。
可用 `EMBODIED_CODEX_BIN` 指定可执行文件。模型必须支持图像，并有实际可用的登录与额度。
驱动每次使用空临时目录、新会话和 read-only 模式，不读取用户 config 中的 MCP/个性化配置；
RGB 真正以 PNG attachment 传入，小矩阵保留数值，大 depth/mask 保留无损公开 NPY 输入。
超时、非法 JSON 或模型错误直接失败，不隐藏重试。服务 token 不继承给 Codex 子进程。
这不是 OS 读权限隔离：请在无原生布局、评分代码和其他私有数据的专用账号/设备运行角色服务。
模型生成与真实物理没有包含在本地 fixture 测试中。

复制 `configs/campaign.example.json` 到 `configs/campaign.local.json`，填入真实公开任务指令、
manifest、冻结库、独立 run_dir、服务地址、SAM3 指纹和 System 1 模型。
复制 `configs/categories.example.json` 到 `configs/categories.local.json`，将 `my_category`
改为用户熟悉的类别名，`task` 填真实 native task 标识（须与 manifest.task 完全一致），
`config` 指向对应 campaign 配置。每类一个 campaign 配置，可以复用公共服务与冻结库。
registry 中的 config 路径相对 registry 文件；campaign 中的 partition_manifest、frozen_knowledge、
run_dir 路径相对 campaign 配置文件。也可全部使用绝对路径。本地配置已被 gitignore 排除。

## 运行

```sh
embodied-harness test-category --list
embodied-harness test-category --category my_category --dry-run
embodied-harness test-category --category my_category
```

可以加 `--registry /absolute/path/categories.local.json`。dry-run 不创建 run_dir，不访问服务；
正式命令先运行只读 doctor，失败则不进入 campaign。doctor 不证明真实标定、模型推理或物理验收。
通过后使用现有 dev 修复→选定冻结→unseen→heldout→独立技能验证完整协议，
没有另建一套宽松评测。结果位于配置的 run_dir；完成已有配置会读取持久证据，
重新实验需要修改 run_dir，禁止通过删除失败 ledger 重试。

## 补齐范围与剩余验收

原先已有 campaign / curriculum / 原生 provisioner，但缺根 AGENTS.md、类别映射/统一启动、
可运行的默认 Codex 模型驱动。以上已补齐。此入口是 shell CLI，尚未提供 ASPIRE 的原生
`mcp__cap__*` 接口；运行整类 campaign 不需要该接口。

服务器最近一次记录见 [部署状态](DEPLOYMENT_STATUS_20261030.zh-CN.md)。仍需现场确认
native reset/capture/close、profile 完整标定、真实 SAM3 文本/点/深度验收、System 1 凭据、
Codex 多模态调用及服务就绪，然后小预算验证。不能据此文档宣称当前服务器已可直接完成测试。

本次本地验证（2026-10-30）：完整 unittest 134 项通过，包含新增 6 项类别/CLI/驱动测试；
HTTP 与技能子进程真实运行，Codex 进程调用、模型和物理使用 mock/fixture。
未调用付费模型，也未启动远端实验。CLI 参数已核对本机 `codex exec --help` 与上述官方文档。
