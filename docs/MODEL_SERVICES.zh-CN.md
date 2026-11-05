# Claude Code / DeepSeek System 2 与 Jev System 1

2026-10-30：新增两个独立、仅loopback监听的认证RPC服务，不使用Docker。

## 服务器实际路径

工作目录：`/home/xuanzhi/EmbodiedHarness/experiment`。

- S2 API key：`secrets/system2.json`，字段`DEEPSEEK_API_KEY`。
- S1 API key：`secrets/system1.json`，字段`TYPESAFE_API_KEY`。
- 配置：`config/system2.json`和`config/system1.json`。文件只保存路径与参数，不保存供应商密钥。
- 内部RPC token由部署脚本生成，存放于`secrets/rpc-tokens.json`，无需手填。

密钥文件须为当前用户所有的普通文件、权限0600；目录0700。模板值以PASTE_开头，在填写前视为未配置。
服务每次请求重新读取密钥，填好后不需要重启服务。密钥不进入CLI参数、模型上下文或审计日志。

## System 2

已安装官方Claude Code，当前部署版本2.1.314；独立npm prefix位于`experiment/tools/claude-code`。
它是npm发布的原生二进制。服务器Node18触发了安装engine警告，但`claude --version/--help`均已验证；
没有因此升级共享Node。真实DeepSeek请求仍需填写key后验证。

配置依据[DeepSeek官方Claude Code接入](https://api-docs.deepseek.com/quick_start/agent_integrations/claude_code/)：
base URL为`https://api.deepseek.com/anthropic`，通过ANTHROPIC_AUTH_TOKEN传入DeepSeek key；
默认模型采用其文档的`deepseek-flash[1m]`。这是供应商模型别名，不声称固定了闭源权重版本。

内置ClaudeCodeDriver以print-mode调用Claude Code，每次创建新工作目录和CLAUDE_CONFIG_DIR，
禁会话恢复/持久化、关闭内置工具和外部MCP，只传公开角色上下文；使用restricted模式，不绕过权限。
这些参数依据[官方CLI文档](https://code.claude.com/docs/en/cli-reference)。
实际RGB以stream-json的image blocks输入；depth/K/E作为公开数值或摘要。
不把base64或numpy字符串冒充已传入图像；大depth数组由生成程序通过公开工具读取。
若供应商模型/API不接受图片或角色JSON，应报错并停止，不能降级成“只看文本”后宣称视觉验收通过。

服务继承AgentService的generate/repair/review/propose/verify协议；Plan仍经源码policy和严格角色校验。
verifier接收独立上下文，不读取proposer历史。输出必须是角色JSON；CLI失败不会把raw stderr或key回传。

```sh
embodied-harness agent-service --config /absolute/experiment/config/system2.json
```

## System 1

独立`system1-service`承接宿主规范化的测量状态与数值菜单，调用已有JevSelector；
服务确认response model与配置一致，返回option/confidence/model，记录无key的审计。
为保持原DualLoop条件，默认固定`jev-1.13.0`；[官方模型页](https://docs.typesafe.ai/models)列出该版本及更新版本，
不自动切换模型或改用latest。实际账户是否能调用，需填key后验证。

```sh
embodied-harness system1-service --config /absolute/experiment/config/system1.json
```

Harness通过`selector:remote`、system1_url/system1_token_env、jev_model接入。
动作额度、数值参数检查与物理成功判定仍由宿主负责；S1服务不能直接驱动机器人。
S1 key只由S1服务读取，不需要传给Harness或S2。

## 运行与验收

部署目录中的`start-model-services.py`启动两个服务；`check-model-services.py`只读查询其describe，
不执行模型请求和机器人动作。填写密钥前，服务可监听，但credentials_configured=false，不可宣称实验就绪。
使用RPC进行真正的select/respond才会调用供应商API，可能收费；不能用describe通过替代真实API/多模态检查。

全局doctor在remote selector条件下会检查S1/S2的凭据声明、模型pin、SAM3及native provisioner。
key已填写仅表示本地配置存在，不表示账户余额、API权限、模型/图像兼容性已通过。

真实实验仍需要标定过的robot profile与native provisioner；本次服务部署不自动开启动作实验。

## 本地测试

新增6项测试：真实假CLI子进程输入RGB、独立目录/配置和无历史、S1凭据不传S2；
占位/权限/符号链接拒绝和密钥热加载；CLI错误不泄密、超时杀进程；
RGB/K元数据；S1真实回环RPC/模型pin；Claude driver到角色RPC链路。
不以fixture测试冒称真实DeepSeek/Jev的API调用成功。
