# 本地构建交付与验证范围

本轮结束：93项CPU/本机HTTP/可信子进程测试通过，相比上轮52项新增41项。
未启动Isaac Sim、真实System 2、真实Jev或SAM3。技能已改为直接Python子进程，不需要Docker。

| 模块 | 本地实现状态 |
| --- | --- |
| contracts/geometry | 观测来源、动作/回执、同帧点云、显式坐标系、位姿/TCP变换 |
| artifacts/perception | 快照、完整定位结果日志、候选消歧、几何关联、有限范围的随动证据 |
| providers/services | SAM3服务/客户端入口；Jev numeric choice客户端与注入式测试 |
| embodiments/adapters | 显式ARX profile、原生字典桥、worker与远程client、动作去重、终止/预算边界 |
| control/runtime | 单步servo、实测到位、阶段计划、选择器、交接/修复、停止与所需证据门控 |
| programs | 持久Python hook协议、真实子进程启动、宿主逐primitive动作计量，已接delegate_python |
| tools/agent接入 | 统一工具字典+认证RPC；外部System 2客户端示例，不内嵌具体LLM代理 |
| evaluation/batch | skills/lessons冻结、哈希/只读staging、固定job计划、失败/中断保留、fresh会话检查 |
| config/CLI | worker/serve/batch/freeze/verify/validate-profile、显式配置与未commissioned模板 |

## 本轮新增测试分组

| 测试文件 | 数量 | 实际检查 |
| --- | --- | --- |
| test_runtime.py | 25 | 原生转换、动作去重/回执、profile、单步控制、预算、原生成功、修复/停止/日志、认证RPC两跳 |
| test_evidence.py | 4 | 唯一/歧义身份关联、随动/原位/未知证据、四元数和TCP换算 |
| test_batch.py | 5 | 全部分母、fresh agent、失败不重跑、中断保留、恢复计划不可改 |
| test_programs.py | 7 | 持久子进程、primitive额度防绕过、输出/死循环/管道超时、解释器选择与凭据环境不继承、hook到worker链路 |

之前52项继续覆盖contract、geometry、快照/定位和SAM3 HTTP。模拟原生worker每次提交
故意增加2个native action count，验证报告不把提交数冒充原生物理消耗。
HTTP测试实际创建loopback服务；模型响应、机器人运动学和Jev回答是fixture。

## 批量运行接口

batch.py为宿主代码，job包含job_id/task/scene_ref，私有scene_ref不进入agent工具注册表。
场景计划由部署侧在冻结知识后确定，运行中不可更换。启动示例：

```sh
embodied-harness batch --plan /private/path/jobs.json --frozen /path/frozen \
  --run-dir /path/new-batch --factory my_deployment:open_episode
```

可信open_episode(job, workspace)返回具有worker_session_id、agent_session_id、run()、close()
的driver，在目标设备初始化新的worker和代理。runner在run前检查会话ID未重复，workspace
只接收skills/lessons，运行前后校验知识，提交前持久记录running状态。
run返回primary_native_success/recovered_native_success/interventions，必须来自原生证据；
runner验证字段一致性，但不能凭字段自行证明远端评分真实性。

恢复只执行pending项，running改为interrupted，failed/completed均不自动重跑。
分母始终包含计划全部job，主成功和修复成功分别汇总。ledger为单宿主控制器使用，
不是分布式任务队列；目前没有GPU资源调度、自动场景抽样或自动agent provisioning。

## 转移到目标设备时还需完成

1. 安装/固定RoboDojo与Isaac版本和资产，提供可信backend factory，核对实际原生字段与生命周期。
2. 实测相机frame、关节顺序/限位、EE/TCP和夹爪极性，填好profile；不能直接使用模板占位值。
3. 安装SAM3并部署本地权重，做真实RGB-D定位与显存/时延验证；提供Jev凭据并核对真实API响应。
4. 安装技能Python环境，配置目标网络和System 2工具客户端，注册任务适用的物理关系检查器。
5. 若做正式评估，补齐设备侧fresh agent/scene driver与隔离挂载，先零动作验收，再小预算闭环。

当前具备本地可测试的最小双循环工程链路，尚不是全部ASPIRE功能的等价复刻。
固定完整任务程序封存/Test局部修复协议、通用抓取/规划服务、全任务关系验证和正式隔离评估
仍未完成，不把本地fixture通过报告为机器人实验成功。
