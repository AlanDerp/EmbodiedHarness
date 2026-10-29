# 基础 contracts（首批实现）

入口为 `embodied_harness.contracts`，仅依赖标准库和 NumPy；无仿真、HTTP 或模型依赖。
本批为新实现，没有复制 ASPIRE 或 DualLoop 源码。

| 类型 | 约定 |
| --- | --- |
| `Pose` | 米、WXYZ 单位四元数、显式 frame；不偷偷归一化错误输入 |
| `ObservationSource` | episode/snapshot/reset_epoch/state_version、原生动作计数、宿主单调时间；未知 sim_time 为 None |
| `CameraObservation` | uint8 RGB、同尺寸光轴深度、无 skew 的 K、经过 adapter 验证的 world-from-optical 刚体变换 |
| `ArmState`、`RobotState` | 命名手臂、显式关节顺序、实测关节与 EE、归一化夹爪读数；核心不硬编码机器人关节数 |
| `Observation` | source、相机映射、机器人状态、当前指令；相机映射键必须等于 camera_id |
| `Segmentation` | 精确 source、camera_id、布尔 mask、XYXY 框、分数、提示、provider/model_version 和 RGB 哈希 |
| `PointCloud` | source、来源相机、目标 frame 和有限 Nx3 点；允许空云，不把它当成功定位 |
| `ArmCommand`、`RobotAction` | 明确全部手臂、同一 EE/关节模式、每臂夹爪命令、动作来源状态 |
| `Result[T]`、`HarnessError` | 成功数据或结构化错误；失败可保留部分证据；ok 仅指调用结果，不证明物理成功 |

数组复制后使用不可写 bytes backing；调用方修改原数组不影响 contract，也不能通过
`setflags(write=True)` 重新开启写入。映射复制并只读。数组类型不提供结构相等比较，
逐数值比较请使用 NumPy。`Result[T]` 的任意泛型 payload 不自动深度冻结，优先使用
contract 或 tuple 作为 payload。

深度保留 NaN/Inf/非正值以支持后续质量诊断；感知层必须过滤后才生成点云。
分割框采用像素 XYXY、最大边界 exclusive，允许 x2=width、y2=height。
分数限定 [0,1]，不是身份正确概率。RGB SHA256 哈希对象为原始行优先 uint8 RGB 字节，
不是 PNG 文件；形状与相机、快照一起校验。

```python
from embodied_harness.contracts import Result, ErrorCode, HarnessError

# 从 provider 得到候选后，消费前必须验证同一快照与图像。
def accept_candidates(observation, candidates):
    for candidate in candidates:
        candidate.validate_against(observation)
    if not candidates:
        return Result(error=HarnessError(ErrorCode.NO_CANDIDATES, "no matching instance"))
    return Result(data=tuple(candidates))
```

Provider 接口接收完整 Observation 和 camera_id，避免 provider 自己重新采帧。
外层 snapshot_id API 由后续快照存储解析后再调用 provider。
首批 `SegmentationProvider` 支持文本和单正点；没有声明视频跟踪能力。

`ObservationSource.same_state()` 比较 episode、reset_epoch、state_version 与原生动作计数；
重复读取可以得到新 snapshot_id，但不因此产生物理进展。adapter 必须正确维护这些版本，
类型校验无法凭空保证来源真实。分割复用要求精确 source 相等，不能只检查 same_state。
宿主单调时间只在同一时钟域比较；跨进程/机器时需由观测桥定义时间语义。

`RobotAction.validate_against(observation)` 验证状态、全部手臂、命令模式、frame 和关节数。
实际执行前宿主必须调用它；关节限位、碰撞、EE/TCP 转换、夹爪极性和动作预算仍由后续
机器人 profile、adapter 和 executor 校验。动作 contract 不执行运动、不代表安全许可。

几何投影已实现，参见 GEOMETRY.zh-CN.md。
快照序列化/存储与基础定位已实现，参见 PERCEPTION.zh-CN.md。
HTTP wire 与模型服务入口已实现，见 SAM3.zh-CN.md。原生接口桥与基础随动证据也已实现，
见 REMOTE_RUNTIME.zh-CN.md；真实引擎、标定与通用任务关系验证尚未验证。
本批测试只证明本地输入约束，不证明 RoboDojo 相机已标定或 SAM3 已联通。
