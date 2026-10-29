# 快照存储与定位流程

已实现 `artifacts.SnapshotStore` 和 `perception.Perception`，仅依赖 NumPy 与标准库。
SAM3 客户端和模型服务入口已实现，参见 SAM3.zh-CN.md；真实模型运行尚未验证。

```python
from embodied_harness.artifacts import SnapshotStore
from embodied_harness.perception import Perception

store = SnapshotStore("runs/my-run/snapshots")
store.put(observation)  # observation 由后续 adapter 产生
pipeline = Perception(store, provider)  # provider 实现 SegmentationProvider
result = pipeline.localize_object(
    observation.source.episode_id, observation.source.snapshot_id,
    "head", "white bowl",
)
if not result.ok:
    print(result.error.code, result.error.message)
    # result.data 可保留候选、点云或深度覆盖率，不能丢弃这些失败证据。
```

## 存储边界

一个快照目录包含版本化 JSON manifest 和 NPZ 数组，读回后重建并校验完整 contract。
不使用 pickle，保留浮点深度、标定、机器人状态和来源；NPZ 的 SHA256 用于检测意外损坏，
不是对恶意修改者的签名验证。目录名由 episode_id/snapshot_id 的哈希生成，外部 ID 不作路径。

临时目录写完才 rename 提交；同 ID 已存在时失败，不自动覆盖或淘汰历史证据。
写入错误向上传递并清理临时文件。没有 fsync 掉电持久性保证。
默认限制单文件及展开数组为 256 MiB、最多 1000 份快照；这些是可配置容量限制。
当前用于单宿主写入、受宿主权限保护的本地存储，不提供多进程配额事务或远程访问权限系统。
根目录就是 episode 范围的访问边界，代理不应直接得到整个存储目录。

## 定位边界

按 episode + snapshot ID 读取一次观测，把它传给 provider，不重新取帧、不执行动作。
每个候选须通过来源/相机/RGB 哈希/提示校验。provider 返回错误时保留有效部分候选；
非法响应返回 INVALID_RESPONSE，原始不可信对象不进入后续几何计算。

零候选返回 NO_CANDIDATES；多个候选默认返回 AMBIGUOUS 并保留全部候选。
调用方可通过 candidate_index 明确选择，但这个索引不是稳定物体身份；查看证据后必须
针对同一快照选择，不能拿旧快照的索引套到新快照。

选定实例后计算 world 点云和几何。无有效深度返回 NO_VALID_DEPTH；有效深度比例低于
min_valid_depth_fraction 返回 INSUFFICIENT_DEPTH（默认 0.5，是工程门槛，未做物理效果标定）。
几何退化返回 GEOMETRY_FAILED，保留已生成点云。方向歧义保留在 geometry 中，
不等同于定位调用失败，也不能用来宣称方向确定。

Localization 返回候选、选择索引、Projection 和 ObjectGeometry。本批持久化的是输入快照，
Localization 仍是内存结果；分割/定位结果持久化、叠加图、服务调用账本在后续 recorder 接入。

## 测试测了什么

本批新增 15 项测试，合计 39 项 CPU 测试。

| 范围 | 检查内容 |
| --- | --- |
| 完整快照往返 | RGB、NaN 深度、K、相机外参、关节、位姿、指令、来源和数组不可变 |
| 证据不可覆盖 | 同快照 ID 写入失败，原记录不改变 |
| 路径与身份 | 带 ../ 的 ID 仍留在存储内，不同 episode 不串读 |
| 损坏检测 | 数组被修改、错误 schema 或 manifest 身份不符均拒绝 |
| 容量 | 写入字节限制、快照数量限制、读取文件大小限制 |
| 失败写入 | 模拟磁盘写错后，不留下可被误读的完整快照或临时目录 |
| 定位端到端 | 保存→读取→fixture provider→同帧点云→几何，来源保持一致 |
| 候选歧义 | 多候选不自动选择；显式选择低分候选也按指定索引执行 |
| 无候选/无 provider | 明确返回对应错误 |
| provider 失败 | 保留部分候选，不继续生成几何 |
| provider 非法数据 | 拒绝错误返回类型、错误候选、旧快照、错误相机/图像/提示 |
| 深度不足 | 区分零有效深度与覆盖率不足，均保留测量 |
| 几何退化 | 共线点返回失败，并保留点云 |
| 非法调用 | 无效参数和不存在的相机不调用 provider |

测试中的 FakeProvider 只是协议 fixture，没有模型推理，也没有仿真动作。
尚未验证真实 RoboDojo 观测桥、SAM3 识别、相机标定和任何操作成功率。
