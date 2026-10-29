# Geometry：接口与测试范围

实现：`embodied_harness.geometry`。本批代码为新实现，仅依赖 contracts 与 NumPy。

```python
from embodied_harness.geometry import mask_to_point_cloud, estimate_geometry

# observation 和 segmentation 来自同一快照；此示例不采图、不调用模型。
projection = mask_to_point_cloud(observation, segmentation, target_frame="world")
if projection.ok:
    geometry = estimate_geometry(projection.data.cloud)
    # geometry.ok 仅表示数值估计成功；不是物体身份或操作完成的证据。
```

`mask_to_point_cloud` 首先校验分割的完整观测来源、camera_id、mask 形状和 RGB 哈希。
使用光轴深度反投影，光学轴为右/下/前。输出 frame 可为 `world` 或
`camera/<camera_id>/optical`；其他坐标系必须显式调用 `transform_point_cloud`，
提供 source_frame、target_frame 与 T_target_from_source。

过滤规则：深度必须有限且严格大于 min_depth_m（默认 0）；若指定 max_depth_m，
则允许等于最大值。返回 Projection，包含点云、掩码像素数、有效深度数及比例。
无有效深度返回 NO_VALID_DEPTH，仍保留空点云和覆盖率；错误配置或混用快照抛 ValueError。
暂未实现边缘腐蚀、背景混合剔除、统计离群过滤或质量阈值决策，后续由感知流水线加入。

`estimate_geometry` 返回可见表面 PCA 包围盒的 center、axes（列为轴）、extents、来源及点数。
少于三点或点/线退化返回 GEOMETRY_FAILED；平面和近似重复特征值标记 orientation_ambiguous。
平面包围盒允许零厚度。当前歧义阈值是数值诊断：最小特征值 <= 最大值的 1e-6，
或相邻特征值差 <= 最大值的 5%；不代表经过真实任务标定的置信度。
轴符号采用确定性规则，即使 ambiguity 为 False，也不提供物体语义朝向或完整 6D 真值。

## 本轮实际测试

执行 `PYTHONPATH=src python -m unittest discover -s tests -v`：24 项通过，
包含之前 13 项 contract 测试与新增的 11 项 geometry 测试。

| 新增测试 | 检查什么 |
| --- | --- |
| 已知像素反投影 | 用手算坐标检查焦距、主点、深度和光学轴方向 |
| 旋转平移与逆变换 | 检查 world-from-camera 方向及往返一致性 |
| 像素回投影 | 三维点重新投影后恢复原始像素网格 |
| 掩码和深度过滤 | 掩码外像素、NaN/Inf/非正值、深度上下限和覆盖率 |
| 全部深度无效 | 返回明确错误、空云和零覆盖率，保留失败证据 |
| 不同快照/reset | 阻止分割与另一状态的深度混用 |
| 当前相机变换 | 相机变换更新后点云随之改变，不使用旧外参 |
| 非法 frame/变换/范围 | 拒绝未知 frame、源 frame 不符、缩放、反射及非法深度配置 |
| 已知旋转长方体 | 恢复中心和边长、右手正交轴，且包围盒包含所有输入点 |
| 对称/平面点云 | 标记方向歧义，避免静默返回确定方向 |
| 空云/单点/重复点/直线 | 返回几何失败，避免构造虚假物体位姿 |

这些都是合成 CPU 数值测试，未运行 SAM3、Isaac Sim 或真实 RoboDojo。
尚不能证明原生相机外参正确、分割识别准确或可完成抓取。
