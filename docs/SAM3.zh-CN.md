# SAM3 客户端与模型服务

已实现版本化 wire、标准库 HTTP 客户端、独立 HTTP 服务及懒加载 SAM3 backend。
与 ASPIRE 的旧 wire 不兼容；内部仍通过 SegmentationProvider 接入定位流水线。

## 安装与启动

客户端环境（不安装模型依赖）：

```sh
python -m pip install -e '.[sam3-client]'
```

模型环境需要另行安装官方 SAM3 源码及其依赖，准备获授权的本地 checkpoint；
安装本项目 extra 不会自动下载 SAM3 源码或权重。记录实际 SAM3 revision 与依赖版本。
本轮没有建立可复现的 GPU 依赖锁或自动 provisioning。

```sh
python -m pip install -e '.[sam3-service]'
embodied-sam3 --checkpoint /absolute/path/to/sam3.pt --device cuda:0 --port 8114
```

backend 根据官方 build_sam3_image_model(checkpoint_path, device, load_from_HF,
enable_inst_interactivity)、Sam3Processor 和 predict_inst 接口实现；加载一次、eval、
inference_mode，模型内部预处理由官方 processor 完成。传输保持原始 PNG 分辨率，
分割结果须回到原图尺寸。checkpoint SHA256 进入 model_version；该字段当前不包含源码 revision，
部署记录仍须另外保存 revision。加载在 HTTP 监听前完成，加载失败则启动失败。

核对来源（2026-10-29）：
[官方 model_builder](https://github.com/facebookresearch/sam3/blob/main/sam3/model_builder.py)、
[官方 processor](https://github.com/facebookresearch/sam3/blob/main/sam3/model/sam3_image_processor.py)、
[官方 predict_inst](https://github.com/facebookresearch/sam3/blob/main/sam3/model/sam3_image.py)。
接口已核对，真实权重/GPU 执行尚未验证。

## 接入已有快照

```python
from embodied_harness.artifacts import SnapshotStore
from embodied_harness.perception import Perception
from embodied_harness.providers.sam3 import Sam3Provider

store = SnapshotStore("runs/my-run/snapshots")
provider = Sam3Provider("http://127.0.0.1:8114", timeout_s=30)
pipeline = Perception(store, provider)
result = pipeline.localize_object("episode-id", "snapshot-id", "cam_head", "white bowl")
```

client 同时支持 `segment_point(observation, camera_id, (x, y))`，只接受一个图内正点。
点提示结果的本地 prompt 为 `point:x,y`；高层 Perception 当前仅提供文字定位。
客户端禁止环境代理和重定向，不做自动重试；本地服务默认只监听 127.0.0.1。

## 协议与资源

- GET `/health`：进程存活；GET `/ready`：backend 已加载，否则 503。
- POST `/v1/segment/text`、`/v1/segment/point`：schema_version=1、request_id、
  input_rgb_sha256、image_png_base64、prompt。请求只含图像与提示，不含机器人状态。
- 返回相同 request_id/图像哈希，以及 provider、model_version、candidates。
  mask 为原图大小的 uint8 0/1 base64，框为 XYXY exclusive，最多 5 个候选。
- client 用原请求上下文补上 ObservationSource/camera_id；拒绝跨请求、跨图像、
  错误尺寸、非二值 mask 和非法分数。空候选是有效推理响应，由定位层返回 NO_CANDIDATES。
- 请求/响应上限 16 MiB、图像上限 1568×1568 像素总数；不自动裁剪或缩放。
- 同时最多接受两项 HTTP 请求，最多一项模型推理，额外请求 503。
  排队及 socket 读有超时；CUDA 推理本身没有硬中止机制，客户端超时不证明推理已取消。
  health/ready 也受总请求限额影响。此服务没有认证或 TLS，远程部署需宿主提供访问边界。

当前 HTTP 错误 429/503 映射 UNAVAILABLE，其他 HTTP 错误映射 INVALID_RESPONSE；
推理异常返回 500，不伪装成空候选。原始异常和模型耗时账本尚未接 recorder，
生产诊断和自动进程恢复仍待实现。

## 本轮测试

新增 13 项 CPU 测试，总计 52 项：

| 测试 | 验证内容 |
| --- | --- |
| PNG/提示往返 | 原始 RGB 不变，点坐标边界、负值、NaN 与布尔值拒绝 |
| 非法请求 | 错误版本、图像哈希、无效 base64、额外字段拒绝 |
| 响应绑定 | request_id/哈希/版本、mask 尺寸和字节、候选数量检查 |
| 真实本机 HTTP 链路 | client→HTTP server→模拟 backend→分割 contract→快照定位/几何 |
| health/ready | 服务存活与模型就绪分开，未就绪返回 UNAVAILABLE |
| HTTP 无效输入 | 缺失请求字段返回 400 |
| 推理失败 | 模拟 backend 抛错返回失败，不变成成功 |
| 空/错误尺寸输出 | 空预测保留空候选，错误 mask 尺寸拒绝 |
| 并发限额 | 槽位耗尽时返回服务不可用 |
| 客户端错误 | 模拟超时、拒绝连接、畸形 JSON/响应身份及非法参数 |
| 依赖隔离 | 导入客户端不加载 torch、sam3、Pillow 或 simulator |

其中 HTTP 测试会临时绑定本机端口；需要允许 loopback 网络。模型是明确标记的
FixtureBackend，没有真实 SAM3、torch 或 GPU 推理，因此不能据此声称识别能力已验证。
