# SAM3 分割服务与真实验收

Harness 安装 `.[sam3-client]` 只获得 PNG/HTTP 客户端，不在本机部署权重。
模型设备另行安装获授权的[官方 SAM3 源码](https://github.com/facebookresearch/sam3)和依赖，
固定实际源码 revision、依赖版本及本地 checkpoint SHA256；本项目不分发权重、不代替授权流程。

```sh
python -m pip install -e '.[sam3-service]'
embodied-sam3 --checkpoint /srv/models/sam3.pt --checkpoint-sha256 ACTUAL_64_HEX \
  --device cuda:0 --port 8114 --audit-log /srv/logs/sam3.jsonl
```

先验证 checkpoint hash，禁止隐式 HF 下载。构造 SAM3 segmentation+instance interactive 模型，
按官方 detector/tracker 键映射严格加载权重；缺键/多键/格式不兼容则失败，不能用随机初始化的部分模型继续。
模型 `.eval()`，inference_mode/autocast，单进程串行推理；先加载成功才监听 HTTP。
model_version=`sam3:code:<SHA256>:checkpoint:<SHA256>`；code hash 覆盖实际安装包的 .py/.gz 文件，
包括 tokenizer 资源。它是部署指纹，不是对来源/许可证的密码学认证，仍需操作员核对官方 revision 与权重许可。

文本：Sam3Processor.set_image + set_text_prompt。点提示：同一官方图像特征的 predict_inst，
一个原图像素正点、multimask。接口核对日期 2026-10-29：
[builder](https://github.com/facebookresearch/sam3/blob/main/sam3/model_builder.py)、
[processor](https://github.com/facebookresearch/sam3/blob/main/sam3/model/sam3_image_processor.py)、
[predict_inst](https://github.com/facebookresearch/sam3/blob/main/sam3/model/sam3_image.py)。

服务返回原图分辨率二值 mask、有限0..1分数与紧包围框；最多5个候选。空结果合法，缺字段/尺寸错误不合法。
wire 校验 request ID、RGB SHA256、PNG尺寸、mask/box一致性；client 可锁定完整 model_version，
不自动重试，禁重定向与环境代理。图像不从 URL 拉取；服务不接触 simulator 真值。
默认 loopback 绑定；跨设备通过 tunnel 或受控网络。请求/推理并发、字节、尺寸、超时均有限额。
服务 audit 记录输入 hash、提示、路由、模型指纹、耗时和候选数，Harness 保存实际 mask 与快照关联。

真实验收使用目标设备捕获的公开 RGB-D 快照：

```sh
embodied-harness check-sam3 --url http://127.0.0.1:8114 \
  --store /srv/run/snapshots --episode EPISODE --snapshot SNAPSHOT --camera CAMERA \
  --query 'public object description' --checkpoint-sha256 ACTUAL_64_HEX --log /srv/logs/acceptance.jsonl
```

命令检查 /ready 的权重/代码指纹、文本分割、从文本候选选一个前景像素再测点提示、RGB-D反投影，
保存可核验报告。它不自动判定物体语义正确，也不证明相机标定或物理任务成功；须目视核对 mask 与目标，
再验证米制 depth、K、世界光学外参、遮挡和多候选消歧。正式定位使用 Perception.localize_object，
多候选不默认选最大/第一项，需显式选择；选定候选分数低于宿主阈值（默认0.5）返回low_confidence，
低深度覆盖或几何歧义保留部分结果并handoff。

本机测试用 CPU fixture 验证协议与 backend 分支，不冒称真实 SAM3 推理。
实际官方 checkpoint 在目标 GPU 的严格加载、文本/点分割、显存/延迟仍是部署验收项。
