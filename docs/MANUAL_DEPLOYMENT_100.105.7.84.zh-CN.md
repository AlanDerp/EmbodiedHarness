# EmbodiedHarness 手动部署与实验启动教程

适用服务器：`xuanzhi@100.105.7.84`。整理日期：2026-10-30。
本教程基于已经上传的代码和实际检查结果；后续命令由你手动执行。

**先完成第1～4步。它们不提交机器人动作，可以判断真实环境和模型是否工作。第6步以后的标定、System 2驱动和凭据未完成前，先不要启动campaign。**

## 0. 当前已有的东西

| 项目 | 已知状态 |
| --- | --- |
| 部署根目录 | `/home/xuanzhi/embodied-harness-deploy-20261030` |
| 项目源码 | 根目录下的 `EmbodiedHarness` |
| 专用Python | 根目录下的 `venv/bin/python` |
| RoboDojo | `/home/xuanzhi/RoboDojo` |
| GPU | 约24GB显存，驱动580.65.06 |
| 依赖 | Python3.11.17、Torch2.7.0+cu128；已有IsaacLab、CuRobo和SAM3 |
| 本地权重 | `/home/xuanzhi/.cache/aspire-robodojo/sam3.pt` |
| 代码测试 | 在服务器通过128项，含真实子进程/HTTP，模型和物理使用fixture |
| 真实SAM3 | 曾严格加载成功，监听`127.0.0.1:8114`；本教程重新检查当前状态 |
| Isaac探针 | 第二次启动请求已执行，但结果尚未取回；不是已通过验收 |
| profile / S1 / S2 | 尚未完成整体就绪核验 |

不需要重新下载源码、模型或重建共享Conda环境。以下使用已有专用venv；暂不升级Torch。
不要把测试fixture当作SAM3，也不要把128项测试通过当作真实物理实验成功。

## 1. 登录，设置每个终端的公共环境

**在你的电脑终端执行：**

```bash
ssh xuanzhi@100.105.7.84
```

**从这里开始，除明确标注外，命令均在这台服务器执行。**

```bash
export EH_ROOT=/home/xuanzhi/embodied-harness-deploy-20261030
export EH_REPO="$EH_ROOT/EmbodiedHarness"
export EH_PY="$EH_ROOT/venv/bin/python"
export PATH="$EH_ROOT/venv/bin:$PATH"
export OMNI_KIT_ACCEPT_EULA=YES
cd "$EH_REPO"

hostname
"$EH_PY" -c 'import sys, embodied_harness; print(sys.executable); print(embodied_harness.__file__)'
embodied-harness --help
nvidia-smi
```

预期：hostname是`peilab-System-Product-Name`；Python和Harness来自上述部署目录；CLI包含
`native-worker`、`provisioner`、`agent-service`、`doctor`、`campaign`等命令。

你已经明确接受Omniverse许可，故这里设置`OMNI_KIT_ACCEPT_EULA=YES`。它必须由启动Isaac的进程继承。
每开一个新SSH终端都重新执行本步的环境设置。不要在全局环境里盲目安装或升级依赖。

## 2. 先读取已经启动的Isaac探针结果

```bash
"$EH_PY" -m json.tool "$EH_ROOT/native-probe-report-02.json"
```

关注：

- `passed: true`：完成了reset和公开观测捕获。
- `closed: true`：本次检查正常关闭了原生环境。
- `native_action_limit`：原生任务的正数预算。
- `native_action_count: 0`：本探针没有调用动作提交。
- `cameras`：各相机的RGB/depth/K/E形状及有效深度比例。

只有报告中`passed`和`closed`都为true，才把这一关记为通过。`success: false`在初始场景是正常结果，
不需要为了让它变true而发送动作。

如果文件不存在，先读日志并检查进程：

```bash
tail -n 80 "$EH_ROOT/logs/native-probe-02.log"
"$EH_PY" - <<'PY'
import json, os, subprocess
from pathlib import Path
record = Path(os.environ['EH_ROOT']) / 'native-process-02.json'
if record.exists():
    pid = json.loads(record.read_text())['pid']
    subprocess.run(['ps', '-p', str(pid), '-o', 'pid,etime,stat,args'])
else:
    print('没有第二次探针进程记录')
PY
```

第一次启动可能耗时较长。日志持续推进时先等候；不要重复启动同一探针。
若进程仍在且日志长期不动，把末尾日志保留下来分析。不要用`pkill python`、`pkill isaac`等命令清理整台共享服务器。

若进程已退出但没有报告，保留日志，当前判为失败；不要删目录假装没运行过。
如果报告为`passed:false`，读取`error_type`、`error`和`traceback`。

| 结果/日志 | 下一步 |
| --- | --- |
| 仍在询问EULA | 确认启动该进程的终端已export本教程第1步变量 |
| CUDA out of memory | 先检查`nvidia-smi`，确认是否有其他任务；不要直接杀其他进程 |
| 模块导入失败 | 使用上面的`EH_PY`；不要随意切到系统Python |
| Renderer / Vulkan / PhysX报错 | 保留完整日志，先处理引擎启动；不进入动作实验 |
| `FileExistsError` | 重用了已有run_dir；新检查使用新目录，旧证据保留 |

### 确认旧探针已退出后，如何重新做一次零动作检查

下面从已有`native-probe.py`生成带唯一编号的新脚本，前台运行，便于观察。它仍只测试
`align_blocks`、scene_ref=0的reset/capture/close，不发动作。

```bash
"$EH_PY" - <<'PY'
import os, time, uuid
from pathlib import Path
root = Path(os.environ['EH_ROOT'])
tag = time.strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6]
source = (root / 'native-probe.py').read_text()
source = source.replace('native-probe-01', 'native-probe-' + tag)
source = source.replace('native-public-01', 'native-public-' + tag)
source = source.replace('native-probe-report.json', 'native-probe-report-' + tag + '.json')
script = root / ('native-probe-' + tag + '.py')
script.write_text(source)
(root / 'manual-last-probe.txt').write_text(str(script))
print('本次脚本:', script)
print('报告:', root / ('native-probe-report-' + tag + '.json'))
print('公开图像:', root / ('native-public-' + tag))
PY

EH_PROBE=$(cat "$EH_ROOT/manual-last-probe.txt")
set -o pipefail
"$EH_PY" -u "$EH_PROBE" 2>&1 | tee "${EH_PROBE%.py}.log"
```

脚本会捕获异常并写报告，因此**即使shell退出码为0，也必须查看JSON中的passed/closed**。
如果使用这次新探针，后面凡提到`native-public-02`的地方，都改成它打印的实际公开图像目录。

## 3. 检查SAM3服务，必要时手动启动

```bash
curl --fail --silent --show-error http://127.0.0.1:8114/ready
```

预期包含`ready:true`，已部署权重对应的指纹是：

```text
sam3:code:267ca11c97ecba8455e519d720b74c96cfadb02f87429629d53c6b7c9d459757:checkpoint:9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e
```

若当前服务正常，不要再启动第二份。若指纹不同，先核对代码/权重变更，不要为了通过检查直接把配置改成新值。

如果连接失败：

```bash
tail -n 80 "$EH_ROOT/logs/sam3.log"
ss -ltn | grep ':8114'
```

只有确认端口没有被占用后，打开一个专门的SSH终端，执行第1步环境设置，再前台启动：

```bash
export EH_CHECKPOINT_SHA=9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e
embodied-sam3 \
  --checkpoint /home/xuanzhi/.cache/aspire-robodojo/sam3.pt \
  --checkpoint-sha256 "$EH_CHECKPOINT_SHA" \
  --device cuda:0 \
  --host 127.0.0.1 \
  --port 8114 \
  --audit-log "$EH_ROOT/logs/sam3-audit.jsonl"
```

保持这个终端打开，另开终端检查`/ready`。严格加载失败时停在这里，不绕过hash检查或缺键检查。
`/ready`通过只证明模型已加载，下一步才检查真实图像分割。

## 4. 用Isaac公开图像验证真实SAM3文本与点分割

本步不依赖已commissioned的profile，不进行三维动作。先从第2步报告确认公开图像目录，执行：

```bash
ls "$EH_ROOT/native-public-02"
```

如果本次新探针使用别的目录，把下方路径改成实际目录。从其中选择一个确实有目标的相机PNG。
可以通过你的SFTP客户端下载PNG查看。以下两个变量需要按实际图像填写：

```bash
export EH_IMAGE="$EH_ROOT/native-public-02/实际相机名.png"
export EH_QUERY='根据图像填写具体可见物体的英文描述'
```

下面通过真实HTTP协议检查文本和点提示，不构造假depth或假相机位姿。结果输出到新的验收目录，
包含文本mask与overlay，便于目视检查。

```bash
"$EH_PY" - <<'PY'
import base64, hashlib, io, json, math, os, time, urllib.request, uuid
from pathlib import Path
import numpy as np
from PIL import Image
from embodied_harness.providers.sam3.wire import TEXT_PATH, POINT_PATH, MAX_BODY_BYTES, encode_candidate

image = Image.open(os.environ['EH_IMAGE']).convert('RGB')
rgb = np.asarray(image)
stream = io.BytesIO()
image.save(stream, format='PNG')
url = 'http://127.0.0.1:8114'
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open(url + '/ready', timeout=10) as response:
    ready = json.loads(response.read(MAX_BODY_BYTES + 1))
expected = ('sam3:code:267ca11c97ecba8455e519d720b74c96cfadb02f87429629d53c6b7c9d459757:'
            'checkpoint:9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e')
assert ready.get('ready') is True and ready.get('model_version') == expected, ready
out = Path(os.environ['EH_ROOT']) / ('sam3-manual-' + time.strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6])
out.mkdir()

def infer(route, prompt, label):
    payload = {'schema_version': 1, 'request_id': uuid.uuid4().hex,
               'input_rgb_sha256': hashlib.sha256(rgb.tobytes()).hexdigest(),
               'image_png_base64': base64.b64encode(stream.getvalue()).decode(), 'prompt': prompt}
    body = json.dumps(payload, allow_nan=False).encode()
    assert len(body) <= MAX_BODY_BYTES
    request = urllib.request.Request(url + route, body, headers={'Content-Type': 'application/json'})
    with opener.open(request, timeout=180) as response:
        raw = response.read(MAX_BODY_BYTES + 1)
    assert len(raw) <= MAX_BODY_BYTES
    result = json.loads(raw)
    assert result['schema_version'] == 1 and result['provider'] == 'sam3'
    assert result['request_id'] == payload['request_id']
    assert result['input_rgb_sha256'] == payload['input_rgb_sha256']
    assert result['model_version'] == expected
    assert isinstance(result['candidates'], list) and len(result['candidates']) <= 5
    masks = []
    for i, item in enumerate(result['candidates']):
        assert item['shape'] == list(rgb.shape[:2])
        values = np.frombuffer(base64.b64decode(item['mask_base64'], validate=True), dtype=np.uint8)
        assert values.size == rgb.shape[0] * rgb.shape[1] and np.all(values <= 1)
        mask = values.reshape(rgb.shape[:2]).astype(bool)
        assert math.isfinite(item['score']) and 0 <= item['score'] <= 1
        assert encode_candidate(mask, item['score'])['box_xyxy'] == item['box_xyxy']
        Image.fromarray(mask.astype('uint8') * 255).save(out / f'{label}_{i}_mask.png')
        overlay = rgb.copy()
        overlay[mask] = (.5 * rgb[mask] + .5 * np.array([255, 0, 0])).astype('uint8')
        Image.fromarray(overlay).save(out / f'{label}_{i}_overlay.png')
        masks.append(mask)
    (out / (label + '.json')).write_text(json.dumps(result))
    print(label, 'candidates=', len(masks), 'scores=', [x['score'] for x in result['candidates']])
    return masks

masks = infer(TEXT_PATH, os.environ['EH_QUERY'], 'text')
assert masks, '文本未找到候选；检查图像和公开提示，不能当作通过'
pixels = np.argwhere(masks[0])
y, x = pixels[len(pixels) // 2]
points = infer(POINT_PATH, [int(x), int(y)], 'point')
assert points, '点提示没有候选，检查服务日志'
print('服务文本/点分支检查通过。请目视核对overlay；三维定位还未验收。结果目录:', out)
PY
```

通过条件：两个分支都有合法原图mask；overlay确实覆盖你描述的物体。
无候选时，可以检查图像和修改描述；服务500时读`logs/sam3.log`与`sam3-audit.jsonl`，不要换fixture冒充通过。
多候选时需判断目标身份；分数低于0.5不能作为当前宿主默认接触执行依据。

## 5. 完成机器人profile：这是进入动作实验前的停点

当前模板还不能直接运行。先复制一份到部署目录，避免修改仓库模板：

```bash
mkdir -p "$EH_ROOT/manual/config"
cp -n configs/arx_x5.profile.template.json "$EH_ROOT/manual/config/arx_x5.profile.json"
nano "$EH_ROOT/manual/config/arx_x5.profile.json"
```

按下表填写实际值。**不是把commissioned改为true就算完成。**

| 字段 | 如何确认 |
| --- | --- |
| `joint_names` | 与实际公开proprioception顺序逐项对应；已有测量显示两臂各为joint1～joint6，仍核对本次runtime |
| `joint_lower/upper` | 核对当前机器人配置/URDF及控制器有效限位；±10rad之类宽URDF范围不自动等于安全运行范围 |
| `workspace_lower/upper` | 按世界坐标、米制设置经检查的保守可达区域，涵盖本次开发动作；不要沿用模板的±1 |
| `T_ee_from_tcp` | 依据末端link6和实际选定TCP计算/验证刚体变换；夹爪配置中的标量bias不能直接当通用TCP变换 |
| `camera_convention` | 确认当前原生E矩阵的轴定义；验证后选`opengl_verified`或`optical_verified` |
| `calibration_id` | 指向你保存的实际标定记录及版本 |
| `commissioned` | 上述信息确认并记录后再设true |

已有设备材料可作为核对依据：

```text
/home/xuanzhi/robodojo-device-identification-20261008/calibration_retry1/device_metadata.json
/home/xuanzhi/robodojo-device-identification-20261008/calibration_retry1/MEASURED_DEVICE_PROFILE.json
```

后一个文件不是Harness的RobotProfile格式，不能直接替换。它只覆盖局部小角度测量，不证明抓取或全工作空间。
不要把旧任务解题代码或评分信息交给模型作为标定材料。

相机验证可使用第2步保存的`public-arrays.npz`：检查depth为米制光轴深度，K匹配图像尺寸；
将多个可见平面像素反投影后核对高度、距离、左右方向及不同相机的一致性。
OpenGL camera-parent E只有在确认轴转换后才使用`E @ diag(1,-1,-1,1)`。
确认TCP定义后，用公开机器人配置/正运动学或受控测量核对其位置与轴方向；动作标定要单独限定范围和预算。
**这些真实标定尚未替你完成；没有合格profile时可以停在第4步，不启动动作实验。**

另一个需要记住的原生语义：`ee_joint_state`可能是归一化夹爪控制目标，不是实测指尖间距；
不能用它单独证明夹住物体或任务成功。

完成填写后检查schema：

```bash
embodied-harness validate-profile "$EH_ROOT/manual/config/arx_x5.profile.json"
```

预期`schema_valid:true`、`commissioned:true`。这只是schema/声明检查，不能替代上述测量。
若无法确认某字段，保留false并记录缺项，不填虚构数值。

## 6. 启动一个worker，只读取真实快照并验收RGB-D链路

### 6.1 创建一次性的本地RPC令牌文件

在服务器上执行一次。以后各终端source同一个文件；不要每个终端重新生成不同token。

```bash
"$EH_PY" - <<'PY'
import os, secrets
from pathlib import Path
path = Path(os.environ['EH_ROOT']) / 'manual/tokens.env'
if path.exists():
    print('已有token文件，继续复用:', path)
else:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as stream:
        for key in ('EMBODIED_WORKER_TOKEN', 'EMBODIED_PROVISIONER_TOKEN', 'EMBODIED_AGENT_TOKEN'):
            stream.write('export ' + key + '=' + secrets.token_hex(32) + '\n')
    print('已创建:', path)
PY
source "$EH_ROOT/manual/tokens.env"
```

不要把token文件提交Git或贴进聊天。服务端和调用端必须拿到对应的同一个token。

### 6.2 新建worker运行配置

确认第2步探针已退出。下面只为独立验收选择align_blocks的dev布局0，不是正式评估结果。

```bash
"$EH_PY" - <<'PY'
import json, os, time, uuid
from pathlib import Path
root = Path(os.environ['EH_ROOT'])
tag = time.strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6]
config = {'runtime_root': '/home/xuanzhi/RoboDojo', 'task': 'align_blocks', 'scene_ref': 0,
          'run_dir': str(root / 'manual' / ('worker-native-' + tag))}
path = root / 'manual/config' / ('worker-' + tag + '.json')
path.write_text(json.dumps(config, indent=2))
(root / 'manual/last-worker-config.txt').write_text(str(path))
print(path)
PY
```

**终端W：**完成第1步和source tokens，然后运行以下命令，保持前台：

```bash
source "$EH_ROOT/manual/tokens.env"
EH_WORKER_CONFIG=$(cat "$EH_ROOT/manual/last-worker-config.txt")
embodied-harness native-worker \
  --config "$EH_WORKER_CONFIG" \
  --profile "$EH_ROOT/manual/config/arx_x5.profile.json" \
  --episode-id manual_rgbd_acceptance \
  --phase playground \
  --host 127.0.0.1 --port 8120 \
  --token-env EMBODIED_WORKER_TOKEN
```

**终端C：**另开SSH，完成第1步、source tokens，读取一次快照：

```bash
source "$EH_ROOT/manual/tokens.env"
"$EH_PY" - <<'PY'
import json, os
from pathlib import Path
from embodied_harness.adapters.robodojo import RemoteRoboDojo
from embodied_harness.artifacts import SnapshotStore
from embodied_harness.rpc import RpcClient
root = Path(os.environ['EH_ROOT'])
rpc = RpcClient('http://127.0.0.1:8120', os.environ['EMBODIED_WORKER_TOKEN'], timeout_s=240)
adapter = RemoteRoboDojo(rpc)
print('worker:', adapter.metadata)
obs = adapter.observe()
store = SnapshotStore(root / 'manual/snapshots')
store.put(obs)
index = {'store': str(store.root), 'episode': obs.source.episode_id, 'snapshot': obs.source.snapshot_id,
         'cameras': list(obs.cameras), 'instruction': obs.instruction}
(root / 'manual/last-snapshot.json').write_text(json.dumps(index, indent=2))
print(json.dumps(index, indent=2))
for name, camera in obs.cameras.items():
    print(name, 'RGB=', camera.rgb.shape, 'depth=', camera.depth_m.shape)
PY
```

出现字段/shape错误时停下，保存错误及worker日志，这是实际兼容性问题；不要改fixture蒙混过去。

### 6.3 SAM3同帧快照验收

从上一步`last-snapshot.json`选择实际相机名，按图像填写query：

```bash
export EH_CAMERA='填写实际相机名'
export EH_QUERY='填写目标物体的公开英文描述'
"$EH_PY" - <<'PY'
import json, os, subprocess
from pathlib import Path
root = Path(os.environ['EH_ROOT'])
index = json.loads((root / 'manual/last-snapshot.json').read_text())
assert os.environ['EH_CAMERA'] in index['cameras']
subprocess.run(['embodied-harness', 'check-sam3', '--url', 'http://127.0.0.1:8114',
               '--store', index['store'], '--episode', index['episode'], '--snapshot', index['snapshot'],
               '--camera', os.environ['EH_CAMERA'], '--query', os.environ['EH_QUERY'],
               '--checkpoint-sha256', '9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e',
               '--log', str(root / 'manual/sam3-rgbd-acceptance.jsonl')], check=True)
PY
```

预期输出`passed:true`。它覆盖文本、点提示和深度投影；仍需目视检查语义和坐标是否正确。
检查完在**终端W按Ctrl+C**关闭worker。确认已退出后再进入provisioner，避免两份Isaac同时占用GPU。

## 7. 接通System 2：需要真实模型驱动

**这一点必须说清楚：当前代码已有角色服务、Plan协议和提示词，但没有绑定某家模型API的生产驱动。**
`agent-service --factory my_model_driver:create`里的名称是接入点，不是已经存在的模块。
仅设置OPENAI_API_KEY或其他key不会自动产生这个驱动。

先确定你使用的System 2模型/服务，再准备一个可导入的模块。它须满足：

```text
create() -> 新对话driver

driver.respond(role, instructions, context) -> 解析后的Python dict
可选driver.close()
```

需要实现的行为：

1. 使用`instructions`中的角色契约；把context中的实际RGB作为图像输入，不能只发送numpy数组字符串。
2. `generate/repair/verify`返回Plan：`{goal, subtasks:[{task_id,goal,source,parameters,relation}]}`。
3. `propose`返回`{skills:{...}, skill_report:[...]}`；review返回新鲜snapshot证据。
4. 每次create建立新模型对话；独立verifier不能继承proposer历史。
5. 仅使用公开观测/API和可见技能库；不接触RoboDojo私有布局/原生评分。

完整契约：`$EH_REPO/src/embodied_harness/instructions/fix_loop.md`。
**如果没有这个驱动，先停在这里，记录所用模型和API形式，不能启动完整campaign。**

驱动完成并安装到运行System 2服务的Python后，设置真实factory，例如`你的包.模块:create`：

```bash
export EH_S2_FACTORY='替换为真实可导入模块:create'
"$EH_PY" - <<'PY'
import importlib, os
module, function = os.environ['EH_S2_FACTORY'].split(':', 1)
factory = getattr(importlib.import_module(module), function)
assert callable(factory)
print('factory可导入；尚未证明真实多模态生成可用')
PY
```

在System 2所在设备的专用终端配置其实际模型凭据，载入相同的agent token，再运行：

```bash
embodied-harness agent-service \
  --factory "$EH_S2_FACTORY" \
  --host 127.0.0.1 --port 8123 \
  --token-env EMBODIED_AGENT_TOKEN
```

如果System 2在另一台设备，那里也需要安装本项目的同一版本、实际driver和依赖。
在RoboDojo服务器另开一个终端建立隧道，替换`USER@SYSTEM2_HOST`为真实地址：

```bash
ssh -N -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:8123:127.0.0.1:8123 USER@SYSTEM2_HOST
```

隧道运行在RoboDojo服务器上，campaign仍访问它本机的`127.0.0.1:8123`。
agent token需通过你选择的安全方式在服务端配置一致，不在命令行参数或聊天中贴出。
`describe`通过只代表角色服务存在；真正多模态生成要在开发smoke里验证。

## 8. 配置System 1

正式双循环使用你可访问的Jev model ID及TYPESAFE_API_KEY。**在运行campaign的终端**输入key，避免写进shell history：

```bash
read -r -s -p 'TYPESAFE_API_KEY: ' TYPESAFE_API_KEY
printf '\n'
export TYPESAFE_API_KEY
export EH_JEV_MODEL='替换成你的真实固定模型ID'
```

不要在不同终端只设置key却忘了campaign所在终端。教程不猜测你的模型ID。

可先独立验证当前selector与真实服务的协议；这一步会调用System 1 API，但不发机器人动作：

```bash
"$EH_PY" - <<'PY'
import os
from embodied_harness.runtime.selectors import JevSelector
selector = JevSelector(os.environ['EH_JEV_MODEL'])
answer = selector.select(
    {'position_error_m': 0.005, 'scope': 'numeric_selector_protocol_check_no_robot'},
    {'fine': 'translation_step_m = 0.001 m', 'normal': 'translation_step_m = 0.01 m'})
print(answer)
PY
```

预期返回合法选项和confidence。401/模型不存在/响应schema错误时先解决模型接入。
这个合成数字检查不证明机器人上的决策质量。
`fixed`可用于单独的工程基线，但不能把fixed运行记为已验证Jev双循环。

## 9. 启动provisioner，管理每场景的新worker

确认第6步的手动worker已关闭。**provisioner和campaign都在这台RoboDojo服务器运行**，
因为provisioner返回本机loopback worker地址。

生成当前机器的配置：

```bash
"$EH_PY" - <<'PY'
import json, os, sys
from pathlib import Path
root = Path(os.environ['EH_ROOT'])
config = {'runtime_root': '/home/xuanzhi/RoboDojo',
          'profile': str(root / 'manual/config/arx_x5.profile.json'),
          'root': str(root / 'manual/native-allocations'),
          'python_command': [sys.executable], 'worker_token_env': 'EMBODIED_WORKER_TOKEN',
          'startup_timeout_s': 600, 'rpc_token_env': 'EMBODIED_PROVISIONER_TOKEN',
          'host': '127.0.0.1', 'port': 8122}
path = root / 'manual/config/provisioner.json'
with path.open('x') as stream:
    json.dump(config, stream, indent=2)
print(path)
PY
```

这里使用已通过第2/6步的同一个Python入口。如果之前实际只有IsaacLab启动脚本能跑通，则把
`python_command`改成那个已验证的argv；不要只因模板写了`/srv/...`就复制不存在的路径。
`FileExistsError`表示已经生成过配置，检查/编辑现有文件，不要覆盖未知状态。

**终端P：**执行第1步、source tokens，保持前台运行：

```bash
source "$EH_ROOT/manual/tokens.env"
embodied-harness provisioner --config "$EH_ROOT/manual/config/provisioner.json"
```

它启动时不会立刻创建Isaac；只有campaign申请episode才会启动新native-worker。
worker继承provisioner终端的EULA与依赖环境。

## 10. 准备小规模开发smoke的任务、分区和冻结库

先选实际任务。初次联调可沿用已验收的`align_blocks`，再读取其真实布局清单：

```bash
export EH_TASK=align_blocks
"$EH_PY" - <<'PY'
import os
from embodied_harness.adapters.robodojo_native import layout_catalog
catalog = layout_catalog('/home/xuanzhi/RoboDojo', os.environ['EH_TASK'])
for key, value in list(catalog.items())[:12]:
    print(key, value)
print('total layouts:', len(catalog))
print('banks:', sorted({value['bank'] for value in catalog.values()}))
PY
```

scene_ref=`bank*1000000+ordinal`；ordinal是数字文件名排序的下标，不是文件名后缀。
不要直接假定bank1/2存在。下面`0/1/2`只是同一bank内三个布局的**工程smoke示例**，必须先确认存在，
且0作为dev包含前面启动检查已看过的场景。若这些布局曾作为其他实验的held-out，重新明确本条件划分。

下面refs文件只在首次创建时写入；若已经存在，先检查已有内容，不重复覆盖。

```bash
cat > "$EH_ROOT/manual/config/smoke.refs.json" <<'JSON'
{"development": [0], "unseen": [1], "heldout": [2]}
JSON

embodied-harness partition \
  --runtime-root /home/xuanzhi/RoboDojo --task "$EH_TASK" \
  --refs "$EH_ROOT/manual/config/smoke.refs.json" \
  --output "$EH_ROOT/manual/config/smoke.partition.json"
```

partition会校验存在性、互斥性和布局hash。失败时选真实不同布局，不跳过检查。
这三个场景只用于工程smoke，不作为正式泛化性能结论。

冻结初始知识库一次：

```bash
mkdir -p "$EH_ROOT/manual/knowledge-empty/skills" "$EH_ROOT/manual/knowledge-empty/lessons"
embodied-harness freeze "$EH_ROOT/manual/knowledge-empty" "$EH_ROOT/manual/knowledge-initial"
embodied-harness verify "$EH_ROOT/manual/knowledge-initial"
```

再次freeze同一目标报已存在时，用verify检查已有库，不删证据后重跑。

## 11. 生成campaign配置，先跑doctor

**终端C：**执行第1步、source tokens、设置System 1 key和实际model ID。
任务goal应来自公开任务指令；第6步保存的`manual/last-snapshot.json`中有同一align_blocks的instruction。
若你换任务，先读取新任务公开观测，不把旧指令套过去。

```bash
source "$EH_ROOT/manual/tokens.env"
export EH_GOAL='替换为本任务的实际公开指令'
"$EH_PY" - <<'PY'
import json, os, time, urllib.request, uuid
from pathlib import Path
root = Path(os.environ['EH_ROOT'])
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open('http://127.0.0.1:8114/ready', timeout=10) as response:
    ready = json.loads(response.read())
expected = ('sam3:code:267ca11c97ecba8455e519d720b74c96cfadb02f87429629d53c6b7c9d459757:'
            'checkpoint:9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e')
assert ready.get('ready') is True and ready.get('model_version') == expected
assert os.environ.get('TYPESAFE_API_KEY'), '缺System 1 key'
model = os.environ['EH_JEV_MODEL']
goal = os.environ['EH_GOAL']
assert model and '替换' not in model and goal and '替换' not in goal
run = root / 'manual' / ('smoke-' + time.strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6])
config = {'goal': goal, 'partition_manifest': str(root / 'manual/config/smoke.partition.json'),
          'frozen_knowledge': str(root / 'manual/knowledge-initial'), 'run_dir': str(run),
          'max_versions': 1, 'agent_url': 'http://127.0.0.1:8123', 'agent_token_env': 'EMBODIED_AGENT_TOKEN',
          'provisioner_url': 'http://127.0.0.1:8122', 'provisioner_token_env': 'EMBODIED_PROVISIONER_TOKEN',
          'harness': {'worker_token_env': 'EMBODIED_WORKER_TOKEN',
                      'sam3_url': 'http://127.0.0.1:8114', 'sam3_model_version': expected,
                      'selector': 'jev', 'jev_model': model, 'timeout_s': 240,
                      'max_submissions': 5, 'skill_python': str(root / 'venv/bin/python')}}
path = root / 'manual/config' / (run.name + '.json')
with path.open('x') as stream:
    json.dump(config, stream, indent=2)
(root / 'manual/last-campaign-config.txt').write_text(str(path))
print('config:', path)
print('run:', run)
PY

EH_CAMPAIGN_CONFIG=$(cat "$EH_ROOT/manual/last-campaign-config.txt")
embodied-harness doctor --config "$EH_CAMPAIGN_CONFIG"
```

预期`passed:true`，provisioner为commissioned/idle，SAM3 pin匹配，角色服务就绪。
doctor不调用模型生成、不启动物理、不证明System 1 key有效，因此第8步不能省略。
它同样不能证明profile测量是真的，也不能证明System 2模型驱动会正确看图。

这份smoke配置限制每个episode的宿主提交数为5，原生预算仍独立生效；**不是5个仿真帧**。
max_versions=1用于先排通链路；不是完整的多版本修复实验。极低预算未完成任务属于预期可能结果。

## 12. 开始第一次小预算运行

确认以下进程分工，每个服务终端保持运行：

| 终端 | 工作 |
| --- | --- |
| S | SAM3（若已有健康服务则复用，不重复启动） |
| A | 真实System 2 agent-service，或连接该设备的SSH隧道 |
| P | RoboDojo provisioner |
| C | campaign、System 1 key、查看报告 |

**从本步开始会提交仿真机器人动作，并调用实际模型。**前面所有门槛完成后再执行：

```bash
set -o pipefail
EH_CAMPAIGN_CONFIG=$(cat "$EH_ROOT/manual/last-campaign-config.txt")
embodied-harness campaign --config "$EH_CAMPAIGN_CONFIG" 2>&1 | tee "${EH_CAMPAIGN_CONFIG%.json}.log"
```

campaign会依次做preview、生成程序、dev、unseen、heldout、技能提议/验证。
它按需要多次重新启动Isaac，不是只有3次episode；技能验证也有独立episode和模型调用。
当前max_versions=1不会修复dev程序。frozen评估禁止episode内System 2修复；若把所有阶段放进一个完整技能，
可持续到原生成功；多个subtask需可信宿主relation evaluator，不能靠code done冒充物理完成。

结果保存在配置的run_dir：

```bash
"$EH_PY" - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ['EH_ROOT'])
config = json.loads(Path((root / 'manual/last-campaign-config.txt').read_text()).read_text())
run = Path(config['run_dir'])
print('run_dir:', run)
ledger = json.loads((run / 'campaign.json').read_text())
print('complete:', ledger.get('complete', False))
for key, record in ledger['trials'].items():
    result = record.get('result', {})
    print(key, record['status'], result.get('status'), record.get('reason'))
PY
```

检查三件事：

1. 工程链路是否完整结束：`complete:true`、native进程关闭、日志可追溯。
2. 任务是否实际成功：看各trial的`primary_native_success`和原生终止，不把complete当成功率。
3. 程序是否确实使用SAM3、System 1和公开API：检查各trial的`harness/events.jsonl`以及服务audit。

工程smoke成功后，再为正式实验单独准备分区和run_dir，将max_versions改为3、
设置经过验证的动作额度并保留原生预算。每版本全dev复测，按dev选定；unseen/heldout结果不用于调参。
不要把调试中反复查看的scene继续称为新的unseen/heldout。

## 13. 失败、停止和恢复

- 原生启动失败：读`manual/native-allocations/<allocation_id>/process.log`。
- SAM3失败：读`logs/sam3.log`和audit；确认真实mask与当前快照。
- 模型401/超时/JSON不符合Plan：检查System 1/System 2接入与各自日志，不修改原生成功判定。
- 标为failed/interrupted的trial不会自动重跑；保留目录。开始新实验使用新run_dir，不删除ledger换成功分母。
- campaign终端Ctrl+C只能中断调用，不保证远端episode已关闭。首先联系仍运行的provisioner检查分配：

```bash
"$EH_PY" - <<'PY'
import os
from embodied_harness.rpc import RpcClient
rpc = RpcClient('http://127.0.0.1:8122', os.environ['EMBODIED_PROVISIONER_TOKEN'], timeout_s=60)
print(rpc.call('describe'))
PY
```

停止整个调试会话时，在终端P按Ctrl+C，让provisioner清理自己持有的worker；观察关闭结果。
再按需要停止agent-service、隧道和你启动的SAM3。旧部署中的SAM3可能是后台进程，
不要依据历史PID直接kill；先核对当前进程命令、用户和端口再处理。
若出现强制清理/进程退出异常，保留对应日志，当前trial不计作正常完成。

## 14. 如果卡住，反馈这些即可

不用贴API key、tokens.env、整个历史实验目录或私有评分代码。

- 正在执行的步骤编号、命令及完整错误末尾。
- 第2步探针报告中的passed/closed/error/traceback。
- `/ready`输出和`nvidia-smi`占用。
- native日志末尾约80行，及是否还存在该进程。
- doctor输出或campaign中失败trial的status/reason。
- 若卡在第7步，说明你选择的System 2模型、API/CLI形式和部署位置；该信息决定实际driver如何接入。

当前最有价值的下一步是**第2步：取回native-probe-report-02.json**。先确认真实模拟器reset/capture/close，
再逐项往后走；不需要先重新部署全部依赖。
