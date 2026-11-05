# 100.105.7.84 部署检查（2026-10-30）

结论：已部署代码及真实 SAM3；完整实验就绪性尚未通过。

## 已完成且有远端输出证明

- SSH：xuanzhi@100.105.7.84，hostname=peilab-System-Product-Name。
- 新部署目录：/home/xuanzhi/embodied-harness-deploy-20261030；没有修改旧 ASPIRE/RoboDojo 安装。
- 源码包 SHA256：8d8cd59e235747df5c4f76817795d288184bd39930a1bbd1a7ecbafa9ef50164。
- 独立 venv，复用已有依赖；editable 安装成功。
- 远端完整测试：128项通过，9.773秒。CPU、真实子进程/HTTP；物理和模型 fixture。
- GPU：24455MiB，compute capability 12.0，driver580.65.06。
- 环境：Python3.11.17，Torch2.7.0+cu128；SAM3/IsaacLab/CuRobo 均已存在。
- 真实SAM3严格权重加载成功，监听127.0.0.1:8114，/ready返回true。
- model_version：sam3:code:267ca11c97ecba8455e519d720b74c96cfadb02f87429629d53c6b7c9d459757:checkpoint:9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e。
- 当时SAM3 GPU占用约4150MiB；服务启动PID1487561，PID只是检查时记录，需重新验证存活。
- 日志：部署目录/logs/sam3.log、logs/sam3-audit.jsonl、cpu-tests.log。

## 尚未通过

1. 第一次真实native零动作探针停在Omniverse EULA提示，未启动App、未reset、未发动作。
   证据：部署目录/native-probe-report.json、logs/native-probe.log。
   用户随后明确回复“我已接受，继续启动”，授权设置OMNI_KIT_ACCEPT_EULA=YES。
   两次后续启动请求均被本地工具自动权限审批超时拒绝，命令未执行；这不表示模拟器故障或授权缺失。
2. 真实SAM3文本/点提示、当前RGB-D反投影尚未执行；/ready仅证明严格加载和服务启动。
3. 未确认可用的commissioned profile。旧设备测量给出了实际关节映射/限位；其范围仅为局部小角度探针，
   不足以宣称相机光学系、TCP、workspace及抓取已验证。
4. 登录环境TYPESAFE_API_KEY/OPENAI_API_KEY/ANTHROPIC_API_KEY/HF_TOKEN均未设置；没有读取任何密钥值。
   System2模型驱动与远端服务地址也未核实。不排除凭据保存在其他明确配置位置，当前未对其做全盘搜索。
5. 未启动provisioner/agent-service，未创建正式任务分区/配置，也未执行完整doctor或实验。

## 下一次继续顺序

1. 复用已获授权的EULA设置，以新的探针目录执行真实native reset/capture/close并保留旧失败记录。
2. 保存公开RGB-D图像；真实SAM3文本/点分割和反投影验收。
3. 完成profile真实标定；核实System1凭据和System2多模态驱动/服务。
4. 启动provisioner/agent-service，配置实际任务与互斥分区，运行doctor。
5. 先小预算dev smoke，再启动正式campaign；不把fixture结果计入实验。

## 本轮重试更新

- 再次确认SAM3 /ready=true，模型指纹不变，GPU占用4150MiB。
- 已获批准并实际执行设置OMNI_KIT_ACCEPT_EULA=YES的远端启动命令。
- 创建native-probe-02.py，新的run_dir=native-probe-02、公开输出目录native-public-02。
- 远端返回native_probe_pid=1507143；记录在native-process-02.json。
- 预期结果：native-probe-report-02.json；日志：logs/native-probe-02.log。
- 后续两次只读获取结果均遭工具自动权限审批超时，未执行；普通沙箱SSH也返回Operation not permitted。
- 因此不能确认App启动/reset/capture/close是否成功，也没有提交任何动作或正式campaign。
- 需要从服务器读取上述结果文件后继续判断，不能把PID提交成功计为验收通过。
