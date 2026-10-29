# 持久 Python 技能执行边界

本地代码提供 PersistentProgram、PythonSkill 和 Session.delegate_python。
技能代码直接在独立的持久 Python 子进程中运行，不需要 Docker。

默认使用 Harness 当前 Python 解释器。若技能需要独立虚拟环境，可在配置中设置：

```json
{"skill_python": "/absolute/path/to/skill-venv/bin/python"}
```

不指定 skill_python 即使用 sys.executable。该解释器需安装 NumPy；worker 使用本次 Harness
的源码路径，确保通信 contract 与宿主一致。每次委托使用临时工作目录，委托内保留 globals，
关闭后清理临时目录。停止/超时会终止子进程组，读写管道均有截止时间。

子进程只继承 PATH/LANG/LC_ALL/TZ 及指定的 SDK 路径，不主动传入宿主 API token。
这是进程生命周期和动作接口的分离，不是文件系统、网络或账号权限沙箱；技能代码仍有
运行账号的系统访问权限。远端调试使用你控制的源码与专用运行环境。

## hooks 与远端工具

System 2 上传源码和 JSON 参数，需定义五个 hook：

- skill_init(parameters) → 状态，无动作。
- skill_observe(state) → feedback，无动作。
- skill_context(state) → 包含 status 的字典，无动作。
- skill_options(state) → 选项描述字典，无动作；完成时可为空。
- skill_tick(state, selected_option) → feedback，最多一次 step。

命名空间在当前委托中持续存在；修复会启用新子进程，不能假设继承上一版本的 Python globals。
技能通过 get_observation、localize_object 和 step 访问宿主。参数/导出值使用明确白名单 wire
类型，不使用 pickle 或从 wire 动态导入模块。step 接收新 RobotAction，返回 StepReceipt，
不是旧四原语的 native dict 返回格式；旧技能需要改接口，不能原样复用。

宿主逐条仲裁实际 primitive 请求。保存 step 别名、捕获额度异常或直接伪造协议，都不能
增加物理动作额度。子进程的 done/ok 仅表示代码状态，不证明物理关系。
单次 stdout/stderr 各截为8192字符，错误 traceback 最多8192字符进入交接诊断。
读、写管道均有截止时间，超时关闭程序；已提交原生动作不能因此假定未执行。

远端调用示例：

```python
from pathlib import Path

report = client.call(
    "delegate_python", task_id="current-task", goal="current operator goal",
    source=Path("examples/ee_skill.py").read_text(),
    parameters={"arm": "left", "target": measured_target_pose},
    revision=1, max_actions=100, max_decisions=150, max_seconds=180,
)
```

measured_target_pose 为当前场景推导的 `{position, quaternion_wxyz, frame}`；例子不提供
任何可直接用于真实场景的坐标。FixedSelector 要求 active 菜单含 normal。
Jev 支持合法 choice 菜单，宿主验证选项/概率/置信度，但无法证明源码对选项的文字描述
与真实效果一致，仍须检查动作日志。此条件标记 python-hooks-v1，不混为声明式 servo 条件。

源码和参数保存到事件记录。最多3次修订，revision递增，目标相同，不得扩大首次委托预算；
相同 AST+参数或只改注释不能重启。Python 任务不能切换到声明式同 task_id 来绕过限制。
与声明式计划不同，任意 Python 修复的“已完成任务前缀”不能静态证明，需 System 2
根据物体状态保留；当前没有完整 fixed-program/Test 局部修复协议。

代码 status=done 默认返回 awaiting_verification。只有可信宿主 relation evaluator 返回
同一当前观测的 true，才返回 verified_subtask；原生终止成功才返回 native_task_succeeded。
未知/低置信度/预算到限/观察或动作失败返回 needs_system2。

## 测试测了什么

7项测试直接使用生产 PersistentProgram 构造函数启动真实 Python 子进程，不再替换为
测试专用进程创建器；不运行真实模型代码：

1. globals跨提交保留，Observation可经管道往返；非法导出返回诊断错误。
2. zero-action hook保存别名并捕获异常后，仍不能发动作且报告失败。
3. tick第二次调用step被宿主拒绝，即使子进程吞掉异常仍报告额度违规。
4. 输出有界；无限循环按时终止。
5. 伪造done后继续占用子进程，下一次大管道写仍会超时退出。
6. 子进程与宿主PID不同，使用当前解释器，宿主TYPESAFE_API_KEY不进入子进程环境。
7. Python hooks通过真实进程管道接到模拟worker，代码done只能得到awaiting_verification。

完整93项本地测试通过。尚未测试远程System 2模型生成的技能、GPU或接触物理。
