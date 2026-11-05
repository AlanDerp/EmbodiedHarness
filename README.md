# EmbodiedHarness

参照 ASPIRE 的工程框架与 tabletop fix-loop，结合 EmbodiedDualLoop 在 RoboDojo 上的适配，
实现 System 2 规划/修复、System 1 有界数值执行及 SAM3 视觉感知。全部是普通 Python 服务/子进程，无 Docker。

已实现：公开观测与同帧 RGB-D 几何、SAM3 文本/点分割服务、原生 RoboDojo 启动与生命周期、
工具 registry/源码契约、Python hooks、数值参数仲裁、S2 接力、ASPIRE campaign、多任务 verified 库传递与独立 rerun。
本机验证包括真实 HTTP 和 Python 子进程；模型与物理使用 fixture。真实 GPU、Isaac、Jev 和模型驱动需在服务器验收。

```sh
python -m pip install -e '.[sam3-client]'
python -m unittest discover -s tests -q
embodied-harness --help
```

- [Codex 按类别启动测试（AGENTS.md 入口）](docs/CODEX_CATEGORY_TEST.zh-CN.md)
- [缺口清单、实施顺序与验收记录](docs/REBUILD_PLAN.zh-CN.md)
- [服务器部署、分区和实验命令](docs/SERVER_DEBUG.zh-CN.md)
- [这台服务器的逐步手动部署教程](docs/MANUAL_DEPLOYMENT_100.105.7.84.zh-CN.md)
- [本地测试覆盖与远端验收边界](docs/LOCAL_BUILD.zh-CN.md)
- [SAM3 安装、权重指纹与真实快照验收](docs/SAM3.zh-CN.md)
- [Python 技能与 System 1 参数契约](docs/PYTHON_SKILLS.zh-CN.md)
- [项目 fix-loop skill](.agents/skills/fix-loop/SKILL.md)
- [来源、许可与改编说明](THIRD_PARTY_NOTICES.md)

核心入口：`campaign` 单任务 dev→选定→unseen→heldout→独立技能验证；`curriculum` 按任务顺序传递库，
最后对低分且库已变化的任务进行单独 `rerun_02`。`doctor` 只读检查服务，`check-sam3` 验收真实快照。
交互式开发保留 `serve/run_task` 的 S2 接力；严格 campaign 中 S2 只在 dev 版本之间修复。

使用新的 `aspire-fixloop-dualloop-robodojo-v1` 协议，不冒用 ASPIRE 原始 benchmark 或旧四原语实验标签。
源码白名单和无凭据子进程是公开 API 执行契约，不是 OS 权限沙箱；部署使用专用账号与已标定 profile。
