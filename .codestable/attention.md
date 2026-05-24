# Attention

本文件是 CodeStable 技能启动必读的项目注意事项入口。所有 CodeStable 子技能开始工作前必须读取它。

## 项目碎片知识

<!-- cs-note managed: 用 cs-note 维护，新条目按下面分节追加 -->

### 编译与构建

### 运行与本地起服务

- 任何 `samd` import / 测试都依赖 `torch` / `transformers`（~4.46.x）/ `safetensors`，缺一个会在 `samd/__init__.py:1` 立刻 `ImportError`；按 README 的版本要求装齐再跑

### 测试

### 命令与脚本陷阱

- Llama-3 系列 tokenizer 默认 `pad_token=None`，`tests/test_samd.py:113` 用 `padding=True` 会 raise；脚本里已加 `if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token` 兜底，未来扩展 tokenizer 路径时记得保留这条

### 路径与目录约定

### 环境变量与凭证

### 其他

- EAGLE3 官方权重发布**不含** `embed_tokens.weight`（draft 共享 base 的 embedding，训练时 `load_emb=True` 从 base 路径加载）；samd 在 `Eagle3.__init__` 末尾从 `lm.model.embed_tokens.weight` 复制兜底，未来其他 EAGLE 系列权重接入要确认该字段在不在权重里
