# 智能拼音（SmartPinyin）

一个在 Mac 本机运行的中文拼音输入法。它会结合前文判断你真正想打的词，能容忍常见误触，并从你正常的打字、删改中自动学习你的习惯。打完最后一个字母可以直接按空格，此时智能首选已经准备好。

- 智能判断：Qwen3-0.6B-Base（0.6B，Q8 量化，640 MB），在本机 GPU 上用 llama.cpp 运行。它只给候选输出概率，不生成文字。
- 全部本地运行，个人数据不出本机。

设计细节见 [docs/DESIGN.md](docs/DESIGN.md)，评测结果见 [docs/RESULTS.md](docs/RESULTS.md)。

## 安装 / 更新

```sh
./scripts/build_app.sh   # 生成 build/SmartPinyin.app（含词库、语言模型、小模型）
./scripts/install.sh     # 装到 ~/Library/Input Methods 并启用
```

然后在菜单栏输入法列表里选“智能拼音”。如果没出现：系统设置 → 键盘 → 输入法 → “+” → 简体中文 → 智能拼音。

安装后第一次启动要编译 GPU 内核，大约 13 秒；这段时间用 n-gram 排序，照常可以打字。日志在 `~/Library/Logs/SmartPinyin.log`，出现 `neural model ready` 就表示智能判断已就绪。

卸载：删除 `~/Library/Input Methods/SmartPinyin.app`，再在输入法设置里移除。个人数据在 `~/Library/Application Support/SmartPinyin/`。

## 使用

| 按键 | 作用 |
|---|---|
| 拼音字母 | 输入。正文里的拼音按音节分开显示（zhi neng），候选第一个是整句的智能首选 |
| 空格 | 上屏首选（打完最后一个字母立刻按即可） |
| 1–9 / 鼠标点击 | 选择候选；选中的词只覆盖前半段时，剩下的拼音继续转换 |
| ← → / Tab | 移动高亮 |
| ↓ / 点右侧箭头 | 展开成多行候选，可以一直往下翻；↑ 或 Esc 收起 |
| - = 或 [ ] | 翻页 |
| 回车 | 上屏原字母 |
| Esc | 取消 |
| 中/英键（Caps Lock） | 在 ABC 和智能拼音之间切换（需在“系统设置 › 键盘 › 输入法”中打开“使用中/英键切换”） |
| `'` | 手动分隔音节（xi'an） |

智能排序只决定前几个候选。后面仍按常用程度列出所有能匹配的字和词（包括 4 万多个生僻字），和普通输入法一样可以一直翻页找到。

**设置**：点输入法菜单 →“智能拼音设置…”，可以调：
- 候选窗主题色（默认南科大橙 #ED6C00，也可以跟随系统强调色或自选颜色）
- 横排/竖排、浅色/深色/跟随系统、字号、每页候选数
- 拼音分隔显示、全角标点
- 智能判断、自动学习的开关，以及清除个人数据

菜单里还能看到智能判断是否就绪、空格等待统计。

## 它怎么学习（不需要你做任何标注）

正常打字就行：
- 上屏后留下来的文本，算一个弱的正向信号。
- 删掉刚上屏的词、用同样拼音换了别的字：说明选词错了，会记住“在这种语境下要这个词”。
- 删掉后重新输入的按键只差一两处：说明是误触，会记住你常犯的误触类型。
- 删掉后写了完全不同的内容：视为改写或改主意，不认为原来的字错了。
- 在候选里手动选了别的词：说明更喜欢这个，同样结合语境记住。
- 一个字一个字拼出来的人名、新词（比如输入 linjiwen 后依次选“林”“霁”“雯”）：词库里没有，输入法自己也组不出来，所以直接记成一个“用户词”。下次再打同样的拼音就是首选，放在句子中间、只打了一半、有一处误触时也能认出来。刚拼完就删掉的不算；后来被你删掉改成别的，它会降下去直到被忘掉。

保存的只有词计数、哈希后的上下文特征、统计量，以及你教给它的用户词（人名、新词）和选词记忆里的那几个字，不保存成段的原文。密码管理器等应用不参与学习（密码框本来也不会调用输入法）。

## 开发

```sh
swift build -c release                 # 引擎 + 输入法 + 评测工具
swift test --scratch-path /tmp/sp-test # 单元测试（Desktop 在 iCloud 里，签名需要换目录）

.build/release/stbench decode zhineng "时间不够，我" --nn        # 看候选、概率与各项特征
.build/release/stbench eval data/eval/test.jsonl --nn            # 留出语料上的首选准确率 / 误触恢复 / 误纠
.build/release/stbench latency --key-ms 70 --space-ms 40         # 真实节奏回放：空格时是否已有智能首选
.build/release/stbench personalize murmur --habit uo             # 模拟真实用户长期使用，看学习效果
```

数据与模型构建（Python 环境在 `tools/.venv`）：

```sh
tools/build_lexicon.py && tools/build_lexbin.py    # rime-ice 词库 → lex.bin
tools/prepare_corpus.py                            # 语料清洗（维基、知乎、V2EX、FineWeb-Edu 中文）
tools/build_ngram.py                               # 分词 + 改进 Kneser-Ney 二元模型 → lm.bin
tools/make_evalset.py data/eval/test.jsonl 3000 --split test
.build/release/stbench eval data/eval/train.jsonl --nn --dump feats.jsonl
tools/fit_weights.py feats.jsonl data/build/weights.json
```

大体积数据（语料、模型、构建产物，约 10 GB）放在 `~/Developer/smarter-typing-data`，项目里只留符号链接。原因是 Desktop 开启了 iCloud 同步。

## 数据来源与许可

- 词库：[rime-ice（雾凇拼音）](https://github.com/iDvel/rime-ice)，GPL-3.0
- 语料：中文维基（CC BY-SA）、知乎 KOL、V2EX、Chinese-FineWeb-Edu-v2（Apache-2.0）。仅用于本地统计，不随程序分发
- 模型：[Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base)，Apache-2.0
- 推理：[llama.cpp](https://github.com/ggml-org/llama.cpp)，MIT
