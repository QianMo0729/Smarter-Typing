# 第三方组件与许可

智能拼音的源码按 GPL-3.0 发布，全文见 [LICENSE](LICENSE)。安装包里还包含下面这些第三方内容：

| 组件 | 在安装包里的形式 | 许可 |
|---|---|---|
| [rime-ice（雾凇拼音）](https://github.com/iDvel/rime-ice) 词库 | 编译成二进制词库 `lex.bin` | GPL-3.0，见 [LICENSE](LICENSE) |
| [Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base) | `model.gguf`：原始权重转换为 GGUF 并量化为 Q8，没有其他改动 | Apache-2.0，见 [licenses/Apache-2.0.txt](licenses/Apache-2.0.txt) |
| [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) | 静态链接进主程序 | MIT，见 [licenses/llama.cpp-MIT.txt](licenses/llama.cpp-MIT.txt) |

`lm.bin` 是从公开语料统计出来的词频和二元概率，语料来源见 README 的“数据来源与许可”。语料本身不随程序分发。
