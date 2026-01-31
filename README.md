# llama2.q + llama2.k
Inference for [Llama2]-like Transformer models in one [Q] file, or one [K] file

Inspired by the Andrej Karpathy's [llama2.c]. 

No libraries are used.

### Features
- Mostly binary compatible (i.e. should produce exactly the same outputs as the C version given the parameters and random seed)
- Achieves around 75/27/13 tokens per second on my laptop for the 15/45/110M models, respectively.
- Full 7B model is not loadable by this version bc the trick where we abuse the IPC serializer as the binary parser doesn't work with 26GB file. You're free to add chunked loading if you want ;)
- Still likely the **shortest self-contained transformer inference** in the world.
- ~5500 bytes, 76 lines of Q or 78 lines of K + <500 KB kdb
- <70 lines and under 5KB if you are fine with deterministic sampling

Includes the [TinyStories] 15M model. Tested with Kdb+ 4.0

### Usage

## K (K4) version

```sh
q llama2.k -checkpoint stories15M.bin -tokenizer tokenizer.bin -prompt "Once upon a time"
```

## Q version

```sh
q llama2.q -checkpoint stories15M.bin -tokenizer tokenizer.bin -prompt "Once upon a time"
```

Larger TinyStories models:
```sh
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.bin
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.bin
```

Arguments:
- `-checkpoint <string>` - model weights
- `-tokenizer <string>` - tokenizer config
- `-prompt <string>` - initial prompt
- `-temp <float>` - temperature (0..1, 0 = deterministic argmax)
- `-seed <int>` - random seed
- `-steps <int>` - number of tokens to generate (0..256, default 256)
- `-topp <float>` - p value for nucleus sampling, default 0.9


[TinyStories]: https://arxiv.org/abs/2305.07759
[llama2.c]: https://github.com/karpathy/llama2.c
[Llama2]: https://ai.meta.com/llama/
[Q]: https://code.kx.com/q/learn/startingkdb/language/
[K]: https://en.wikipedia.org/wiki/K_%28programming_language%29