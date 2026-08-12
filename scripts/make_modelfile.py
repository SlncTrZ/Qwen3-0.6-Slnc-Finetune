import sys, os, struct, argparse
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = r"D:\ScriptPython\Training"

TEMPLATE_QWEN3_CHAT = """{{- if .Messages }}
{{- if or .System .Tools }}<|im_start|>system
{{ .System }}
{{- if .Tools }}

# Tools

You are provided with function signatures within <tools></tools> XML tags:
<tools>{{- range .Tools }}
{"type": "function", "function": {{ .Function }}}{{- end }}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
{{- end }}<|im_end|>
{{ end }}
{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 -}}
{{- if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
{{ else if eq .Role "assistant" }}<|im_start|>assistant
{{ if .Content }}{{ .Content }}
{{- else if .ToolCalls }}<tool_call>
{{ range .ToolCalls }}{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}
{{ end }}</tool_call>
{{- end }}{{ if not $last }}<|im_end|>
{{ end }}
{{- else if eq .Role "tool" }}<|im_start|>user
<tool_response>
{{ .Content }}
</tool_response><|im_end|>
{{ end }}
{{- if and (ne .Role "assistant") $last }}<|im_start|>assistant
{{ end }}
{{- end }}
{{- else }}
{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ end }}{{ .Response }}{{ if .Response }}<|im_end|>{{ end }}"""

TEMPLATE_QWEN3_THINK = """{{- if .Messages }}
{{- if or .System .Tools }}<|im_start|>system
{{ .System }}
{{- if .Tools }}

# Tools

You are provided with function signatures within <tools></tools> XML tags:
<tools>{{- range .Tools }}
{"type": "function", "function": {{ .Function }}}{{- end }}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
{{- end }}<|im_end|>
{{ end }}
{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 -}}
{{- if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
{{ else if eq .Role "assistant" }}<|im_start|>assistant
{{ if .Content }}{{ .Content }}
{{- else if .ToolCalls }}<tool_call>
{{ range .ToolCalls }}{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}
{{ end }}</tool_call>
{{- end }}{{ if not $last }}<|im_end|>
{{ end }}
{{- else if eq .Role "tool" }}<|im_start|>user
<tool_response>
{{ .Content }}
</tool_response><|im_end|>
{{ end }}
{{- if and (ne .Role "assistant") $last }}<|im_start|>assistant
<think>
{{ end }}
{{- end }}
{{- else }}
{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
<think>
{{ end }}{{ .Response }}{{ if .Response }}<|im_end|>{{ end }}"""

DEFAULT_SYSTEM = "You are a helpful assistant."


def detect_think_with_tokenizer(model_dir):
    """Dung tokenizer that de kiem tra generation prompt co <think> khong."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_dir)
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": "x"}],
            tokenize=False, add_generation_prompt=True,
        )
        tail = prompt.rstrip()
        return tail.endswith("<think>") or tail.endswith("<think>\n")
    except Exception as e:
        print("[warn] khong load duoc tokenizer tu", model_dir, "->", e)
        return None


def read_gguf_template(path):
    try:
        f = open(path, "rb")
        f.read(4)
        f.read(4)
        f.read(8)
        n_kv = struct.unpack("<Q", f.read(8))[0]

        def read_string():
            ln = struct.unpack("<Q", f.read(8))[0]
            return f.read(ln).decode("utf-8", errors="replace")

        def read_value(t):
            if t == 0: return struct.unpack("<B", f.read(1))[0]
            if t == 1: return struct.unpack("<b", f.read(1))[0]
            if t == 2: return struct.unpack("<H", f.read(2))[0]
            if t == 3: return struct.unpack("<h", f.read(2))[0]
            if t == 4: return struct.unpack("<I", f.read(4))[0]
            if t == 5: return struct.unpack("<i", f.read(4))[0]
            if t == 6: return struct.unpack("<f", f.read(4))[0]
            if t == 7: return struct.unpack("<B", f.read(1))[0]
            if t == 8: return read_string()
            if t == 10: return struct.unpack("<Q", f.read(8))[0]
            if t == 11: return struct.unpack("<q", f.read(8))[0]
            if t == 12: return struct.unpack("<d", f.read(8))[0]
            if t == 9:
                elem_t = struct.unpack("<I", f.read(4))[0]
                n = struct.unpack("<Q", f.read(8))[0]
                return [read_value(elem_t) for _ in range(n)]
            return None

        for _ in range(n_kv):
            key = read_string()
            t = struct.unpack("<I", f.read(4))[0]
            val = read_value(t)
            if key == "tokenizer.chat_template":
                f.close()
                return val
        f.close()
    except Exception as e:
        print("[warn] khong doc duoc chat_template tu GGUF:", e)
    return None


def main():
    parser = argparse.ArgumentParser(description="Sinh Modelfile chuan Qwen3 cho Ollama")
    parser.add_argument("--gguf", required=True, help="Duong dan file GGUF")
    parser.add_argument("--name", default="qwen3-finetuned", help="Ten model ollama")
    parser.add_argument("--out", default=os.path.join(ROOT, "Modelfile"), help="Duong dan Modelfile")
    parser.add_argument("--system", default=DEFAULT_SYSTEM, help="System prompt mac dinh")
    parser.add_argument("--model", default=None,
                        help="Thu muc model base/merged (co tokenizer) de detect thinking chinh xac")
    parser.add_argument("--template", choices=["auto", "chat", "think"], default="auto",
                        help="Loai template: auto (tu detect), chat (khong think), think (co think)")
    args = parser.parse_args()

    if args.template == "auto":
        is_think = None
        if args.model:
            is_think = detect_think_with_tokenizer(args.model)
        if is_think is None:
            embedded = read_gguf_template(args.gguf)
            name_hint = (args.name or "") + "/" + (args.model or "")
            is_think = bool(embedded) and " thinking" in embedded
            if "thinking" in name_hint.lower() or "think" in name_hint.lower():
                is_think = True
            print("Embedded GGUF chat_template:", "co" if embedded else "khong")
        if is_think:
            template = TEMPLATE_QWEN3_THINK
            kind = "think"
        else:
            template = TEMPLATE_QWEN3_CHAT
            kind = "chat"
    elif args.template == "think":
        template = TEMPLATE_QWEN3_THINK
        kind = "think"
    else:
        template = TEMPLATE_QWEN3_CHAT
        kind = "chat"

    gguf = args.gguf.replace("\\", "/")
    lines = []
    lines.append("FROM " + gguf)
    lines.append("")
    lines.append("TEMPLATE \"\"\"" + template + "\"\"\"")
    lines.append("")
    lines.append("SYSTEM \"\"\"" + args.system.replace("\\", "\\\\").replace('"', '\\"') + "\"\"\"")
    lines.append("")
    lines.append("PARAMETER temperature 0.6")
    lines.append("PARAMETER top_p 0.95")
    lines.append("PARAMETER top_k 20")
    lines.append("PARAMETER repeat_penalty 1.1")
    lines.append("")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("Da ghi Modelfile:", args.out)
    print("Loai template   :", kind)
    print("Model ollama    :", args.name)


if __name__ == "__main__":
    main()
