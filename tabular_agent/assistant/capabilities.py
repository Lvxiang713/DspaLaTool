"""Conservative model profiles. Unknown models receive no reasoning parameters."""
import re


def reasoning_capability(provider, model):
    if provider=="others":
        provider="deepseek" if "deepseek" in str(model).casefold() else "qwen" if "qwen" in str(model).casefold() else "others"
    name=str(model).casefold().split("/")[-1]
    kind, levels, note = "unknown", ["off"], "Capability not confirmed. Uses model defaults; no thinking parameters are sent."
    if provider=="qwen":
        if re.match(r"qwen3\.8[-_:]",name):
            kind,levels,note="effort",["off","low","medium","high"],"Qwen3.8 profile: High maps to xhigh. Verified on the configured local deployment."
        elif "instruct-2507" in name or name.startswith("qwen2") or "non-thinking" in name:
            kind,levels,note="none",["off"],"This model profile does not support configurable thinking."
        elif "thinking" in name and name.startswith("qwen3"):
            kind,levels,note="fixed",["on"],"Thinking-only model. Thinking is always enabled; no strength control is sent."
        elif re.fullmatch(r"qwen3-\d+(?:\.\d+)?b(?:-a\d+b)?(?:-2504)?",name):
            kind,levels,note="toggle",["off","on"],"Qwen3 hybrid model: thinking switch only. No intensity parameter is sent."
    elif provider=="deepseek":
        if name in ("deepseek-v4-pro","deepseek-v4-flash","deepseek-flash"):
            kind,levels,note="effort",["off","low","high","max"],"DeepSeek V4 API profile. Endpoint compatibility depends on the deployment."
        elif name=="deepseek-reasoner" or name.startswith("deepseek-r1"):
            kind,levels,note="fixed",["on"],"Reasoning model: no configurable intensity is assumed."
        elif name=="deepseek-chat":
            kind,levels,note="none",["off"],"Chat model profile: no thinking controls are sent."
    return {"kind":kind,"levels":levels,"note":note,
            "label":"Thinking intensity" if kind=="effort" else "Thinking mode"}
