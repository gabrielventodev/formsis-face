"""Exports the MiniFASNet anti-spoofing models of Silent-Face-Anti-Spoofing (Apache 2.0,
https://github.com/minivision-ai/Silent-Face-Anti-Spoofing) to ONNX with a softmax head, so the
service can run them with OpenCV and without PyTorch.

Usage (needs torch, which the service itself does not):
    git clone https://github.com/minivision-ai/Silent-Face-Anti-Spoofing /tmp/sfas
    python tools/export_minifasnet.py /tmp/sfas models
    (cd models && sha256sum *.onnx > SHA256SUMS)
"""
import sys, os, collections, torch
S = sys.argv[1]; out = sys.argv[2]
sys.path.insert(0, S)
from src.model_lib.MiniFASNet import MiniFASNetV2, MiniFASNetV1SE
from src.utility import get_kernel
for name, cls in [("2.7_80x80_MiniFASNetV2", MiniFASNetV2), ("4_0_0_80x80_MiniFASNetV1SE", MiniFASNetV1SE)]:
    m = cls(conv6_kernel=get_kernel(80, 80))
    sd = torch.load(f"{S}/resources/anti_spoof_models/{name}.pth", map_location="cpu")
    sd = collections.OrderedDict((k[7:] if k.startswith("module.") else k, v) for k, v in sd.items())
    m.load_state_dict(sd); m.eval()
    class W(torch.nn.Module):
        def __init__(s, m): super().__init__(); s.m = m
        def forward(s, x): return torch.softmax(s.m(x), dim=1)
    x = torch.rand(1, 3, 80, 80) * 255
    torch.onnx.export(W(m), x, f"{out}/{name}.onnx", input_names=["input"], output_names=["prob"], opset_version=13, dynamo=False)
    print("ok", name, os.path.getsize(f"{out}/{name}.onnx"))
