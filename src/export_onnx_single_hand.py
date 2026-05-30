import os
import torch
import joblib

from train_single_hand import (
    ABLATION_CONFIGS,
    build_model,
)

class ClassifierOnlyWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        logits, _, _ = self.model(x)
        return logits

def export_onnx():
    model_path = "./results_single_hand/models/full_model_seed42.pth"
    label_encoder_path = "./results_single_hand/label_encoder_full_model_seed42.pkl"
    onnx_path = "./results_single_hand/full_model_single_hand_classifier.onnx"

    cfg = ABLATION_CONFIGS[-1]

    label_encoder = joblib.load(label_encoder_path)
    num_classes = len(label_encoder.classes_)

    model = build_model(cfg, num_classes=num_classes)
    model.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True)
    )
    model.eval()

    wrapped_model = ClassifierOnlyWrapper(model)
    wrapped_model.eval()

    dummy = torch.randn(1, cfg.num_landmarks, cfg.coord_dim)

    torch.onnx.export(
        wrapped_model,
        dummy,
        onnx_path,
        input_names=["landmarks"],
        output_names=["logits"],
        dynamic_axes={
            "landmarks": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    print(f"Saved ONNX model to: {onnx_path}")
    print(f"ONNX model size: {os.path.getsize(onnx_path) / 1024 / 1024:.4f} MB")

if __name__ == "__main__":
    export_onnx()