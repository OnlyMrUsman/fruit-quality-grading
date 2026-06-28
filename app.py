import os
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
import gradio as gr

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


# 1. Load model once

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_PATH = "outputs/checkpoints/best_model.pth"
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

def build_model(num_classes, dropout_p=0.3):
    m = models.resnet18(weights=None)
    m.fc = nn.Sequential(nn.Dropout(dropout_p), nn.Linear(m.fc.in_features, num_classes))
    return m

ckpt    = torch.load(CKPT_PATH, map_location=DEVICE)
CLASSES = ckpt["classes"]
model   = build_model(len(CLASSES)).to(DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()
cam = GradCAM(model=model, target_layers=[model.layer4[-1]])


# 2. Helpers

EMOJI = {"Apple": "🍎", "Banana": "🍌", "Orange": "🍊"}
COND  = {
    "Fresh":  {"accent": "#059669", "tint": "#ecfdf5", "note": "Ready to eat"},
    "Unripe": {"accent": "#d97706", "tint": "#fffbeb", "note": "Needs more time"},
    "Rotten": {"accent": "#dc2626", "tint": "#fef2f2", "note": "Do not consume"},
    "Unknown":{"accent": "#475569", "tint": "#f1f5f9", "note": ""},
}

def parse_class(cls_name):
    n = cls_name.lower().replace(" ", "")
    condition = ("Fresh" if n.startswith("fresh") else "Rotten" if n.startswith("rotten")
                 else "Unripe" if n.startswith("unripe") else "Unknown")
    fruit = ("Apple" if "apple" in n else "Banana" if "banana" in n
             else "Orange" if "orange" in n else "Unknown")
    return fruit, condition

def predict(image):
    if image is None:
        return ("<div class='verdict empty'>Upload a fruit image to see the analysis</div>", {}, None)
    image = image.convert("RGB")
    x = eval_transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()

    top  = int(probs.argmax()); conf = float(probs[top])
    fruit, condition = parse_class(CLASSES[top])
    c = COND[condition]; deg = conf * 360

    hero = f"""
    <div class='verdict' style='background:{c["tint"]}'>
      <div class='emoji'>{EMOJI.get(fruit,'❓')}</div>
      <div class='fruit'>{fruit}</div>
      <div class='pill' style='background:{c["accent"]}'>{condition}</div>
      <div class='note'>{c["note"]}</div>
      <div class='ring' style='background:conic-gradient({c["accent"]} {deg:.1f}deg,#e9eef4 0deg)'>
        <div class='ring-in'>
          <span class='ring-num'>{conf*100:.0f}<small>%</small></span>
          <span class='ring-lbl'>confidence</span>
        </div>
      </div>
    </div>
    """
    confidences = {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}
    rgb = np.array(image.resize((224, 224)), dtype=np.float32) / 255.0
    overlay = show_cam_on_image(rgb, cam(input_tensor=x,
                                         targets=[ClassifierOutputTarget(top)])[0], use_rgb=True)
    return hero, confidences, overlay

# safely collect a few example images from the test set
def _first_img(folder):
    try:
        fs = [f for f in sorted(os.listdir(folder)) if f.lower().endswith((".jpg",".jpeg",".png"))]
        return os.path.join(folder, fs[0]) if fs else None
    except Exception:
        return None
EXAMPLES = [[p] for p in [
    _first_img("data/test/freshapples"), _first_img("data/test/rottenbanana"),
    _first_img("data/test/unripe orange"), _first_img("data/test/freshoranges"),
] if p]


# 3. Styling

CSS = """
.gradio-container{background:radial-gradient(1200px 600px at 20% -10%,#eafaf1 0%,transparent 55%),
                  radial-gradient(1000px 500px at 100% 0%,#e8f0fe 0%,transparent 50%),
                  #f5f8fb !important; font-family:'Inter','Segoe UI',system-ui,sans-serif;}
#hero-head{text-align:center; padding:30px 12px 10px;}
#hero-head h1{font-size:32px; font-weight:800; margin:0; letter-spacing:-.5px;
  background:linear-gradient(90deg,#059669,#2563eb); -webkit-background-clip:text; -webkit-text-fill-color:transparent;}
#hero-head .sub{color:#64748b; font-size:14.5px; margin-top:8px;}
#hero-head .chips{margin-top:12px; display:flex; gap:8px; justify-content:center; flex-wrap:wrap;}
#hero-head .chip{background:#fff; border:1px solid #e2e8f0; color:#475569; font-size:12px;
  font-weight:600; padding:5px 12px; border-radius:30px; box-shadow:0 1px 3px rgba(30,60,100,.05);}
.card{background:#fff !important; border-radius:20px !important; padding:18px !important;
  border:1px solid #e8eef4 !important; box-shadow:0 10px 30px rgba(30,60,100,.07) !important;}
.verdict{border-radius:16px; padding:26px 18px 30px; text-align:center; transition:background .4s ease;}
.verdict.empty{color:#94a3b8; font-size:15px; padding:80px 18px; background:#f8fafc; border:2px dashed #e2e8f0;}
.verdict .emoji{font-size:66px; line-height:1;}
.verdict .fruit{font-size:30px; font-weight:800; color:#0f172a; margin-top:4px; letter-spacing:-.5px;}
.verdict .pill{display:inline-block; color:#fff; font-weight:700; font-size:14px; letter-spacing:.6px;
  padding:6px 20px; border-radius:30px; margin-top:12px; text-transform:uppercase;}
.verdict .note{color:#64748b; font-size:13.5px; margin-top:8px;}
.ring{width:128px; height:128px; border-radius:50%; margin:22px auto 0; display:flex;
  align-items:center; justify-content:center;}
.ring-in{width:98px; height:98px; border-radius:50%; background:#fff; display:flex; flex-direction:column;
  align-items:center; justify-content:center; box-shadow:inset 0 1px 4px rgba(0,0,0,.06);}
.ring-num{font-size:28px; font-weight:800; color:#0f172a;}
.ring-num small{font-size:15px; font-weight:700; color:#64748b;}
.ring-lbl{font-size:10.5px; color:#94a3b8; text-transform:uppercase; letter-spacing:1px; margin-top:2px;}
button.primary{background:linear-gradient(90deg,#059669,#2563eb) !important; border:none !important;
  font-weight:700 !important; letter-spacing:.3px !important; box-shadow:0 6px 18px rgba(37,99,235,.25) !important;}
#foot{text-align:center; color:#94a3b8; font-size:12px; padding:16px;}
"""

FORCE_LIGHT = """
() => {
  const url = new URL(window.location.href);
  if (url.searchParams.get('__theme') !== 'light') {
    url.searchParams.set('__theme','light'); window.location.replace(url.href);
  }
}
"""

with gr.Blocks(title="Fruit Quality Grader") as demo:
    gr.HTML("""<div id='hero-head'>
        <h1>🍏 Fruit Ripeness &amp; Defect Grader</h1>
        <div class='sub'>Upload a fruit image — the model grades its ripeness and condition in real time.</div>
        <div class='chips'><span class='chip'>ResNet18</span><span class='chip'>9 classes</span>
            <span class='chip'>Grad-CAM explainability</span><span class='chip'>GPU accelerated</span></div>
    </div>""")

    with gr.Row(equal_height=True):
        with gr.Column(scale=5):
            with gr.Group(elem_classes="card"):
                inp = gr.Image(type="pil", label="Fruit image", height=330, sources=["upload", "clipboard"])
                btn = gr.Button("🔍  Analyze image", variant="primary", size="lg")
                if EXAMPLES:
                    gr.Examples(examples=EXAMPLES, inputs=inp, label="Or try a sample")
        with gr.Column(scale=6):
            with gr.Group(elem_classes="card"):
                hero = gr.HTML("<div class='verdict empty'>Upload a fruit image to see the analysis</div>")
                lab  = gr.Label(num_top_classes=3, label="Top class probabilities")
                cam_o = gr.Image(label="Grad-CAM · where the model focused", height=230)

    gr.HTML("<div id='foot'>Final Project · Deep Learning-Based Fruit Ripeness & Defect Grading</div>")

    btn.click(predict, inputs=inp, outputs=[hero, lab, cam_o])
    inp.change(predict, inputs=inp, outputs=[hero, lab, cam_o])

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(primary_hue="emerald", secondary_hue="blue", radius_size="lg"),
                css=CSS, js=FORCE_LIGHT, server_name="127.0.0.1", server_port=7860, share=False)