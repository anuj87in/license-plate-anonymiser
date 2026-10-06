# License Plate Anonymiser

Blur license plates in photos, directly in the browser.

**Live demo:** https://anuj87in.github.io/license-plate-anonymiser/

**Full technical report:** [License_Plate_Anonymisation_Report.pdf](report/License_Plate_Anonymisation_Report.pdf)
(data audit, exploratory analysis, model design, training and evaluation; 31 pages)

A YOLOv8s detection model runs on the visitor's own device through
[ONNX Runtime Web](https://onnxruntime.ai/) (WebGPU where available, WebAssembly otherwise).
Images are never uploaded: there is no server, no database and no tracking.

## How it works

1. **Detect.** The photo is letterboxed to 640 × 640 px and passed to the model; boxes are
   filtered by confidence and non-maximum suppression (IoU 0.5).
2. **Pad.** Each box is enlarged by 15% on every side, so a slightly tight box cannot leave a
   plate edge readable.
3. **Blur.** A Gaussian blur as wide as the plate is applied inside each box only. The rest of
   the image is left untouched.

The confidence threshold starts at its lowest setting, 0.05 (recall first: a missed plate is a
privacy risk, an extra blur is harmless). This also blurs plates from countries the model saw little
of during training; raising the slider removes unneeded blurs. Detections below 0.5 are flagged for
review on screen.

| Threshold | Plates fully covered | Plates missed | Blur boxes on regions with no plate |
|---|---|---|---|
| 0.05 (default) | 92.2% | 23 | 119 |
| 0.1 | 90.8% | 30 | 79 |
| 0.2 | 90.4% | 34 | 51 |
| 0.5 | 82.8% | 84 | 16 |

Measured with the web model on the test set (386 images, 512 plates).

## Model

| | |
|---|---|
| Architecture | YOLOv8s, fine-tuned for 90 epochs (best checkpoint: epoch 47) |
| Training data | about 25,000 labelled images, single class `license_plate` |
| Input size | 640 × 640 |
| Test set | 386 real-world street images, 512 plates |
| Precision / recall | 0.920 / 0.879 |
| mAP@50 / mAP@50-95 | 0.921 / 0.682 |
| Plates fully covered by the blur at threshold 0.2 (original PyTorch model) | 90.2% |
| Web model | `model/plate_fp16.onnx`, FP16 weights (22 MB), verified to match the original model's coverage on the test set |

Very small or distant plates (under about 40 px wide) are the most likely to be missed. Always
review the output before publishing a photo.

## Run locally

Any static file server works, for example:

```bash
python3 -m http.server 8000
# open http://localhost:8000
```

Add `?backend=wasm` to the URL to force the WebAssembly backend.

## Files

```
index.html             page and styles
app.js                 pre-processing, inference, NMS and blur
model/plate_fp16.onnx  detection model
samples/               example photos (optional)
report/                full technical report (PDF)
```

## Licence

AGPL-3.0, as required by [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics),
which was used to train the model. See [LICENSE](LICENSE).
