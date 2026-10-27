# Meal photo → nutrition API: deployment guide

The API takes one photo of a meal and finds every dish on the plate: roti, dal, sabzi, rice, curd, salad, idli and so on. For each dish it returns the dish name, a count or number of servings, and calories, protein, carbohydrate, fat and fibre. It also returns totals for the whole meal.

- **It runs on CPU only.** No GPU is needed. One photo takes about 1–1.5 s on 4–8 CPU cores.
- **Everything is in this folder.** The two models (about 390 MB) and the nutrition table are included, so nothing is downloaded at run time.
- **Interactive API docs** are at `/docs` once the server is running.

| In this package | What it is |
|---|---|
| `plate/` | API code (`plate/api.py`) and the analysis pipeline |
| `models/yolo/detector.pt` | dish detector: finds each bowl, plate or piece (YOLO11m, fine-tuned) |
| `models/plate_classifier/` | dish classifier: names each box, 110 dishes (SigLIP-B/16, fine-tuned) |
| `data/nutrition/dish_nutrition.csv` | nutrition per 100 g and standard serving for 113 dishes, with the source of every value |
| `Dockerfile`, `requirements-serve.txt` | how to build and run it |
| `api_smoke_test.py`, `samples/` | end-to-end check against a running server, with 4 held-out meal photos |
| `VERSION.json`, `SHA256SUMS.txt` | model version and file checksums |

**Accuracy** was measured on 136 real user meal photos that were never used in training (330 dishes):

| What was measured | Result |
|---|---|
| Dishes found | 84% |
| Found dishes named correctly | 75% (90% have the right dish in the top 3) |
| Meal calories vs. the labelled meal | median error 8%; within 20% for 64% of meals |

Portions are standard household servings: one katori, one roti, one glass. They are not measured from the photo. The full report is in *Meal_Nutrition_from_Photo_Report.pdf*.

---

## 1. Run it with Docker (recommended)

First get the files onto the server. Either unzip `food-plate-api.zip`, or clone the private GitHub repository (ask for access). The model files are stored with Git LFS, so install git-lfs before cloning:

```bash
git lfs install
git clone https://github.com/dhruv-upmanyu/food-plate-api.git && cd food-plate-api
```

Then build and start it:

```bash
docker build -t food-plate-api .          # roughly 2-3 GB image; several minutes the first time
docker run -d --name food-api --restart unless-stopped -p 8000:8000 \
  -e PLATE_API_KEY=change-me food-plate-api
```

**The Docker build hasn't been run yet.** Docker wasn't available on the development laptop. The same code, models and pinned package versions were tested by running the API directly, as in section 2, with all checks passing. Please treat the first `docker build` as the first test of the Dockerfile, and send back the error output if it fails.

The models take about 10–20 s to load. After that, `http://<server>:8000/health` returns `"status": "ok"` and the interactive docs are at `http://<server>:8000/docs`.

## 2. Run it without Docker (Linux, macOS or Windows; Python 3.10)

```bash
python3.10 -m venv venv && . venv/bin/activate          # Windows: venv\Scripts\activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.11.0 torchvision==0.26.0
pip install -r requirements-serve.txt
uvicorn plate.api:app --host 0.0.0.0 --port 8000
```

On a bare Linux server, OpenCV also needs two system libraries: `sudo apt install libgl1 libglib2.0-0`.

## 3. Check that it works

```bash
python api_smoke_test.py --url http://<server>:8000 --key change-me     # needs only Python, no packages
```

The script checks `/health`, `/v1/dishes` and the error handling, then analyses the 4 photos in `samples/`. It prints every dish with its nutrition and ends with `ALL CHECKS PASSED`. For example:

```
thali_chole_roti.jpg: 4 dishes, server 975 ms
  dish                         count  grams   kcal  prot  carb   fat fibre  conf
  Chole / chana masala             1    150    245   9.2  30.0  10.3   7.1  0.71
  Roti / chapati                   1     40     81   2.4  14.3   1.4   2.5  1.00
  Curd (dahi)                      1    100     61   3.5   4.7   3.2   0.0  0.98
  Salad (cucumber, tomato, oni     1    100     30   1.0   5.5   0.3   2.6  1.00
  MEAL TOTAL                            390    417  16.1  54.5  15.2  12.2
```

The samples are real user photos the models never saw in training. They include typical mistakes on purpose:

| Sample | Labelled dishes | API result |
|---|---|---|
| `thali_chole_roti.jpg` | chole, roti, curd, salad | all 4 right |
| `idli_vada_sambar.jpg` | 2 idli, vada, 2 chutneys | idli ×2 and vada right; finds 1 of the 2 chutneys |
| `thali_roti_rice_6_dishes.jpg` | rice, roti, mixed veg, dry sabzi, curd, salad | 5 of 6 right; the dry sabzi is named egg bhurji |
| `rajma_rice_bhindi.jpg` | rajma, rice, bhindi, curd, salad | rice, bhindi, curd, salad right; the rajma poured over the rice is missed and that region counts as plain rice (a known weakness, see section 9) |

---

## 4. API reference

If `PLATE_API_KEY` is set, every `/v1/...` request must send the header `X-API-Key: <key>`. `/health` stays open for load balancers.

### `POST /v1/analyze`

Send the photo as a multipart form upload in a field named `image`. JPEG, PNG and WebP are accepted, up to 15 MB and 50 megapixels.

```bash
curl -H "X-API-Key: change-me" -F "image=@meal.jpg" http://<server>:8000/v1/analyze
```

```python
import requests
r = requests.post("http://<server>:8000/v1/analyze", headers={"X-API-Key": "change-me"},
                  files={"image": open("meal.jpg", "rb")}, timeout=60)
meal = r.json()
```

```javascript
const form = new FormData();
form.append("image", fileInput.files[0]);
const meal = await (await fetch("https://<server>/v1/analyze", {
  method: "POST", headers: { "X-API-Key": "change-me" }, body: form })).json();
```

Response (shortened to 2 of 4 items):

```json
{
  "items": [
    {
      "dish": "chole", "name": "Chole / chana masala",
      "confidence": 0.707, "unsure": false,
      "alternatives": [{"name": "Roasted chana", "confidence": 0.207},
                       {"name": "Sprouts / chana / corn salad", "confidence": 0.024}],
      "count": 1.0, "unit": "1 katori", "grams": 150.0,
      "kcal": 245.1, "protein_g": 9.2, "carb_g": 30.0, "fat_g": 10.3, "fibre_g": 7.1,
      "boxes": [[0.059, 0.5587, 0.409, 0.8722]]
    },
    {
      "dish": "roti", "name": "Roti / chapati",
      "confidence": 0.996, "unsure": false,
      "alternatives": [{"name": "Plain paratha", "confidence": 0.002},
                       {"name": "Kathi roll / wrap", "confidence": 0.0}],
      "count": 1.0, "unit": "1 medium roti", "grams": 40.0,
      "kcal": 80.9, "protein_g": 2.4, "carb_g": 14.3, "fat_g": 1.4, "fibre_g": 2.5,
      "boxes": [[0.3519, 0.536, 0.7235, 0.9991]]
    }
  ],
  "totals": {"kcal": 416.9, "protein_g": 16.1, "carb_g": 54.5, "fat_g": 15.2, "fibre_g": 12.2, "grams": 390.0},
  "image_width": 1200, "image_height": 1600,
  "processing_ms": 1044,
  "model_version": "76ec58bc09aa"
}
```

| Field | Meaning |
|---|---|
| `items` | One entry per dish found, highest calories first. An empty list means no food was found. |
| `dish` | A stable key, for example `roti`. Use it in app logic; `name` is for display. |
| `confidence`, `unsure` | `unsure` is true when confidence is below 0.4. In that case, show `alternatives` or ask the user to confirm the dish. |
| `count`, `unit`, `grams` | `count` is the number of pieces for countable dishes (roti, idli, egg, …) and the number of servings for everything else. `grams` = count × standard serving. Let users edit the count or portion: the nutrition scales linearly. |
| `carb_g` | Available carbohydrate, with fibre not included. |
| `boxes` | Where the dish is, as `[x1, y1, x2, y2]` fractions (0–1) of the photo after EXIF rotation, so the app can draw them. |
| `model_version` | Fingerprint of the model files. Store it with each result. |

| Errors | When |
|---|---|
| `400` | The file is empty or isn't a readable image. |
| `401` | The API key is missing or wrong. |
| `413` | The photo is over the size limit. |
| `422` | The `image` form field is missing. |

All errors return JSON in the form `{"detail": "..."}`.

### `GET /v1/dishes`

This returns all 113 dishes in the nutrition table, with `serving`, `serving_g`, `per_100g`, `per_serving` and `source`. `recognised: false` marks the 3 dishes that are in the nutrition table but not yet recognised from photos: curd rice, rasam and paneer bhurji. The app can use this list to let users fix a wrong dish or add a missed one.

### `GET /health`

```json
{"status": "ok", "api_version": "1.0.0", "model_version": "76ec58bc09aa", "dishes_recognised": 110, "cpu_threads": 8}
```

---

## 5. Configuration (environment variables, all optional)

| Variable | Default | Purpose |
|---|---|---|
| `PLATE_API_KEY` | none (open) | Requires the `X-API-Key` header on `/v1/*`. **Set it on any public server.** |
| `PLATE_CORS_ORIGINS` | none | Comma-separated website origins allowed to call the API from a browser, e.g. `https://app.example.com`. Not needed for mobile apps or server-to-server calls. |
| `PLATE_MAX_UPLOAD_MB` | 15 | Largest upload accepted. |
| `PLATE_TORCH_THREADS` | all CPUs the process can use | CPU threads per worker. Set this when running in a container limited with `--cpus`, or when running several workers (see section 6). |
| `PLATE_DET_CONF` | 0.25 | Detector threshold. 0.25 was chosen on validation photos; lower values find more dishes but also more false ones. |

## 6. Sizing and scaling

These timings are per photo, with the models already loaded. They were measured on the development laptop's CPU; cloud vCPUs are usually somewhat slower.

| CPU threads | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| Median time per photo | 5.5 s | 2.8 s | 1.5 s | 1.0 s |

- **RAM** is about 1.2 GB per worker process. A 12-megapixel phone photo doesn't change this much: it takes about 1.4 s.
- **One worker analyses one photo at a time.** Other requests wait in a queue, so throughput is about 1 / (time per photo) per worker.
- **A good test server is 4 vCPU with 4–8 GB RAM**, running one worker: about 1.5 s per photo. For more traffic, add machines or containers behind a load balancer. Alternatively, run more workers on one bigger machine: `uvicorn plate.api:app --workers 2 ...` with `PLATE_TORCH_THREADS` = vCPUs ÷ 2, plus 1.2 GB RAM per worker.
- **Serverless containers** (Cloud Run, App Runner, Azure Container Apps and similar) work, but each new instance spends 10–20 s loading the models. Keep at least one instance always running.
- **A GPU isn't needed** at this scale.

## 7. Putting it on the internet

- **Use HTTPS.** Put a reverse proxy or the platform's load balancer in front of the API. With Caddy, which handles certificates itself, the whole Caddyfile is:

  ```
  food.example.com {
      reverse_proxy 127.0.0.1:8000
  }
  ```

  With nginx, set `client_max_body_size 16m;` and `proxy_read_timeout 60s;`.
- **Set `PLATE_API_KEY`**, and call the API from your app's backend so the key never ships inside a mobile app. Rate-limit at the proxy.
- **Privacy:** photos are processed in memory and never written to disk or logged. The access log records only the path, status and client IP. If you later want to keep photos to improve the model, get users' consent first.
- **Ultralytics phone-home is disabled** by the API and the Dockerfile (`YOLO_OFFLINE`, `YOLO_AUTOINSTALL=False`), so the library never sends usage analytics or pip-installs packages at run time.

## 8. Updating models or nutrition values

- **To replace the models or the nutrition table**, swap the files in `models/` or `data/nutrition/dish_nutrition.csv` and rebuild or restart. `model_version` in `/health` changes automatically.
- **Retraining** (new dishes or more photos) is done in the project repository on a free Kaggle GPU: `kaggle/train_plate.py`. The commands are in the appendix of the report. Adding a dish takes three things: a taxonomy entry, nutrition values, and photos of the dish.

## 9. Known limitations

- **Portions** are standard servings, not measured from the photo. A big bowl and a small bowl of dal give the same numbers. The app should let users adjust them.
- **Gravy over rice** (rajma-chawal, dal-chawal on one plate) is often read as plain rice.
- **Touching pieces**, such as a pile of momos or 3 overlapping rotis, may be counted as 1.
- About **1 in 4 found dishes gets the wrong name**. Usually it is a similar dish, such as mixed veg vs. dry sabzi or dal vs. sambar, and the right one is in the top 3 for 90% of found dishes. Showing `alternatives` for `unsure` items helps.
- **Only 110 dishes are recognised.** A dish outside this list is named as the closest one it knows. Expanding the dish list is the next phase of work.
- The training and test labels were produced with AI assistance and have **not yet been checked by a person**.

## 10. Licences: check before a commercial launch

| Component | Licence | What it means |
|---|---|---|
| Ultralytics YOLO library and YOLO11 weights (the detector was fine-tuned from them) | **AGPL-3.0** | If the service is offered to outside users, AGPL requires making the service's source code available to them. The alternative is an Ultralytics Enterprise licence. Internal testing is fine. If neither option is acceptable, the detector can be retrained on a permissively licensed model, for example RT-DETR in Hugging Face transformers (Apache-2.0); that needs retraining and re-testing. |
| SigLIP base model, transformers, PyTorch, FastAPI, uvicorn | Apache-2.0 / BSD / MIT | Fine for commercial use. |
| Training photos | Company user photos plus 6 public food datasets | Some of the public datasets don't state a licence. Check them before commercial use; the list is in the report. |
| Nutrition values | INDB, IFCT 2017 (NIN-ICMR), USDA FoodData Central (public domain) | The source of each dish's values is in `dish_nutrition.csv`. Check the INDB and IFCT terms before showing the values in a commercial product. |

## 11. Troubleshooting

| Symptom | Fix |
|---|---|
| `ImportError: libGL.so.1` (bare Linux) | `sudo apt install libgl1 libglib2.0-0` |
| More than 3 s per photo | Check `cpu_threads` in `/health`. Give the container more CPUs and set `PLATE_TORCH_THREADS` to match. |
| Container killed, or out of memory | Allow about 1.2 GB per worker; use fewer workers or more RAM. |
| `413` on large photos | Raise `PLATE_MAX_UPLOAD_MB` and the proxy's body-size limit, or resize the photo on the phone (about 1600 px on the long side is plenty). |
| `400` for iPhone HEIC files | Send JPEG; browsers and most upload libraries convert HEIC automatically. Or `pip install pi-heif==1.4.0`, which is optional and was not tested in this build. |
| Health check fails right after start | The models are still loading; allow 60–90 s. |
| `... is a Git LFS pointer` at start | The repository was cloned without git-lfs. Run `git lfs install && git lfs pull`, then rebuild. |
| Wrong file or corrupted copy | Run `sha256sum -c SHA256SUMS.txt` in the package folder. |
