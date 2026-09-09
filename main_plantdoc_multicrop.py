from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from ultralytics import YOLO
from PIL import Image
import io
import os
import base64
import threading

app = FastAPI(title="SmartCrop AI")


# ============================================================
# SMARTCROP AI - MODEL CONFIGURATION
# ============================================================

MODEL_PATH = r"C:\Users\unuku\OneDrive\Documents\smartcropai\dataset\PlantDoc\PlantDoc-Dataset-windows-compatible-master\runs\classify\train-3\weights\best.pt"

# Detection settings
MODEL_CONFIDENCE = 0.20
MODEL_IMAGE_SIZE = 640
MODEL_IOU = 0.45
MAX_DETECTIONS = 50


# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"\nYOLO model not found!\n\n"
        f"Expected model at:\n{MODEL_PATH}\n\n"
        f"Please check MODEL_PATH in main.py."
    )

print("=" * 60)
print("SMARTCROP AI")
print("=" * 60)
print("Loading YOLO model...")
print(f"Model: {MODEL_PATH}")

model = YOLO(MODEL_PATH)

print("Model loaded successfully.")
print("Classes:", model.names)
print("=" * 60)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_class_name(name):
    """
    Convert model class names into farmer-friendly text.
    """
    name = str(name)

    replacements = {
        "_": " ",
        "-": " ",
        "___": " ",
    }

    for old, new in replacements.items():
        name = name.replace(old, new)

    return " ".join(name.split()).title()


def is_healthy(name):
    """
    Detect whether the model class represents a healthy plant.
    """
    text = str(name).lower()

    healthy_words = [
        "healthy",
        "normal",
        "good",
        "fresh"
    ]

    return any(word in text for word in healthy_words)


def get_guidance(name):
    """
    Farmer-friendly guidance.
    """

    text = str(name).lower()

    if "septoria" in text:
        return (
            "Septoria leaf spot may be present. "
            "Remove badly affected leaves, avoid overhead watering, "
            "and maintain good air circulation."
        )

    if "late blight" in text:
        return (
            "Possible late blight detected. "
            "Remove severely affected plant parts and avoid wetting "
            "the leaves unnecessarily."
        )

    if "early blight" in text:
        return (
            "Possible early blight detected. "
            "Remove affected leaves and maintain good spacing "
            "between plants."
        )

    if "leaf mold" in text:
        return (
            "Possible leaf mold detected. "
            "Improve ventilation and avoid excess moisture on leaves."
        )

    if "healthy" in text:
        return (
            "The plant appears healthy. "
            "Continue regular watering, nutrition and crop monitoring."
        )

    return (
        "A possible crop problem was detected. "
        "Inspect nearby plants and consult a local agriculture expert "
        "before applying treatment."
    )


def image_to_base64(image):
    """
    Convert PIL image to base64 for displaying in browser.
    """

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=90
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return "data:image/jpeg;base64," + encoded


# ============================================================
# HOME PAGE
# ============================================================

@app.get("/", response_class=HTMLResponse)
def home():

    return """
<!DOCTYPE html>

<html>

<head>

<title>SmartCrop AI</title>

<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<style>

* {
    box-sizing: border-box;
}

body {

    margin: 0;

    font-family:
        Arial,
        Helvetica,
        sans-serif;

    background:
        linear-gradient(
            rgba(238,247,237,0.92),
            rgba(238,247,237,0.92)
        );

    color: #243b25;
}


/* =========================================================
   HEADER
========================================================= */

header {

    background:
        linear-gradient(
            135deg,
            #1b5e20,
            #43a047
        );

    color: white;

    padding: 18px 25px;

    display: flex;

    justify-content: space-between;

    align-items: center;
}

.logo {

    font-size: 25px;

    font-weight: bold;
}

.language {

    background:
        rgba(255,255,255,0.18);

    padding: 8px 14px;

    border-radius: 20px;
}


/* =========================================================
   HERO
========================================================= */

.hero {

    text-align: center;

    padding:
        35px 15px 20px;
}

.hero h1 {

    font-size: 34px;

    color: #1b5e20;

    margin-bottom: 8px;
}

.hero p {

    color: #557255;

    font-size: 17px;
}


/* =========================================================
   CONTAINER
========================================================= */

.container {

    max-width: 950px;

    margin: auto;

    padding: 15px;
}

.card {

    background: white;

    border-radius: 22px;

    padding: 25px;

    margin-bottom: 20px;

    box-shadow:
        0 5px 20px
        rgba(0,0,0,0.08);
}

.card h2 {

    color: #2e7d32;

    margin-top: 0;
}


/* =========================================================
   BUTTON
========================================================= */

.scan-button {

    width: 100%;

    padding: 18px;

    border: none;

    border-radius: 15px;

    background: #2e7d32;

    color: white;

    font-size: 20px;

    font-weight: bold;

    cursor: pointer;

    margin: 10px 0;
}

.scan-button:hover {

    background: #1b5e20;
}

.stop-button {

    padding: 12px 20px;

    border: none;

    border-radius: 10px;

    background: #d32f2f;

    color: white;

    font-size: 16px;

    cursor: pointer;

    margin-top: 10px;
}


/* =========================================================
   CAMERA
========================================================= */

.camera-area {

    text-align: center;
}

.video-wrapper {

    position: relative;

    width: 100%;

    max-width: 700px;

    margin: auto;
}

video {

    width: 100%;

    border-radius: 18px;

    background: #111;

    margin-top: 15px;
}

#overlay {

    position: absolute;

    left: 0;

    top: 15px;

    width: 100%;

    height: calc(100% - 15px);

    pointer-events: none;
}


/* =========================================================
   UPLOAD
========================================================= */

.upload-box {

    border:
        2px dashed #81c784;

    border-radius: 15px;

    padding: 25px;

    text-align: center;

    background: #f8fff7;
}

input[type="file"] {

    margin: 15px 0;

    width: 100%;
}

.upload-button {

    padding: 13px 20px;

    border: none;

    border-radius: 10px;

    background: #66bb6a;

    color: white;

    font-size: 16px;

    cursor: pointer;
}


/* =========================================================
   RESULT
========================================================= */

#result {

    margin-top: 20px;
}

.summary-box {

    background: #e8f5e9;

    border-radius: 14px;

    padding: 14px;

    margin: 12px 0;

    font-size: 17px;

    font-weight: bold;
}

.box-info {

    font-size: 15px;

    color: #557255;

    margin-top: 6px;
}

.result-card {

    padding: 22px;

    border-radius: 18px;

    background: #f1f8e9;

    text-align: center;
}

.detection {

    font-size: 25px;

    font-weight: bold;

    color: #d32f2f;

    margin: 10px;
}

.healthy {

    font-size: 25px;

    font-weight: bold;

    color: #2e7d32;

    margin: 10px;
}

.confidence {

    font-size: 20px;

    font-weight: bold;
}


/* =========================================================
   IMAGE RESULT
========================================================= */

.result-image {

    width: 100%;

    max-width: 700px;

    border-radius: 15px;

    margin-top: 15px;

    box-shadow:
        0 4px 15px
        rgba(0,0,0,0.15);
}


/* =========================================================
   FEATURES
========================================================= */

.features {

    display: grid;

    grid-template-columns:
        repeat(3, 1fr);

    gap: 15px;
}

.feature {

    background: white;

    padding: 20px;

    border-radius: 18px;

    text-align: center;

    box-shadow:
        0 3px 12px
        rgba(0,0,0,0.06);
}

.feature-icon {

    font-size: 35px;
}

.feature h3 {

    color: #2e7d32;
}


/* =========================================================
   STATUS
========================================================= */

.loading {

    font-size: 20px;

    font-weight: bold;

    color: #2e7d32;
}

.warning {

    background: #fff3cd;

    padding: 15px;

    border-radius: 12px;

    margin-top: 15px;

    color: #795548;
}

.success {

    background: #e8f5e9;

    padding: 15px;

    border-radius: 12px;

    margin-top: 15px;
}


/* =========================================================
   FOOTER
========================================================= */

footer {

    text-align: center;

    padding: 25px;

    color: #668066;
}


/* =========================================================
   MOBILE
========================================================= */

@media(max-width:650px) {

    .hero h1 {

        font-size: 27px;
    }

    .features {

        grid-template-columns: 1fr;
    }

    .card {

        padding: 18px;
    }

    header {

        flex-direction: column;

        gap: 10px;
    }
}

</style>

</head>


<body>


<header>

<div class="logo">

🌾 SmartCrop AI

</div>

<div class="language">

🌐 తెలుగు / English

</div>

</header>



<div class="hero">

<h1>🌱 Protect Your Crop</h1>

<p>
AI-powered crop disease and pest detection
</p>

</div>



<div class="container">


<!-- =====================================================
     LIVE CAMERA
===================================================== -->

<div class="card">

<h2>📷 Live Crop Scanner</h2>

<p>
Point your camera towards a tomato plant or leaf.
SmartCrop AI will continuously analyze the image
and identify possible crop problems.
</p>


<div class="camera-area">

<button
class="scan-button"
onclick="startCamera()">

📷 Start Live Crop Scan

</button>


<div class="video-wrapper">

<video
id="camera"
autoplay
playsinline
style="display:none;">
</video>

<canvas
id="overlay"
style="display:none;">
</canvas>

</div>


<br>


<button
class="stop-button"
onclick="stopCamera()">

🛑 Stop Camera

</button>

</div>

</div>



<!-- =====================================================
     IMAGE UPLOAD
===================================================== -->

<div class="card">

<h2>🖼️ Upload Crop Photo</h2>

<div class="upload-box">

<p>
Upload a clear tomato plant or leaf image.
</p>

<input
type="file"
id="image"
accept="image/*"
onchange="previewImage()">

<div id="preview"></div>

<br>

<button
class="upload-button"
onclick="scanCrop()">

🔍 Analyze Crop

</button>

</div>

</div>



<!-- =====================================================
     RESULT
===================================================== -->

<div class="card">

<h2>🌿 AI Detection Result</h2>

<div id="result">

<div class="result-card">

<p>
Upload an image or start the camera
to analyze your crop.
</p>

</div>

</div>

</div>



<!-- =====================================================
     FEATURES
===================================================== -->

<div class="features">


<div class="feature">

<div class="feature-icon">
📷
</div>

<h3>
Live Scan
</h3>

<p>
Scan crops directly using your camera.
</p>

</div>



<div class="feature">

<div class="feature-icon">
📦
</div>

<h3>
Object Detection
</h3>

<p>
AI identifies affected areas using
bounding boxes.
</p>

</div>



<div class="feature">

<div class="feature-icon">
🌾
</div>

<h3>
Farmer Friendly
</h3>

<p>
Simple and easy-to-understand results.
</p>

</div>


</div>


</div>



<footer>

🌱 SmartCrop AI |
Early Crop Disease Detection

</footer>



<script>


// ========================================================
// GLOBAL VARIABLES
// ========================================================

let cameraStream = null;

let scanning = false;

let liveRequestRunning = false;

let liveTimer = null;


// ========================================================
// IMAGE PREVIEW
// ========================================================

function previewImage() {

    const input =
        document.getElementById("image");

    const preview =
        document.getElementById("preview");

    if (!input.files.length) {

        preview.innerHTML = "";

        return;
    }

    const file =
        input.files[0];

    const url =
        URL.createObjectURL(file);

    preview.innerHTML = `

        <img
            src="${url}"
            class="result-image"
            alt="Selected crop image">

    `;
}


// ========================================================
// UPLOAD IMAGE
// ========================================================

async function scanCrop() {

    const input =
        document.getElementById("image");

    const result =
        document.getElementById("result");


    if (!input.files.length) {

        alert(
            "Please select a crop image first!"
        );

        return;
    }


    result.innerHTML = `

        <div class="result-card">

            <div class="loading">

                ⏳ AI is analyzing your crop...

            </div>

            <p>
                Please wait...
            </p>

        </div>

    `;


    const formData =
        new FormData();

    formData.append(
        "file",
        input.files[0]
    );


    try {

        const response =
            await fetch(
                "/predict",
                {
                    method: "POST",
                    body: formData
                }
            );


        if (!response.ok) {

            throw new Error(
                "Server error: " +
                response.status
            );
        }


        const data =
            await response.json();


        if (data.error) {

            result.innerHTML = `

                <div class="result-card">

                    ❌ ${data.error}

                </div>

            `;

            return;
        }


        showResult(data);


    } catch (error) {

        console.error(error);

        result.innerHTML = `

            <div class="result-card">

                ❌ Unable to analyze image.

                <p class="box-info">

                    Make sure the FastAPI server
                    is running correctly.

                </p>

            </div>

        `;
    }
}


// ========================================================
// SHOW RESULT
// ========================================================

function showResult(data) {

    const result =
        document.getElementById("result");


    const detections =
        data.detections || [];


    // ----------------------------------------------------
    // NO DETECTION
    // ----------------------------------------------------

    if (detections.length === 0) {

        result.innerHTML = `

            <div class="result-card">

                <div style="font-size:50px;">
                    🔍
                </div>

                <h2>
                    No disease detected
                </h2>

                <p>
                    The AI could not find a
                    strong enough disease pattern
                    in this image.
                </p>

                <div class="warning">

                    ⚠️ This does NOT necessarily mean
                    the plant is healthy.

                    <br><br>

                    Try uploading a closer,
                    brighter image of the affected leaf.

                </div>

            </div>

        `;

        return;
    }


    // ----------------------------------------------------
    // GROUP DETECTIONS
    // ----------------------------------------------------

    const grouped = {};


    detections.forEach(
        function(item) {

            const name =
                item.name;


            if (!grouped[name]) {

                grouped[name] = {

                    count: 0,

                    maxConfidence: 0

                };

            }


            grouped[name].count += 1;


            grouped[name].maxConfidence =
                Math.max(
                    grouped[name].maxConfidence,
                    Number(item.confidence)
                );

        }
    );


    let html = `

        <div class="result-card">

            <div style="font-size:50px;">
                🤖
            </div>

            <h2>
                AI Detection Result
            </h2>

            <div class="summary-box">

                🔎
                ${detections.length}
                affected area(s) detected

            </div>

    `;


    // ----------------------------------------------------
    // DETECTION CARDS
    // ----------------------------------------------------

    Object.keys(grouped).forEach(
        function(name) {

            const item =
                grouped[name];


            const healthy =
                name
                    .toLowerCase()
                    .includes("healthy");


            html += `

                <div style="
                    background:white;
                    padding:18px;
                    border-radius:15px;
                    margin-top:12px;
                ">

                    <div class="${
                        healthy
                        ? 'healthy'
                        : 'detection'
                    }">

                        ${
                            healthy
                            ? '🟢'
                            : '🔴'
                        }

                        ${name}

                    </div>


                    <div class="confidence">

                        🎯 Confidence:
                        ${item.maxConfidence.toFixed(2)}%

                    </div>


                    <div class="box-info">

                        📍 Affected areas:
                        ${item.count}

                    </div>

                </div>

            `;

        }
    );


    // ----------------------------------------------------
    // ANNOTATED IMAGE
    // ----------------------------------------------------

    if (data.annotated_image) {

        html += `

            <div style="
                margin-top:20px;
            ">

                <h3>
                    📦 Detected Areas
                </h3>

                <img
                    src="${data.annotated_image}"
                    class="result-image"
                    alt="AI detection result">

            </div>

        `;

    }


    // ----------------------------------------------------
    // GUIDANCE
    // ----------------------------------------------------

    if (data.guidance) {

        html += `

            <div class="warning">

                <b>
                    🌱 Farmer Guidance
                </b>

                <p>
                    ${data.guidance}
                </p>

            </div>

        `;

    }


    html += `

        </div>

    `;


    result.innerHTML = html;


    // Draw camera boxes
    drawBoxes(detections);
}


// ========================================================
// DRAW BOUNDING BOXES
// ========================================================

function drawBoxes(detections) {

    const video =
        document.getElementById("camera");

    const overlay =
        document.getElementById("overlay");


    if (
        !video ||
        !overlay ||
        !video.videoWidth
    ) {

        return;
    }


    overlay.width =
        video.videoWidth;

    overlay.height =
        video.videoHeight;


    const ctx =
        overlay.getContext("2d");


    ctx.clearRect(
        0,
        0,
        overlay.width,
        overlay.height
    );


    detections.forEach(
        function(item) {

            if (
                !item.box ||
                item.box.length !== 4
            ) {

                return;
            }


            const [
                x1,
                y1,
                x2,
                y2
            ] = item.box;


            ctx.lineWidth = 4;

            ctx.strokeStyle =
                "#ff2222";

            ctx.strokeRect(
                x1,
                y1,
                x2 - x1,
                y2 - y1
            );


            const label =

                item.name +
                " " +
                Number(
                    item.confidence
                ).toFixed(1) +
                "%";


            ctx.font =
                "bold 18px Arial";


            const textWidth =
                ctx.measureText(
                    label
                ).width;


            ctx.fillStyle =
                "#ff2222";


            ctx.fillRect(

                x1,

                Math.max(
                    0,
                    y1 - 30
                ),

                textWidth + 14,

                30

            );


            ctx.fillStyle =
                "white";


            ctx.fillText(

                label,

                x1 + 7,

                Math.max(
                    21,
                    y1 - 9
                )

            );

        }
    );
}


// ========================================================
// START CAMERA
// ========================================================

async function startCamera() {

    const video =
        document.getElementById("camera");

    const overlay =
        document.getElementById("overlay");

    const result =
        document.getElementById("result");


    try {

        stopCamera();


        cameraStream =
            await navigator
                .mediaDevices
                .getUserMedia({

                    video: {

                        facingMode:
                            {
                                ideal:
                                "environment"
                            },

                        width:
                            {
                                ideal:
                                1280
                            },

                        height:
                            {
                                ideal:
                                720
                            }

                    },

                    audio: false

                });


        video.srcObject =
            cameraStream;


        video.style.display =
            "block";


        overlay.style.display =
            "block";


        scanning = true;


        result.innerHTML = `

            <div class="result-card">

                <div style="font-size:40px;">
                    📷
                </div>

                <h3>
                    Camera Started
                </h3>

                <p>
                    Show a tomato leaf clearly
                    in front of the camera.
                </p>

                <p>
                    🤖 AI will scan automatically.
                </p>

            </div>

        `;


        video.onloadedmetadata =
            function() {

                overlay.width =
                    video.videoWidth;

                overlay.height =
                    video.videoHeight;

                scanLiveFrame();

            };


    } catch (error) {

        console.error(error);

        alert(
            "Camera permission is required. " +
            "Please allow camera access."
        );

    }
}


// ========================================================
// LIVE CAMERA SCAN
// ========================================================

async function scanLiveFrame() {

    if (!scanning) {

        return;
    }


    if (liveRequestRunning) {

        scheduleNextScan();

        return;
    }


    const video =
        document.getElementById("camera");


    if (
        !video ||
        !video.videoWidth ||
        !video.videoHeight
    ) {

        scheduleNextScan();

        return;
    }


    liveRequestRunning = true;


    try {

        const canvas =
            document.createElement(
                "canvas"
            );


        canvas.width =
            video.videoWidth;

        canvas.height =
            video.videoHeight;


        const context =
            canvas.getContext("2d");


        context.drawImage(

            video,

            0,
            0,

            canvas.width,
            canvas.height

        );


        const blob =
            await new Promise(
                function(resolve) {

                    canvas.toBlob(
                        resolve,
                        "image/jpeg",
                        0.85
                    );

                }
            );


        if (!blob) {

            liveRequestRunning =
                false;

            scheduleNextScan();

            return;
        }


        const formData =
            new FormData();


        formData.append(
            "file",
            blob,
            "live_camera.jpg"
        );


        const response =
            await fetch(
                "/predict",
                {

                    method: "POST",

                    body: formData

                }
            );


        const data =
            await response.json();


        if (data.error) {

            console.error(
                data.error
            );

        } else {

            showResult(data);

        }


    } catch (error) {

        console.error(
            "Live scan error:",
            error
        );

    }


    liveRequestRunning = false;


    scheduleNextScan();
}


// ========================================================
// SCHEDULE CAMERA SCAN
// ========================================================

function scheduleNextScan() {

    if (!scanning) {

        return;
    }


    clearTimeout(
        liveTimer
    );


    liveTimer =
        setTimeout(
            scanLiveFrame,
            1500
        );
}


// ========================================================
// STOP CAMERA
// ========================================================

function stopCamera() {

    scanning = false;

    liveRequestRunning = false;


    clearTimeout(
        liveTimer
    );


    if (cameraStream) {

        cameraStream
            .getTracks()
            .forEach(
                function(track) {

                    track.stop();

                }
            );

        cameraStream = null;
    }


    const video =
        document.getElementById(
            "camera"
        );

    const overlay =
        document.getElementById(
            "overlay"
        );


    if (video) {

        video.srcObject = null;

        video.style.display =
            "none";

    }


    if (overlay) {

        overlay.style.display =
            "none";

        const ctx =
            overlay.getContext("2d");

        ctx.clearRect(
            0,
            0,
            overlay.width,
            overlay.height
        );

    }
}

</script>

</body>

</html>
"""


# ============================================================
# OBJECT DETECTION API
# ============================================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        if not file:
            return {"error": "No image uploaded."}

        contents = await file.read()
        if not contents:
            return {"error": "Uploaded image is empty."}

        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception:
            return {"error": "Invalid image file. Please upload JPG, JPEG or PNG."}

        results = model.predict(
            source=image,
            imgsz=224,
            verbose=False
        )
        result = results[0]

        if result.probs is None:
            return {"error": "The PlantDoc model could not classify this image."}

        class_id = int(result.probs.top1)
        confidence = float(result.probs.top1conf) * 100

        if isinstance(result.names, dict):
            raw_name = result.names.get(class_id, f"Class {class_id}")
        else:
            raw_name = result.names[class_id]

        name = clean_class_name(raw_name)
        guidance = get_guidance(name)

        return {
            "status": "detected",
            "message": "PlantDoc multi-crop classification completed.",
            "best_detection": {
                "name": name,
                "confidence": round(confidence, 2)
            },
            "detections": [{
                "name": name,
                "raw_name": str(raw_name),
                "confidence": round(confidence, 2),
                "box": None,
                "class_id": class_id
            }],
            "total_detections": 1,
            "guidance": guidance,
            "model_type": "PlantDoc multi-crop classification"
        }

    except Exception as e:
        print("Prediction error:", repr(e))
        return {"error": "Prediction failed: " + str(e)}

