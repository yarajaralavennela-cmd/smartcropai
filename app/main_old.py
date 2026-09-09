from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from ultralytics import YOLO
from PIL import Image
import io
import os

app = FastAPI()

# =========================================================
# SMARTCROP AI - OBJECT DETECTION MODEL
# =========================================================

MODEL_PATH = r"C:\Users\unuku\OneDrive\Documents\smartcropai\dataset\Tomato-Village-main\Tomato-Village-main\Variant-c(Object Detection)\runs\detect\train-3\weights\best.pt"

model = YOLO(MODEL_PATH)

# =========================================================
# FARMER FRIENDLY FRONTEND
# =========================================================

@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html>

<head>

<title>SmartCrop AI</title>

<meta name="viewport" content="width=device-width, initial-scale=1.0">

<style>

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    font-family: Arial, sans-serif;
    background: #eef7ed;
    color: #243b25;
}

/* HEADER */

header {
    background: linear-gradient(135deg, #1b5e20, #43a047);
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
    background: rgba(255,255,255,0.18);
    padding: 8px 14px;
    border-radius: 20px;
}

/* HERO */

.hero {
    text-align: center;
    padding: 35px 15px 20px;
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

/* CONTAINER */

.container {
    max-width: 900px;
    margin: auto;
    padding: 15px;
}

.card {
    background: white;
    border-radius: 22px;
    padding: 25px;
    margin-bottom: 20px;
    box-shadow: 0 5px 20px rgba(0,0,0,0.08);
}

.card h2 {
    color: #2e7d32;
    margin-top: 0;
}

/* BUTTON */

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

/* CAMERA */

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

/* CAMERA CANVAS */

#overlay {
    position: absolute;
    left: 0;
    top: 15px;
    width: 100%;
    height: calc(100% - 15px);
    pointer-events: none;
}

/* UPLOAD */

.upload-box {
    border: 2px dashed #81c784;
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

/* RESULT */

#result {
    margin-top: 20px;
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
    color: #2e7d32;
}

.confidence {
    font-size: 20px;
    font-weight: bold;
}

/* FEATURES */

.features {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 15px;
}

.feature {
    background: white;
    padding: 20px;
    border-radius: 18px;
    text-align: center;
    box-shadow: 0 3px 12px rgba(0,0,0,0.06);
}

.feature-icon {
    font-size: 35px;
}

.feature h3 {
    color: #2e7d32;
}

footer {
    text-align: center;
    padding: 25px;
    color: #668066;
}

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


<!-- LIVE CAMERA -->

<div class="card">

<h2>📷 Live Crop Scanner</h2>

<p>
Point your camera towards a tomato plant or leaf.
AI will detect possible crop problems.
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

<canvas id="overlay" style="display:none;"></canvas>

</div>

<br>

<button
class="stop-button"
onclick="stopCamera()">

🛑 Stop Camera

</button>

</div>

</div>


<!-- IMAGE UPLOAD -->

<div class="card">

<h2>🖼️ Upload Crop Photo</h2>

<div class="upload-box">

<p>
Upload a clear tomato plant/leaf image
</p>

<input
type="file"
id="image"
accept="image/*">

<br>

<button
class="upload-button"
onclick="scanCrop()">

🔍 Analyze Crop

</button>

</div>

</div>


<!-- RESULT -->

<div class="card">

<h2>🌿 AI Detection Result</h2>

<div id="result">

<p>
Your crop analysis will appear here.
</p>

</div>

</div>


<!-- FEATURES -->

<div class="features">

<div class="feature">

<div class="feature-icon">📷</div>

<h3>Live Scan</h3>

<p>
Scan crops directly using your camera.
</p>

</div>


<div class="feature">

<div class="feature-icon">📦</div>

<h3>Object Detection</h3>

<p>
AI identifies affected areas using bounding boxes.
</p>

</div>


<div class="feature">

<div class="feature-icon">🌾</div>

<h3>Farmer Friendly</h3>

<p>
Simple and easy-to-understand results.
</p>

</div>

</div>

</div>


<footer>

🌱 SmartCrop AI | Early Crop Disease Detection

</footer>


<script>

let cameraStream = null;
let scanning = false;


/* =====================================================
   IMAGE UPLOAD
===================================================== */

async function scanCrop() {

    const fileInput =
        document.getElementById("image");

    const result =
        document.getElementById("result");


    if (!fileInput.files.length) {

        alert("Please select a crop image first!");

        return;
    }


    result.innerHTML = `
        <div class="result-card">
            ⏳ AI is analyzing your crop...
        </div>
    `;


    const formData = new FormData();

    formData.append(
        "file",
        fileInput.files[0]
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


        const data =
            await response.json();


        if (data.error) {

            result.innerHTML =
                "❌ " + data.error;

            return;
        }


        showResult(data);


    } catch (error) {

        result.innerHTML = `
            <div class="result-card">
                ❌ Unable to analyze image.
            </div>
        `;

    }

}


/* =====================================================
   SHOW RESULT
===================================================== */

function showResult(data) {

    const result =
        document.getElementById("result");


    if (!data.detections || data.detections.length === 0) {

        result.innerHTML = `
            <div class="result-card">

                🟡 No confident detection found.

                <p>
                Try a clearer image with the leaf or plant
                clearly visible.
                </p>

            </div>
        `;

        return;
    }


    let html = `
        <div class="result-card">

        <div style="font-size:40px;">
        🤖
        </div>

        <h2>
        Possible Crop Problems
        </h2>
    `;


    data.detections.forEach(function(item, index) {

        const healthy =
            item.name.toLowerCase().includes("healthy");


        html += `

        <div style="
            background:white;
            padding:15px;
            border-radius:12px;
            margin-top:10px;
        ">

            <div class="${healthy ? 'healthy' : 'detection'}">

                ${healthy ? '🟢' : '🔴'}
                ${item.name}

            </div>

            <div class="confidence">

                🎯 Confidence:
                ${item.confidence}%

            </div>

        </div>

        `;

    });


    html += `

        <div class="advice">

            <b>🌱 Farmer Guidance</b>

            <p>
            A possible crop problem was detected.
            Check nearby plants and consult a local
            agriculture expert before applying treatment.
            </p>

        </div>

        </div>

    `;


    result.innerHTML = html;c

}


/* =====================================================
   START CAMERA
===================================================== */

async function startCamera() {

    const video =
        document.getElementById("camera");

    const overlay =
        document.getElementById("overlay");

    const result =
        document.getElementById("result");


    try {

        cameraStream =
            await navigator.mediaDevices
            .getUserMedia({
                video: true
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
                📷 Camera started.<br>
                🌿 Show a tomato plant or leaf...
            </div>
        `;


        setTimeout(
            scanLiveFrame,
            2000
        );


    } catch (error) {

        alert(
            "Please allow camera permission."
        );

    }

}


/* =====================================================
   LIVE CAMERA SCAN
===================================================== */

async function scanLiveFrame() {

    if (!scanning) {
        return;
    }


    const video =
        document.getElementById("camera");

    const result =
        document.getElementById("result");


    if (!video.videoWidth) {

        setTimeout(
            scanLiveFrame,
            1000
        );

        return;
    }


    const canvas =
        document.createElement("canvas");


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


    canvas.toBlob(
        async function(blob) {

            if (!blob) {
                return;
            }


            const formData =
                new FormData();


            formData.append(
                "file",
                blob,
                "live_camera.jpg"
            );


            result.innerHTML = `
                <div class="result-card">
                    🤖 AI is scanning...
                </div>
            `;


            try {

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

                    result.innerHTML =
                        "❌ " + data.error;

                } else {

                    showResult(data);

                }


            } catch (error) {

                result.innerHTML = `
                    <div class="result-card">
                        ❌ Live scan failed.
                    </div>
                `;

            }


            if (scanning) {

                setTimeout(
                    scanLiveFrame,
                    2000
                );

            }

        },

        "image/jpeg"

    );

}


/* =====================================================
   STOP CAMERA
===================================================== */

function stopCamera() {

    scanning = false;


    if (cameraStream) {

        cameraStream
            .getTracks()
            .forEach(
                track => track.stop()
            );

        cameraStream = null;

    }


    const video =
        document.getElementById("camera");

    const overlay =
        document.getElementById("overlay");


    video.srcObject = null;

    video.style.display =
        "none";

    overlay.style.display =
        "none";


    document.getElementById(
        "result"
    ).innerHTML = `

        <div class="result-card">

            🛑 Camera stopped.

        </div>

    `;

}

</script>

</body>

</html>
"""


# =========================================================
# OBJECT DETECTION API
# =========================================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:

        contents = await file.read()

        image = Image.open(
            io.BytesIO(contents)
        ).convert("RGB")


        # Run object detection
        results = model.predict(
            image,
            imgsz=416,
            conf=0.25,
            verbose=False
        )


        result = results[0]


        detections = []


        if result.boxes is not None:

            for box in result.boxes:

                class_id = int(
                    box.cls[0]
                )

                confidence = float(
                    box.conf[0]
                )


                name = result.names[
                    class_id
                ]


                detections.append({

                    "name": name,

                    "confidence": round(
                        confidence * 100,
                        2
                    ),

                    "box": [
                        round(float(x), 2)
                        for x in box.xyxy[0].tolist()
                    ]

                })


        return {

            "detections":
                detections

        }


    except Exception as e:

        return {

            "error":
                str(e)

        }
