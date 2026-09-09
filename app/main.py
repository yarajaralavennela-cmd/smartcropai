from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from ultralytics import YOLO
from PIL import Image
import io
import os
import folium

app = FastAPI(title="SmartCrop AI")

# ============================================================
# MODEL CONFIGURATION
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "best.pt")
MODEL_IMAGE_SIZE = 224

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"YOLO model not found!\nExpected model at:\n{MODEL_PATH}"
    )

print("=" * 60)
print("SMARTCROP AI")
print("=" * 60)
print("Loading PlantDoc model...")
print(f"Model: {MODEL_PATH}")

model = YOLO(MODEL_PATH)

print("Model loaded successfully.")
print("Classes:", model.names)
print("=" * 60)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_class_name(name):
    name = str(name)
    name = name.replace("_", " ")
    name = name.replace("-", " ")
    return " ".join(name.split()).title()


def is_healthy(name):
    text = str(name).lower()
    return any(word in text for word in [
        "healthy",
        "normal",
        "good",
        "fresh"
    ])


def get_guidance(name):
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

    if "bacterial spot" in text:
        return (
            "Possible bacterial spot detected. "
            "Remove severely affected leaves and avoid unnecessary "
            "leaf wetness. Monitor nearby plants."
        )

    if "mosaic virus" in text or "yellow virus" in text:
        return (
            "Possible viral infection detected. "
            "Monitor nearby plants and consult a local agriculture "
            "expert before applying treatment."
        )

    if "spider mite" in text:
        return (
            "Possible spider mite infestation detected. "
            "Inspect the underside of leaves and monitor nearby plants."
        )

    if "rust" in text:
        return (
            "Possible rust infection detected. "
            "Remove heavily affected leaves and improve air circulation."
        )

    if "black rot" in text:
        return (
            "Possible black rot detected. "
            "Remove affected plant material and avoid excess moisture."
        )

    if is_healthy(name):
        return (
            "The plant appears healthy. "
            "Continue regular watering, nutrition and crop monitoring."
        )

    return (
        "A possible crop problem was detected. "
        "Inspect nearby plants and consult a local agriculture expert "
        "before applying treatment."
    )


# ============================================================
# WEATHER-BASED RISK FORECASTING
# Prototype heuristic for SIH demonstration
# ============================================================

def calculate_weather_risk(
    disease_name,
    temperature,
    humidity,
    rainfall
):
    disease = str(disease_name).lower()

    if is_healthy(disease):
        return {
            "risk_level": "LOW",
            "risk_score": 10,
            "risk_icon": "🟢",
            "reason": "The AI detected a healthy crop pattern.",
            "recommendation": (
                "Continue regular crop monitoring and maintain "
                "good field hygiene."
            )
        }

    risk_score = 30

    if humidity >= 85:
        risk_score += 35
    elif humidity >= 75:
        risk_score += 25
    elif humidity >= 65:
        risk_score += 15

    if rainfall:
        risk_score += 20

    if 18 <= temperature <= 28:
        risk_score += 20
    elif 15 <= temperature <= 32:
        risk_score += 10

    if "blight" in disease:
        risk_score += 5
    elif "mold" in disease:
        risk_score += 5
    elif "spot" in disease:
        risk_score += 5

    risk_score = min(risk_score, 100)

    if risk_score >= 70:
        risk_level = "HIGH"
        risk_icon = "🔴"
    elif risk_score >= 40:
        risk_level = "MEDIUM"
        risk_icon = "🟠"
    else:
        risk_level = "LOW"
        risk_icon = "🟢"

    if "blight" in disease:
        reason = (
            "Warm, humid and wet conditions can favor "
            "blight development."
        )
        recommendation = (
            "Monitor affected plants closely, improve air circulation, "
            "and consult an agriculture expert before treatment."
        )
    elif "mold" in disease:
        reason = (
            "High humidity and moisture may increase "
            "leaf mold risk."
        )
        recommendation = (
            "Improve ventilation and avoid unnecessary leaf wetness."
        )
    elif "spot" in disease:
        reason = (
            "Moist conditions may increase the risk "
            "of leaf spot development."
        )
        recommendation = (
            "Monitor nearby plants and reduce prolonged leaf wetness."
        )
    else:
        reason = (
            "Current weather conditions may influence "
            "crop disease development."
        )
        recommendation = (
            "Continue monitoring the crop and consult a local "
            "agriculture expert if symptoms increase."
        )

    return {
        "risk_level": risk_level,
        "risk_score": risk_score,
        "risk_icon": risk_icon,
        "reason": reason,
        "recommendation": recommendation
    }


# ============================================================
# DISEASE HOTSPOT DATA
# ============================================================

# Temporary in-memory storage for prototype hotspots.
# Data is cleared when the FastAPI server restarts.
hotspot_data = []


# ============================================================
# DISEASE HOTSPOT MAP
# Demo/sample location only
# ============================================================

def create_hotspot_map():
    m = folium.Map(
        location=[16.5062, 80.6480],
        zoom_start=11
    )

    folium.Marker(
        [16.5062, 80.6480],
        popup="""
        <b>🌿 Crop:</b> Tomato<br>
        <b>🦠 Disease:</b> Tomato Septoria Leaf Spot<br>
        <b>⚠️ Risk:</b> Medium<br>
        <b>📍 Location:</b> Vijayawada<br><br>
        <b>Note:</b> Demo/sample hotspot for prototype presentation.
        """,
        tooltip="🦠 Disease Detected"
    ).add_to(m)

    return m._repr_html_()


@app.post("/add-hotspot")
async def add_hotspot(
    latitude: float = Form(...),
    longitude: float = Form(...),
    disease: str = Form(...),
    confidence: float = Form(...)
):
    try:
        hotspot = {
            "latitude": latitude,
            "longitude": longitude,
            "disease": disease,
            "confidence": round(float(confidence), 2)
        }

        hotspot_data.append(hotspot)

        return {
            "success": True,
            "message": "Disease hotspot saved successfully",
            "hotspot": hotspot
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@app.get("/hotspots")
async def get_hotspots():
    return {
        "success": True,
        "total": len(hotspot_data),
        "hotspots": hotspot_data
    }


@app.get("/hotspot-map", response_class=HTMLResponse)
async def hotspot_map():
    return create_hotspot_map()


# ============================================================
# HOME PAGE
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
<!DOCTYPE html>
<html lang="en">

<head>
<meta charset="UTF-8">
<title>SmartCrop AI</title>

<meta
    name="viewport"
    content="width=device-width, initial-scale=1.0"
>

<style>

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    font-family: Arial, Helvetica, sans-serif;
    background:
        linear-gradient(
            rgba(238,247,237,0.92),
            rgba(238,247,237,0.92)
        );
    color: #243b25;
}

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
    box-shadow: 0 3px 12px rgba(0,0,0,0.15);
}

.logo {
    font-size: 25px;
    font-weight: bold;
}

.language button {
    background: rgba(255,255,255,0.18);
    padding: 8px 14px;
    border-radius: 20px;
    border: none;
    color: white;
    cursor: pointer;
    font-size: 15px;
}

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
    box-shadow: 0 5px 20px rgba(0,0,0,0.08);
}

.card h2 {
    color: #2e7d32;
    margin-top: 0;
}

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

.upload-button,
.expert-button {
    padding: 13px 20px;
    border: none;
    border-radius: 10px;
    background: #66bb6a;
    color: white;
    font-size: 16px;
    cursor: pointer;
}

.expert-button {
    background: #2e7d32;
}

.weather-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 15px;
    margin-top: 15px;
}

.weather-input {
    background: #f8fff7;
    border-radius: 14px;
    padding: 15px;
    border: 1px solid #c8e6c9;
}

.weather-input label {
    display: block;
    font-weight: bold;
    color: #2e7d32;
    margin-bottom: 8px;
}

.weather-input input,
.weather-input select {
    width: 100%;
    padding: 12px;
    border-radius: 10px;
    border: 1px solid #a5d6a7;
    font-size: 16px;
}

.weather-button,
.map-button {
    width: 100%;
    padding: 15px;
    border: none;
    border-radius: 12px;
    background: #1565c0;
    color: white;
    font-size: 17px;
    font-weight: bold;
    cursor: pointer;
    margin-top: 15px;
}

.map-button {
    background: #2e7d32;
}

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

.risk-card {
    margin-top: 20px;
    padding: 22px;
    border-radius: 18px;
    background:
        linear-gradient(
            135deg,
            #e3f2fd,
            #f1f8e9
        );
    border: 2px solid #90caf9;
}

.risk-title {
    font-size: 24px;
    font-weight: bold;
    margin-bottom: 10px;
}

.risk-high {
    color: #c62828;
}

.risk-medium {
    color: #ef6c00;
}

.risk-low {
    color: #2e7d32;
}

.weather-data {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    margin-top: 15px;
}

.weather-data-box {
    background: white;
    padding: 12px;
    border-radius: 12px;
    text-align: center;
    font-weight: bold;
}

.result-image {
    width: 100%;
    max-width: 700px;
    border-radius: 15px;
    margin-top: 15px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.15);
}

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

    .features,
    .weather-grid,
    .weather-data {
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

<button
    onclick="toggleLanguage()"
    id="languageBtn"
>
    🌐 తెలుగు
</button>

</div>

</header>


<div class="hero">

<h1>
    🌱 Protect Your Crop
</h1>

<p>
    AI-powered crop disease and pest detection
</p>

</div>


<div class="container">


<!-- =====================================================
     LIVE CAMERA
===================================================== -->

<div class="card">

<h2>
    📷 Live Crop Scanner
</h2>

<p>
    Point your camera towards a crop plant or leaf.
    SmartCrop AI will continuously analyze the image
    and identify possible crop problems.
</p>

<div class="camera-area">

<button
    class="scan-button"
    onclick="startCamera()"
>
    📷 Start Live Crop Scan
</button>

<div class="video-wrapper">

<video
    id="camera"
    autoplay
    playsinline
    style="display:none;"
>
</video>

<canvas
    id="overlay"
    style="display:none;"
>
</canvas>

</div>

<br>

<button
    class="stop-button"
    onclick="stopCamera()"
>
    🛑 Stop Camera
</button>

</div>

</div>


<!-- =====================================================
     IMAGE UPLOAD
===================================================== -->

<div class="card">

<h2>
    🖼️ Upload Crop Photo
</h2>

<div class="upload-box">

<p>
    Upload a clear crop plant or leaf image.
</p>

<input
    type="file"
    id="image"
    accept="image/*"
    onchange="previewImage()"
>

<div id="preview"></div>

<br>

<button
    class="upload-button"
    onclick="scanCrop()"
>
    🔍 Analyze Crop
</button>

</div>

</div>


<!-- =====================================================
     WEATHER RISK
===================================================== -->

<div class="card">

<h2>
    🌦️ Weather-Based Risk Forecast
</h2>

<p>
    Enter the current field weather conditions.
    SmartCrop AI combines the detected crop problem
    with weather conditions to estimate disease risk.
</p>

<div class="weather-grid">

<div class="weather-input">

<label>
    🌡️ Temperature (°C)
</label>

<input
    type="number"
    id="temperature"
    value="24"
    min="-10"
    max="60"
    step="0.1"
>

</div>


<div class="weather-input">

<label>
    💧 Humidity (%)
</label>

<input
    type="number"
    id="humidity"
    value="75"
    min="0"
    max="100"
    step="1"
>

</div>


<div class="weather-input">

<label>
    🌧️ Rainfall
</label>

<select id="rainfall">

<option value="no">
    No Rain
</option>

<option value="yes">
    Rainfall
</option>

</select>

</div>

</div>

<button
    class="weather-button"
    onclick="calculateRisk()"
>
    🌦️ Assess Disease Risk
</button>

</div>


<!-- =====================================================
     RESULT
===================================================== -->

<div class="card">

<h2>
    🌿 AI Detection Result
</h2>

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
     DISEASE HOTSPOT MAP
===================================================== -->

<div class="card">

<h2>
    🗺️ Disease Hotspot Map
</h2>

<p>
    View sample areas where crop diseases and
    pest problems can be mapped for monitoring.
</p>

<a
    href="/hotspot-map"
    target="_blank"
>
<button class="map-button">
    🗺️ View Disease Hotspots
</button>
</a>

<p class="box-info">
    Demo map for prototype presentation.
</p>

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
    🤖
</div>

<h3>
    Multi-Crop AI
</h3>

<p>
    Identify possible diseases and pest problems
    from crop images.
</p>

</div>


<div class="feature">

<div class="feature-icon">
    🌦️
</div>

<h3>
    Risk Forecast
</h3>

<p>
    Combine crop detection with weather conditions
    to estimate disease risk.
</p>

</div>

</div>


</div>


<footer>
    🌱 SmartCrop AI | Early Crop Disease Detection
</footer>


<script>

// ========================================================
// GLOBAL VARIABLES
// ========================================================

let cameraStream = null;
let scanning = false;
let liveRequestRunning = false;
let liveTimer = null;
let lastDetection = null;


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
            alt="Selected crop image"
        >
    `;
}


// ========================================================
// WEATHER VALUES
// ========================================================

function getWeatherValues() {

    const temperature =
        parseFloat(
            document.getElementById("temperature").value
        );

    const humidity =
        parseFloat(
            document.getElementById("humidity").value
        );

    const rainfall =
        document.getElementById("rainfall").value;

    if (isNaN(temperature)) {
        alert("Please enter temperature.");
        return null;
    }

    if (
        isNaN(humidity) ||
        humidity < 0 ||
        humidity > 100
    ) {
        alert(
            "Please enter valid humidity between 0 and 100."
        );
        return null;
    }

    return {
        temperature: temperature,
        humidity: humidity,
        rainfall: rainfall
    };
}


// ========================================================
// SAVE DISEASE HOTSPOT USING BROWSER GPS
// ========================================================

function saveDiseaseHotspot(disease, confidence) {

    if (!navigator.geolocation) {
        console.log("Geolocation is not supported by this browser.");
        return;
    }

    navigator.geolocation.getCurrentPosition(
        async function(position) {

            const latitude = position.coords.latitude;
            const longitude = position.coords.longitude;

            const formData = new FormData();

            formData.append("latitude", latitude);
            formData.append("longitude", longitude);
            formData.append("disease", disease);
            formData.append("confidence", confidence);

            try {

                const response = await fetch(
                    "/add-hotspot",
                    {
                        method: "POST",
                        body: formData
                    }
                );

                const data = await response.json();

                if (data.success) {
                    console.log(
                        "Disease hotspot saved:",
                        data.hotspot
                    );
                } else {
                    console.error(
                        "Hotspot save failed:",
                        data.error
                    );
                }

            } catch (error) {

                console.error(
                    "Hotspot save error:",
                    error
                );
            }
        },

        function(error) {

            console.log(
                "Location permission not available:",
                error.message
            );
        },

        {
            enableHighAccuracy: true,
            timeout: 10000,
            maximumAge: 60000
        }
    );
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
            <p>Please wait...</p>
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

        lastDetection = data;
        showResult(data);

        // Save disease hotspot only for uploaded image analysis
        if (data.detection) {
            const detectedName = data.detection.class;
            const detectedConfidence =
                Number(data.detection.confidence);

            if (!detectedName.toLowerCase().includes("healthy")) {
                saveDiseaseHotspot(
                    detectedName,
                    detectedConfidence
                );
            }
        }

    } catch (error) {

        console.error("Analyze Error:", error);

        result.innerHTML = `
            <div class="result-card">
                ❌ Unable to analyze image.
                <p class="box-info">
                    Error: ${error.message}
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

    const detection =
        data.detection;

    if (!detection) {

        result.innerHTML = `
            <div class="result-card">
                <div style="font-size:50px;">
                    🔍
                </div>

                <h2>
                    No confident detection found
                </h2>

                <p>
                    Try a clearer image with the
                    crop leaf or plant clearly visible.
                </p>

                <div class="warning">
                    ⚠️ This does NOT necessarily mean
                    the plant is healthy.
                </div>
            </div>
        `;

        return;
    }

    const name =
        detection.class;

    const confidence =
        Number(
            detection.confidence
        );

    const healthy =
        name
            .toLowerCase()
            .includes("healthy");

    const icon =
        healthy ? "🟢" : "🔴";

    const title =
        healthy
        ? "Crop Looks Healthy"
        : "Possible Crop Problem Detected";

    const guidance =
        data.guidance ||
        "Inspect nearby plants and consult a local agriculture expert before applying treatment.";

    const confidenceWarning =
        confidence < 50
        ? `
            <div class="warning">
                ⚠️ Low confidence result.
                Please capture a clearer image
                and request expert validation.
            </div>
        `
        : "";

    result.innerHTML = `

        <div class="result-card">

            <div style="font-size:50px;">
                ${icon}
            </div>

            <h2>
                ${title}
            </h2>

            <div class="${
                healthy ? "healthy" : "detection"
            }">
                ${name}
            </div>

            <p class="confidence">
                🎯 Confidence:
                ${confidence.toFixed(2)}%
            </p>

            <div class="box-info">
                📍 Detected areas: 1
            </div>

            ${confidenceWarning}

            <div class="warning">
                <b>🌱 Farmer Guidance</b>

                <p>
                    ${guidance}
                </p>
            </div>

            <div style="
                margin-top:15px;
                text-align:center;
            ">

                <button
                    onclick="requestExpertValidation()"
                    class="expert-button"
                >
                    🔬 Request Expert Validation
                </button>

                <div
                    id="expert-status"
                    style="
                        margin-top:10px;
                        font-weight:bold;
                    "
                >
                </div>

            </div>

        </div>
    `;
}


// ========================================================
// EXPERT VALIDATION
// ========================================================

function requestExpertValidation() {

    const status =
        document.getElementById(
            "expert-status"
        );

    if (!status) {
        return;
    }

    status.innerHTML =
        "✅ Expert Validation Requested. Your crop result has been sent for expert review.";

    status.style.color =
        "#2e7d32";
}


// ========================================================
// WEATHER RISK
// ========================================================

async function calculateRisk() {

    const weather =
        getWeatherValues();

    if (!weather) {
        return;
    }

    const result =
        document.getElementById("result");

    if (!lastDetection ||
        !lastDetection.detection) {

        result.innerHTML = `
            <div class="warning">
                ⚠️ Please analyze a crop image first.
                <br><br>
                Then enter weather conditions
                and assess disease risk.
            </div>
        `;

        return;
    }

    const disease =
        lastDetection.detection.class;

    const confidence =
        lastDetection.detection.confidence;

    const formData =
        new FormData();

    formData.append(
        "disease_name",
        disease
    );

    formData.append(
        "temperature",
        weather.temperature
    );

    formData.append(
        "humidity",
        weather.humidity
    );

    formData.append(
        "rainfall",
        weather.rainfall
    );

    try {

        const response =
            await fetch(
                "/weather-risk",
                {
                    method: "POST",
                    body: formData
                }
            );

        if (!response.ok) {
            throw new Error(
                "Weather server error: " +
                response.status
            );
        }

        const data =
            await response.json();

        if (data.error) {
            alert(data.error);
            return;
        }

        const risk =
            data.risk;

        let riskClass =
            "risk-low";

        if (risk.risk_level === "HIGH") {
            riskClass =
                "risk-high";
        } else if (
            risk.risk_level === "MEDIUM"
        ) {
            riskClass =
                "risk-medium";
        }

        result.innerHTML = `

            <div class="result-card">

                <div style="font-size:45px;">
                    🌦️
                </div>

                <h2>
                    Weather-Based Disease Risk
                </h2>

                <div class="summary-box">
                    🌿 ${disease}
                    <br>
                    🎯 AI Confidence:
                    ${Number(confidence).toFixed(2)}%
                </div>

                <div class="risk-card">

                    <div
                        class="risk-title ${riskClass}"
                    >
                        ${risk.risk_icon}
                        Disease Risk:
                        ${risk.risk_level}
                    </div>

                    <h3>
                        Risk Score:
                        ${risk.risk_score}/100
                    </h3>

                    <div class="weather-data">

                        <div class="weather-data-box">
                            🌡️
                            ${data.temperature}°C
                            <br>
                            <small>
                                Temperature
                            </small>
                        </div>

                        <div class="weather-data-box">
                            💧
                            ${data.humidity}%
                            <br>
                            <small>
                                Humidity
                            </small>
                        </div>

                        <div class="weather-data-box">
                            🌧️
                            ${data.rainfall
                                ? "Yes"
                                : "No"}
                            <br>
                            <small>
                                Rainfall
                            </small>
                        </div>

                    </div>

                    <div class="warning">

                        <b>
                            ⚠️ Risk Assessment
                        </b>

                        <p>
                            ${risk.reason}
                        </p>

                    </div>

                    <div class="success">

                        <b>
                            🌱 Recommended Action
                        </b>

                        <p>
                            ${risk.recommendation}
                        </p>

                    </div>

                </div>

            </div>
        `;

    } catch (error) {

        console.error(error);

        alert(
            "Unable to calculate weather risk."
        );
    }
}


// ========================================================
// DRAW CAMERA BOXES
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
            ctx.strokeStyle = "#ff2222";

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
                        facingMode: {
                            ideal: "environment"
                        },
                        width: {
                            ideal: 1280
                        },
                        height: {
                            ideal: 720
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
                    Show a crop leaf clearly
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
            liveRequestRunning = false;
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

        if (!data.error) {

            lastDetection =
                data;

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


// ========================================================
// LANGUAGE SWITCH
// ========================================================

let telugu = false;

function toggleLanguage() {

    telugu = !telugu;

    const btn =
        document.getElementById(
            "languageBtn"
        );

    if (telugu) {

        document.querySelector(
            ".hero h1"
        ).innerText =
            "🌱 మీ పంటను రక్షించండి";

        document.querySelector(
            ".hero p"
        ).innerText =
            "AI ఆధారిత పంట వ్యాధులు మరియు పురుగుల గుర్తింపు";

        btn.innerText =
            "🌐 English";

    } else {

        document.querySelector(
            ".hero h1"
        ).innerText =
            "🌱 Protect Your Crop";

        document.querySelector(
            ".hero p"
        ).innerText =
            "AI-powered crop disease and pest detection";

        btn.innerText =
            "🌐 తెలుగు";
    }
}

</script>

</body>
</html>
"""


# ============================================================
# MULTI-CROP CLASSIFICATION API
# ============================================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:

        if not file:
            return {
                "success": False,
                "error": "No image was received."
            }

        contents = await file.read()

        if not contents:
            return {
                "success": False,
                "error": "The selected image is empty."
            }

        try:
            image = Image.open(
                io.BytesIO(contents)
            ).convert("RGB")
        except Exception as e:
            return {
                "success": False,
                "error": f"Invalid image: {str(e)}"
            }

        results = model.predict(
            source=image,
            imgsz=MODEL_IMAGE_SIZE,
            verbose=False
        )

        result = results[0]

        class_id = int(
            result.probs.top1
        )

        confidence = float(
            result.probs.top1conf
        ) * 100

        raw_name = result.names[class_id]

        name = clean_class_name(raw_name)

        guidance = get_guidance(name)

        return {
            "success": True,
            "detection": {
                "class": name,
                "confidence": round(
                    confidence,
                    2
                ),
                "box": None
            },
            "guidance": guidance,
            "total_detections": 1,
            "model_type":
                "Multi-Crop Classification"
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }


# ============================================================
# WEATHER RISK API
# ============================================================

@app.post("/weather-risk")
async def weather_risk(
    disease_name: str = Form(...),
    temperature: float = Form(...),
    humidity: float = Form(...),
    rainfall: str = Form(...)
):

    try:

        rainfall_bool = rainfall.lower() == "yes"

        risk = calculate_weather_risk(
            disease_name,
            temperature,
            humidity,
            rainfall_bool
        )

        return {
            "success": True,
            "disease_name": disease_name,
            "temperature": temperature,
            "humidity": humidity,
            "rainfall": rainfall_bool,
            "risk": risk
        }

    except Exception as e:

        print(
            "Weather risk error:",
            repr(e)
        )

        return {
            "success": False,
            "error":
                "Weather risk calculation failed: "
                + str(e)
        }
