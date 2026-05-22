(function () {
    "use strict";

    /* ── Element refs ───────────────────────────────────────── */
    var dropArea   = document.getElementById("drop-area");
    var fileInput  = document.getElementById("fileInput");
    var urlInput   = document.getElementById("urlInput");
    var predictBtn = document.getElementById("predictBtn");
    var result     = document.getElementById("result");

    /* ── State ──────────────────────────────────────────────── */
    var selectedFile = null;

    /* ── Helpers ────────────────────────────────────────────── */
    function checkReady() {
        var hasFile = selectedFile !== null;
        var hasUrl  = urlInput.value.trim().length > 0;
        predictBtn.disabled = !(hasFile || hasUrl);
    }

    function setFile(file) {
        if (!file || !file.type.startsWith("image/")) return;
        selectedFile = file;
        urlInput.value = "";
        dropArea.querySelector(".drop-label").textContent = file.name;
        dropArea.querySelector(".drop-hint").textContent  = (file.size / 1024).toFixed(1) + " KB — ready";
        dropArea.classList.add("has-file");
        checkReady();
    }

    function showResult(msg) {
        result.innerHTML = msg;
        result.classList.add("visible");
    }

    /* ── Click drop area → open file picker ─────────────────── */
    dropArea.addEventListener("click", function () {
        fileInput.value = "";
        fileInput.click();
    });

    /* ── File chosen via picker ─────────────────────────────── */
    fileInput.addEventListener("change", function () {
        if (fileInput.files && fileInput.files.length > 0) {
            setFile(fileInput.files[0]);
        }
    });

    /* ── Drag events ─────────────────────────────────────────── */
    dropArea.addEventListener("dragenter", function (e) {
        e.preventDefault();
        e.stopPropagation();
        dropArea.classList.add("dragover");
    });

    dropArea.addEventListener("dragover", function (e) {
        e.preventDefault();
        e.stopPropagation();
        dropArea.classList.add("dragover");
    });

    dropArea.addEventListener("dragleave", function (e) {
        e.preventDefault();
        e.stopPropagation();
        dropArea.classList.remove("dragover");
    });

    dropArea.addEventListener("drop", function (e) {
        e.preventDefault();
        e.stopPropagation();
        dropArea.classList.remove("dragover");
        var files = e.dataTransfer.files;
        if (files && files.length > 0) {
            setFile(files[0]);
        }
    });

    /* ── URL input ───────────────────────────────────────────── */
    urlInput.addEventListener("input", function () {
        if (urlInput.value.trim().length > 0) {
            selectedFile = null;
            dropArea.querySelector(".drop-label").textContent = "Drag & drop an image here";
            dropArea.querySelector(".drop-hint").textContent  = "or click to browse files";
            dropArea.classList.remove("has-file");
        }
        checkReady();
    });

    /* ── Predict button ──────────────────────────────────────── */
    predictBtn.addEventListener("click", function () {
        if (selectedFile) {
            runPredictFile(selectedFile);
        } else if (urlInput.value.trim().length > 0) {
            runPredictURL(urlInput.value.trim());
        }
    });

    /* ── Send file — key name matches Flask: request.files["image"] ── */
    function runPredictFile(file) {
        showResult("⏳ Predicting…");
        var formData = new FormData();
        formData.append("image", file);          // must be "image" to match Flask

        fetch("/predict", { method: "POST", body: formData })
            .then(function (r) { return r.json(); })
            .then(function (data) {
                showResult(formatResult(data));
            })
            .catch(function () {
                showResult("❌ Prediction failed. Check your server.");
            });
    }

    /* ── Send URL — as form data to match Flask: request.form["url"] ── */
    function runPredictURL(url) {
        showResult("⏳ Predicting…");
        var formData = new FormData();
        formData.append("url", url);             // must be "url" to match Flask

        fetch("/predict", { method: "POST", body: formData })
            .then(function (r) { return r.json(); })
            .then(function (data) {
                showResult(formatResult(data));
            })
            .catch(function () {
                showResult("❌ Prediction failed. Check your server.");
            });
    }

    /* ── Format the result from predict_image() ─────────────── */
    function formatResult(data) {
        if (data.error) {
            return "❌ Error: " + data.error;
        }
        // Adjust the key below to match what predict_image() actually returns
        var label = data.label || data.class || data.result || JSON.stringify(data);
        var conf  = data.confidence !== undefined
                        ? " (" + (data.confidence ).toFixed(1) + "%)"
                        : "";
        return "🎯 <strong>" + label + conf + "</strong>";
    }

})();