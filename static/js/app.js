/* ── DashSpeed — frontend logic ─────────────────────────────────────────── */

(function () {
  "use strict";

  // ── DOM refs ─────────────────────────────────────────────────────────────
  const fileInput       = document.getElementById("file-input");
  const fileNameLabel   = document.getElementById("file-name");
  const uploadBtn       = document.getElementById("upload-btn");
  const progressWrap    = document.getElementById("upload-progress");
  const progressBar     = progressWrap.querySelector(".bar");
  const videoFeed       = document.getElementById("video-feed");
  const placeholder     = document.getElementById("placeholder");
  const dot             = document.getElementById("status-dot");
  const statusText      = document.getElementById("status-text");
  const tableBody       = document.getElementById("speed-tbody");
  const readingCount    = document.getElementById("reading-count");
  const emptyState      = document.getElementById("empty-state");

  // ── state ─────────────────────────────────────────────────────────────────
  let pollingTimer  = null;
  let knownRows     = 0;
  const POLL_MS     = 1500;

  // ── file picker ───────────────────────────────────────────────────────────
  document.getElementById("pick-btn").addEventListener("click", () => {
    fileInput.click();
  });

  fileInput.addEventListener("change", () => {
    const f = fileInput.files[0];
    if (!f) return;
    fileNameLabel.textContent = f.name;
    uploadBtn.disabled = false;
  });

  // ── upload ────────────────────────────────────────────────────────────────
  uploadBtn.addEventListener("click", async () => {
    const f = fileInput.files[0];
    if (!f) return;

    uploadBtn.disabled = true;
    setStatus("uploading", "Uploading video…");
    progressWrap.style.display = "block";
    progressBar.style.width    = "0%";

    const form = new FormData();
    form.append("video", f);

    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/upload");

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) {
        progressBar.style.width = `${Math.round(e.loaded / e.total * 100)}%`;
      }
    };

    xhr.onload = () => {
      progressWrap.style.display = "none";

      if (xhr.status !== 200) {
        const msg = tryParseError(xhr.responseText);
        setStatus("error", `Upload failed: ${msg}`);
        uploadBtn.disabled = false;
        return;
      }

      const data = JSON.parse(xhr.responseText);
      setStatus("running", `Processing at ${data.fps?.toFixed(1)} fps…`);
      startStream();
      startPolling();
    };

    xhr.onerror = () => {
      progressWrap.style.display = "none";
      setStatus("error", "Network error during upload.");
      uploadBtn.disabled = false;
    };

    xhr.send(form);
  });

  // ── MJPEG stream ──────────────────────────────────────────────────────────
  function startStream() {
    placeholder.style.display = "none";
    videoFeed.style.display   = "block";
    // Cache-bust so the browser doesn't reuse a stale stream
    videoFeed.src = `/stream?t=${Date.now()}`;
  }

  // ── speed-table polling ───────────────────────────────────────────────────
  function startPolling() {
    knownRows = 0;
    clearTable();
    if (pollingTimer) clearInterval(pollingTimer);
    pollingTimer = setInterval(pollSpeedData, POLL_MS);
  }

  async function pollSpeedData() {
    try {
      const [tableRes, statusRes] = await Promise.all([
        fetch("/speed_data"),
        fetch("/status"),
      ]);
      const rows   = await tableRes.json();
      const status = await statusRes.json();

      renderTable(rows);
      readingCount.textContent = rows.length;

      if (!status.is_processing && rows.length > 0) {
        setStatus("done", `Done — ${rows.length} readings recorded`);
        clearInterval(pollingTimer);
        uploadBtn.disabled = false;
      } else if (!status.is_processing) {
        setStatus("idle", "Ready");
        clearInterval(pollingTimer);
        uploadBtn.disabled = false;
      }
    } catch (_) { /* silently ignore transient fetch errors */ }
  }

  // ── table render ──────────────────────────────────────────────────────────
  function renderTable(rows) {
    if (rows.length === 0) {
      emptyState.style.display = "block";
      return;
    }
    emptyState.style.display = "none";

    // Only append new rows
    for (let i = knownRows; i < rows.length; i++) {
      const r = rows[i];

      // Remove "latest" highlight from previous row
      const prev = tableBody.querySelector("tr.latest");
      if (prev) prev.classList.remove("latest");

      const tr = document.createElement("tr");
      if (i === rows.length - 1) tr.classList.add("latest");

      tr.innerHTML = `
        <td>${r.measurement}</td>
        <td>${r.time_start}s – ${r.time_end}s</td>
        <td class="${r.speed_kmh != null ? "speed-val" : "no-data"}">
          ${r.speed_kmh != null ? r.speed_kmh.toFixed(1) : "—"}
        </td>
        <td>${r.speed_ms != null ? r.speed_ms.toFixed(2) : "—"}</td>
        <td>${confidenceBar(r.confidence)}</td>
      `;
      tableBody.appendChild(tr);
    }

    knownRows = rows.length;

    // Auto-scroll to bottom
    const wrap = document.querySelector(".table-wrap");
    wrap.scrollTop = wrap.scrollHeight;
  }

  function confidenceBar(c) {
    const pct = Math.min(Math.round((c / 30) * 100), 100);
    return `<div class="conf-bar-bg"><div class="conf-bar" style="width:${pct}%"></div></div>`;
  }

  function clearTable() {
    tableBody.innerHTML = "";
    knownRows = 0;
    readingCount.textContent = "0";
    emptyState.style.display = "block";
  }

  // ── status helpers ────────────────────────────────────────────────────────
  function setStatus(state, msg) {
    dot.className = `dot ${state}`;
    statusText.textContent = msg;
  }

  function tryParseError(body) {
    try {
      return JSON.parse(body).error || body;
    } catch (_) {
      return body;
    }
  }
})();
