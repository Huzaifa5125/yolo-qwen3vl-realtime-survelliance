// ─── Polling for latest analysis ───
let analysisInterval = null;
let currentPage = 1;
const PAGE_SIZE = 12;

function startPolling() {
    analysisInterval = setInterval(fetchLatestAnalysis, 2000);
    setInterval(fetchStats, 5000);
    fetchLatestAnalysis();
    fetchStats();
    loadAnalyses();
    fetchPromptInfo();  // load current prompt state on init
}

// ═══════════════════════════════════════════
//  LATEST ANALYSIS
// ═══════════════════════════════════════════

async function fetchLatestAnalysis() {
    try {
        const res = await fetch("/api/latest_analysis");
        const data = await res.json();

        const output = document.getElementById("analysis-output");
        const classBadge = document.getElementById("analysis-classification");
        const frameId = document.getElementById("analysis-frame-id");
        const persons = document.getElementById("analysis-persons");

        if (data.status === "ok") {
            output.innerHTML = `<p>${escapeHtml(data.text)}</p>`;
            classBadge.textContent = data.classification.toUpperCase();
            classBadge.className = `badge ${data.classification}`;
            frameId.textContent = `Frame: ${data.frame_id}`;
            persons.textContent = `Persons: ${data.person_ids.join(", ")}`;

            // Show which prompt was used for this analysis
            const promptBadge = document.getElementById("prompt-badge");
            if (data.prompt_used) {
                promptBadge.textContent = data.prompt_used;
            }
        }
    } catch (e) {
        console.error("Failed to fetch analysis:", e);
    }
}

// ═══════════════════════════════════════════
//  STATS
// ═══════════════════════════════════════════

async function fetchStats() {
    try {
        const res = await fetch("/api/stats");
        const data = await res.json();
        document.getElementById("stat-total").textContent = `Total: ${data.total}`;
        document.getElementById("stat-suspicious").textContent = `Suspicious: ${data.suspicious}`;
        document.getElementById("stat-normal").textContent = `Normal: ${data.normal}`;
    } catch (e) {
        console.error("Failed to fetch stats:", e);
    }
}

// ═══════════════════════════════════════════
//  PROMPT MANAGEMENT
// ═══════════════════════════════════════════

async function fetchPromptInfo() {
    try {
        const res = await fetch("/api/prompt");
        const data = await res.json();
        highlightActivePreset(data.name);
        document.getElementById("prompt-badge").textContent = data.name;
        document.getElementById("prompt-preview-text").textContent = data.text;
    } catch (e) {
        console.error("Failed to fetch prompt info:", e);
    }
}

function highlightActivePreset(presetName) {
    // Remove active from all preset buttons
    document.querySelectorAll(".preset-btn").forEach(btn => {
        btn.classList.remove("active");
    });
    // Add active to matching button
    const match = document.querySelector(`.preset-btn[data-preset="${presetName}"]`);
    if (match) {
        match.classList.add("active");
    }
    // If it's a custom prompt, no button will be highlighted (which is correct)
}

async function setPreset(presetKey) {
    try {
        const res = await fetch("/api/prompt", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ preset: presetKey }),
        });
        const data = await res.json();

        if (data.status === "ok") {
            highlightActivePreset(data.prompt.name);
            document.getElementById("prompt-badge").textContent = data.prompt.name;
            document.getElementById("prompt-preview-text").textContent = data.prompt.text;
            // Clear custom input since we switched to a preset
            document.getElementById("custom-prompt-input").value = "";
            showToast(`Prompt switched to: ${presetKey}`);
        } else {
            showToast(`Error: ${data.error}`, true);
        }
    } catch (e) {
        console.error("Failed to set preset:", e);
        showToast("Failed to set preset", true);
    }
}

async function applyCustomPrompt() {
    const textarea = document.getElementById("custom-prompt-input");
    const text = textarea.value.trim();

    if (!text) {
        showToast("Please enter a custom prompt first", true);
        return;
    }

    // Wrap the user's instruction into a surveillance analysis format
    // so that the output format remains consistent (P<id>: ... + Status: ...)
    const fullPrompt =
        "You are a surveillance analyst. " +
        text + " " +
        "For each detected person, describe what you observe in one short line. " +
        "Classify the overall scene as SUSPICIOUS or NORMAL based on your findings. " +
        "Format:\n" +
        "P<id>: <observation>\n" +
        "...\n" +
        "Status: <SUSPICIOUS|NORMAL>";

    try {
        const res = await fetch("/api/prompt", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ custom: fullPrompt }),
        });
        const data = await res.json();

        if (data.status === "ok") {
            highlightActivePreset("custom");  // won't match any button → all deselected
            document.getElementById("prompt-badge").textContent = "custom";
            document.getElementById("prompt-preview-text").textContent = data.prompt.text;
            showToast("Custom prompt applied!");
        } else {
            showToast(`Error: ${data.error}`, true);
        }
    } catch (e) {
        console.error("Failed to apply custom prompt:", e);
        showToast("Failed to apply custom prompt", true);
    }
}

function togglePromptPreview() {
    const el = document.getElementById("prompt-preview");
    el.classList.toggle("hidden");
    // Refresh the preview text
    fetchPromptInfo();
}

// ═══════════════════════════════════════════
//  SHOW BOXES TOGGLE
// ═══════════════════════════════════════════

function isShowBoxes() {
    const cb = document.getElementById("toggle-boxes");
    return cb && cb.checked;
}

function getFrameUrl(analysis) {
    const showBoxes = isShowBoxes();
    if (showBoxes && analysis.id) {
        return `/api/frame_with_boxes/${analysis.id}?show_boxes=1`;
    }
    const framePath = analysis.clean_frame_path || analysis.frame_path;
    return `/api/frame/${encodeURIComponent(framePath)}`;
}

// ═══════════════════════════════════════════
//  SAVED ANALYSES
// ═══════════════════════════════════════════

async function loadAnalyses() {
    const classification = document.getElementById("filter-class").value;
    const startInput = document.getElementById("filter-start").value;
    const endInput = document.getElementById("filter-end").value;

    let url = `/api/analyses?limit=${PAGE_SIZE}&offset=${(currentPage - 1) * PAGE_SIZE}`;
    if (classification) url += `&classification=${classification}`;
    if (startInput) url += `&start_ts=${new Date(startInput).getTime() / 1000}`;
    if (endInput) url += `&end_ts=${new Date(endInput).getTime() / 1000}`;

    try {
        const res = await fetch(url);
        const data = await res.json();
        renderGrid(data.analyses);
        document.getElementById("page-info").textContent = `Page ${currentPage}`;
    } catch (e) {
        console.error("Failed to load analyses:", e);
    }
}

function renderGrid(analyses) {
    const grid = document.getElementById("saved-grid");
    if (!analyses || analyses.length === 0) {
        grid.innerHTML = '<p style="color:#484f58; padding:20px;">No results found.</p>';
        return;
    }

    grid.innerHTML = analyses.map(a => {
        const time = new Date(a.timestamp * 1000).toLocaleString();
        const preview = a.analysis_text
            ? a.analysis_text.substring(0, 80) + "..."
            : "No analysis";
        const classBadge = a.classification === "suspicious"
            ? '<span class="badge suspicious">SUSPICIOUS</span>'
            : '<span class="badge normal">NORMAL</span>';

        const imgUrl = getFrameUrl(a);

        return `
            <div class="saved-card" onclick="openModal(${a.id}, \`${escapeHtml(a.analysis_text)}\`)">
                <img src="${imgUrl}" alt="Frame ${a.frame_id}" loading="lazy">
                <div class="saved-card-info">
                    <div class="time">${time} ${classBadge}</div>
                    <div class="text-preview">${escapeHtml(preview)}</div>
                </div>
            </div>
        `;
    }).join("");
}

function prevPage() {
    if (currentPage > 1) {
        currentPage--;
        loadAnalyses();
    }
}

function nextPage() {
    currentPage++;
    loadAnalyses();
}

function resetFilters() {
    document.getElementById("filter-class").value = "";
    document.getElementById("filter-start").value = "";
    document.getElementById("filter-end").value = "";
    currentPage = 1;
    loadAnalyses();
}

// ═══════════════════════════════════════════
//  MODAL
// ═══════════════════════════════════════════

function openModal(analysisId, text) {
    const modal = document.getElementById("modal");
    const showBoxes = isShowBoxes();
    const imgUrl = showBoxes
        ? `/api/frame_with_boxes/${analysisId}?show_boxes=1`
        : `/api/frame_with_boxes/${analysisId}?show_boxes=0`;
    document.getElementById("modal-img").src = imgUrl;
    document.getElementById("modal-text").textContent = decodeURIComponent(text);
    modal.classList.add("active");
}

function closeModal() {
    document.getElementById("modal").classList.remove("active");
}

// ═══════════════════════════════════════════
//  TOAST NOTIFICATION
// ═══════════════════════════════════════════

function showToast(msg, isError = false) {
    // Remove existing toast if any
    const existing = document.getElementById("toast-notification");
    if (existing) existing.remove();

    const toast = document.createElement("div");
    toast.id = "toast-notification";
    toast.className = `toast ${isError ? "toast-error" : "toast-success"}`;
    toast.textContent = msg;
    document.body.appendChild(toast);

    // Trigger reflow then add visible class for animation
    requestAnimationFrame(() => {
        toast.classList.add("toast-visible");
    });

    setTimeout(() => {
        toast.classList.remove("toast-visible");
        setTimeout(() => toast.remove(), 300);
    }, 2500);
}

// ═══════════════════════════════════════════
//  UTILS
// ═══════════════════════════════════════════

function escapeHtml(text) {
    if (!text) return "";
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

// ─── Init ───
document.addEventListener("DOMContentLoaded", startPolling);