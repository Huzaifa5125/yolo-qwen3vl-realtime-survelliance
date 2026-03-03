// ─── Polling for latest analysis ───
let analysisInterval = null;
let currentPage = 1;
const PAGE_SIZE = 12;

function startPolling() {
    // Poll latest analysis every 2 seconds
    analysisInterval = setInterval(fetchLatestAnalysis, 2000);
    // Poll stats every 5 seconds
    setInterval(fetchStats, 5000);
    // Initial load
    fetchLatestAnalysis();
    fetchStats();
    loadAnalyses();
}

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
        }
    } catch (e) {
        console.error("Failed to fetch analysis:", e);
    }
}

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

// ─── Saved Analyses ───
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

        return `
            <div class="saved-card" onclick="openModal('${encodeURIComponent(a.frame_path)}', \`${escapeHtml(a.analysis_text)}\`)">
                <img src="/api/frame/${encodeURIComponent(a.frame_path)}" alt="Frame ${a.frame_id}" loading="lazy">
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

// ─── Modal ───
function openModal(framePath, text) {
    const modal = document.getElementById("modal");
    document.getElementById("modal-img").src = `/api/frame/${framePath}`;
    document.getElementById("modal-text").textContent = decodeURIComponent(text);
    modal.classList.add("active");
}

function closeModal() {
    document.getElementById("modal").classList.remove("active");
}

// ─── Utils ───
function escapeHtml(text) {
    if (!text) return "";
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

// ─── Init ───
document.addEventListener("DOMContentLoaded", startPolling);