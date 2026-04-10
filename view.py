import os
import json
import sqlite3
import argparse
from flask import Flask, render_template_string, jsonify, send_from_directory

app = Flask(__name__)

# Global variable to define the target directory containing the database and frames.
DATA_DIR = '.'

def get_db_path():
    return os.path.join(DATA_DIR, 'output/surveillance.db')

def get_frames_dir():
    return os.path.join(DATA_DIR, 'output/frames')

# ─── API ENDPOINTS ───

@app.route('/api/data')
def get_data():
    db_path = get_db_path()
    if not os.path.exists(db_path):
        return jsonify([])
        
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT * FROM analyses ORDER BY timestamp DESC").fetchall()
    except sqlite3.OperationalError:
        return jsonify([]) # In case table doesn't exist yet
        
    data = []
    for r in rows:
        d = dict(r)
        # Extract just the filenames since absolute paths from the server won't work locally
        d['clean_filename'] = os.path.basename(d['clean_frame_path']) if d.get('clean_frame_path') else None
        d['annotated_filename'] = os.path.basename(d['frame_path']) if d.get('frame_path') else None
        
        try:
            d['detections'] = json.loads(d['detections_json']) if d.get('detections_json') else []
        except:
            d['detections'] = []
        data.append(d)
        
    conn.close()
    return jsonify(data)

@app.route('/api/delete/<int:record_id>', methods=['POST'])
def delete_record(record_id):
    conn = sqlite3.connect(get_db_path())
    row = conn.execute("SELECT frame_path, clean_frame_path FROM analyses WHERE id=?", (record_id,)).fetchone()
    
    if row:
        # Delete image files
        for path in row:
            if path:
                filename = os.path.basename(path)
                full_path = os.path.join(get_frames_dir(), filename)
                if os.path.exists(full_path):
                    os.remove(full_path)
                    
        # Delete from DB
        conn.execute("DELETE FROM analyses WHERE id=?", (record_id,))
        conn.commit()
        
    conn.close()
    return jsonify({"success": True})

@app.route('/api/delete_all', methods=['POST'])
def delete_all():
    # Clear the database
    conn = sqlite3.connect(get_db_path())
    conn.execute("DELETE FROM analyses")
    conn.commit()
    conn.close()
    
    # Delete all images in the frames directory
    frames_dir = get_frames_dir()
    if os.path.exists(frames_dir):
        for f in os.listdir(frames_dir):
            file_path = os.path.join(frames_dir, f)
            if os.path.isfile(file_path):
                os.remove(file_path)
                
    return jsonify({"success": True})

@app.route('/frames/<path:filename>')
def serve_frame(filename):
    return send_from_directory(get_frames_dir(), filename)

# ─── FRONTEND ───

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Surveillance Viewer</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .image-container { position: relative; display: inline-block; max-width: 100%; border: 1px solid #e5e7eb; border-radius: 0.5rem; overflow: hidden; }
        .image-container img { max-width: 100%; height: auto; display: block; }
        .bbox { position: absolute; border: 2px solid #ef4444; box-sizing: border-box; pointer-events: none; display: none; background: rgba(239, 68, 68, 0.15); }
        .bbox.show { display: block; }
        .bbox-label { position: absolute; background: #ef4444; color: white; font-size: 11px; padding: 2px 6px; top: -22px; left: -2px; white-space: nowrap; border-radius: 4px 4px 0 0; font-weight: bold; }
        .selected { background-color: #e0e7ff !important; border-left: 4px solid #4f46e5; }
    </style>
</head>
<body class="bg-gray-100 h-screen flex flex-col">
    <nav class="bg-gray-900 text-white px-6 py-3 shadow-md flex justify-between items-center shrink-0">
        <h1 class="text-xl font-bold flex items-center gap-2">
            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"></path></svg>
            Surveillance Log Viewer
        </h1>
    </nav>

    <div class="flex-1 flex overflow-hidden p-4 gap-4">
        <!-- Sidebar -->
        <div class="w-80 bg-white rounded-lg shadow-md flex flex-col overflow-hidden shrink-0">
            <div class="p-3 bg-gray-50 border-b flex justify-between items-center">
                <span class="font-semibold text-gray-700">Records</span>
                <button onclick="deleteAll()" class="bg-red-500 hover:bg-red-600 text-white text-xs px-3 py-1.5 rounded transition">Clear All</button>
            </div>
            <div id="sidebar" class="flex-1 overflow-y-auto divide-y">
                <div class="p-4 text-center text-gray-500">Loading...</div>
            </div>
        </div>

        <!-- Main Content -->
        <div id="main-view" class="flex-1 bg-white rounded-lg shadow-md p-6 overflow-y-auto flex flex-col">
            <div class="text-center text-gray-500 mt-20">Select a frame from the sidebar to view details</div>
        </div>
    </div>

    <script>
        let currentData = [];
        let currentIndex = -1; // Track which frame is active for keyboard navigation

        async function loadData() {
            const res = await fetch('/api/data');
            currentData = await res.json();
            renderSidebar();
            if (currentData.length > 0) {
                selectFrame(currentData[0]);
            } else {
                document.getElementById('main-view').innerHTML = '<div class="text-center text-gray-500 mt-20">No records found. DB might be empty.</div>';
            }
        }

        function renderSidebar() {
            const sidebar = document.getElementById('sidebar');
            sidebar.innerHTML = '';
            currentData.forEach(item => {
                const date = new Date(item.timestamp * 1000).toLocaleString();
                const div = document.createElement('div');
                div.className = `p-3 cursor-pointer hover:bg-gray-50 transition border-l-4 border-transparent`;
                div.id = `sidebar-item-${item.id}`;
                
                let badgeColor = item.classification === 'suspicious' ? 'bg-red-100 text-red-800' : 'bg-green-100 text-green-800';
                let badgeText = (item.classification || 'normal').toUpperCase();

                div.innerHTML = `
                    <div class="flex justify-between items-start mb-1">
                        <span class="font-semibold text-sm">Frame #${item.id}</span>
                        <span class="text-[10px] px-2 py-0.5 rounded font-bold ${badgeColor}">${badgeText}</span>
                    </div>
                    <div class="text-xs text-gray-500 mb-1">${date}</div>
                    <div class="text-xs text-gray-600 truncate">Persons: ${item.num_persons}</div>
                `;
                div.onclick = () => selectFrame(item);
                sidebar.appendChild(div);
            });
        }

        function selectFrame(item) {
            // Update the current index based on the selected item's id
            currentIndex = currentData.findIndex(d => d.id === item.id);

            // Update sidebar highlighting
            document.querySelectorAll('#sidebar > div').forEach(el => el.classList.remove('selected', 'bg-indigo-50'));
            const activeItem = document.getElementById(`sidebar-item-${item.id}`);
            if (activeItem) activeItem.classList.add('selected', 'bg-indigo-50');

            const mainView = document.getElementById('main-view');
            
            // Prefer clean_filename if available so we can draw our own boxes dynamically
            const filename = item.clean_filename || item.annotated_filename; 
            const date = new Date(item.timestamp * 1000).toLocaleString();
            const latencyText = (item.vlm_latency !== undefined && item.vlm_latency !== null && !Number.isNaN(parseFloat(item.vlm_latency)))
                ? `${parseFloat(item.vlm_latency).toFixed(2)} s`
                : 'N/A';
            
            if (!filename) {
                mainView.innerHTML = `<div class="text-center text-gray-500 mt-20">Image file missing for this record.</div>`;
                return;
            }

            mainView.innerHTML = `
                <div class="flex justify-between items-center mb-4 pb-4 border-b">
                    <div>
                        <h2 class="text-2xl font-bold">Analysis #${item.id}</h2>
                        <span class="text-sm text-gray-500">${date}</span>
                    </div>
                    <div class="flex gap-3">
                        <label class="flex items-center gap-2 cursor-pointer bg-gray-100 px-3 py-2 rounded border hover:bg-gray-200 transition">
                            <input type="checkbox" id="toggleBbox" onchange="toggleBoxes()" class="w-4 h-4 text-indigo-600" checked> 
                            <span class="text-sm font-medium">Show Bounding Boxes</span>
                        </label>
                        <button onclick="deleteItem(${item.id})" class="bg-red-50 px-3 py-2 rounded border border-red-200 text-red-600 hover:bg-red-500 hover:text-white transition text-sm font-medium">
                            Delete Record
                        </button>
                    </div>
                </div>

                <div class="flex flex-col lg:flex-row gap-6">
                    <!-- Image Area -->
                    <div class="flex-1">
                        <div class="image-container" id="image-container">
                            <img id="surv-img" src="/frames/${filename}" alt="Surveillance Frame">
                        </div>
                    </div>
                    
                    <!-- Details Area -->
                    <div class="w-full lg:w-80 space-y-4">
                        <div class="bg-gray-50 p-4 rounded-lg border">
                            <h3 class="text-sm font-bold text-gray-500 uppercase tracking-wider mb-2">Classification</h3>
                            <p class="text-lg font-semibold capitalize ${item.classification === 'suspicious' ? 'text-red-600' : 'text-green-600'}">
                                ${item.classification || 'Normal'}
                            </p>
                        </div>
                        <div class="bg-indigo-50 p-4 rounded-lg border border-indigo-100">
                            <h3 class="text-sm font-bold text-indigo-800 uppercase tracking-wider mb-2">VLM Analysis</h3>
                            <p class="text-sm text-indigo-900 leading-relaxed whitespace-pre-wrap">${item.analysis_text || 'No detailed analysis available.'}</p>
                        </div>
                        <div class="bg-yellow-50 p-4 rounded-lg border border-yellow-100">
                            <h3 class="text-sm font-bold text-yellow-800 uppercase tracking-wider mb-2">Analysis Time</h3>
                            <p class="text-sm text-yellow-900 leading-relaxed whitespace-pre-wrap">${latencyText}</p>
                        </div>
                    </div>
                </div>
            `;

            const img = document.getElementById('surv-img');
            img.onload = () => {
                const container = document.getElementById('image-container');
                const isChecked = document.getElementById('toggleBbox').checked;
                
                item.detections.forEach(det => {
                    const [x1, y1, x2, y2] = det.bbox;
                    const w = img.naturalWidth;
                    const h = img.naturalHeight;
                    
                    // Convert raw coordinates to percentages so they scale with the responsive image
                    const left = (x1 / w * 100);
                    const top = (y1 / h * 100);
                    const width = ((x2 - x1) / w * 100);
                    const height = ((y2 - y1) / h * 100);
                    
                    const box = document.createElement('div');
                    box.className = 'bbox ' + (isChecked ? 'show' : '');
                    box.style.left = left + '%';
                    box.style.top = top + '%';
                    box.style.width = width + '%';
                    box.style.height = height + '%';
                    
                    const label = document.createElement('div');
                    label.className = 'bbox-label';
                    const conf = det.confidence ? Math.round(det.confidence * 100) + '%' : '';
                    const tid = det.track_id !== undefined ? `ID:${det.track_id}` : '';
                    label.innerText = `Person ${tid} ${conf}`;
                    
                    box.appendChild(label);
                    container.appendChild(box);
                });
            };
        }

        function navigateFrame(offset) {
            if (currentData.length === 0) return;

            // If nothing is selected yet, start at the first item
            if (currentIndex === -1) {
                currentIndex = 0;
            } else {
                currentIndex = (currentIndex + offset + currentData.length) % currentData.length;
            }

            selectFrame(currentData[currentIndex]);
        }

        // Keyboard navigation with ArrowLeft / ArrowRight
        document.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowLeft') {
                e.preventDefault();
                navigateFrame(-1);
            } else if (e.key === 'ArrowRight') {
                e.preventDefault();
                navigateFrame(1);
            }
        });

        function toggleBoxes() {
            const isChecked = document.getElementById('toggleBbox').checked;
            document.querySelectorAll('.bbox').forEach(el => {
                if(isChecked) el.classList.add('show');
                else el.classList.remove('show');
            });
        }

        async function deleteItem(id) {
            if(!confirm('Are you sure you want to delete this frame and its analysis?')) return;
            await fetch(`/api/delete/${id}`, { method: 'POST' });
            loadData();
        }

        async function deleteAll() {
            if(!confirm('WARNING: Are you sure you want to delete ALL records and frames? This cannot be undone.')) return;
            await fetch('/api/delete_all', { method: 'POST' });
            loadData();
        }

        // Initialize
        loadData();
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Surveillance Local Viewer")
    parser.add_argument('--dir', type=str, default='.', help='Directory containing surveillance.db and frames folder')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the viewer on')
    args = parser.parse_args()
    
    DATA_DIR = args.dir
    print(f"Starting viewer...")
    print(f"Looking for database at: {get_db_path()}")
    app.run(debug=True, port=args.port)
