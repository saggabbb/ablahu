import os
import difflib
from flask import Flask, render_template, send_from_directory, jsonify
from detector import process_image

app = Flask(__name__)

# Direktori data
DATASET_DIR = os.path.abspath('dataset')
OUTPUT_DIR = os.path.abspath('output')

@app.route('/')
def index():
    # Mengirimkan antarmuka web
    return render_template('index.html')

@app.route('/api/results')
def get_results():
    """Mengembalikan daftar semua gambar hasil analisis di folder output."""
    if not os.path.exists(OUTPUT_DIR):
        return jsonify([])
        
    results = []
    # Ambil semua file gambar di output
    extensions = {'.png', '.jpg', '.jpeg', '.webp'}
    for filename in os.listdir(OUTPUT_DIR):
        ext = os.path.splitext(filename)[1].lower()
        if ext in extensions:
            results.append(filename)
            
    # Sortir berdasarkan nama file agar rapi
    results.sort()
    return jsonify(results)

@app.route('/image/output/<path:filename>')
def serve_output(filename):
    """Serve gambar dari folder output."""
    return send_from_directory(OUTPUT_DIR, filename)

def find_original_image(filename):
    """Mencari gambar asli di dataset. Jika tidak ada yang sama persis, gunakan fuzzy match."""
    # 1. Exact match
    exact_path = os.path.join(DATASET_DIR, filename)
    if os.path.exists(exact_path):
        return exact_path
        
    # 2. Fuzzy match (karena kadang nama di output ketambahan/kurang digit '1' atau lainnya)
    if os.path.exists(DATASET_DIR):
        all_files = os.listdir(DATASET_DIR)
        matches = difflib.get_close_matches(filename, all_files, n=1, cutoff=0.8)
        if matches:
            return os.path.join(DATASET_DIR, matches[0])
            
    return None

@app.route('/image/dataset/<path:filename>')
def serve_dataset(filename):
    """Serve gambar asli."""
    target_path = find_original_image(filename)
    if target_path:
        return send_from_directory(os.path.dirname(target_path), os.path.basename(target_path))
    
    # Fallback jika tidak ketemu
    return send_from_directory(DATASET_DIR, filename)

@app.route('/api/analyze/<path:filename>')
def analyze_image(filename):
    """
    Menjalankan proses deteksi secara dinamis tanpa menyimpan gambar,
    hanya untuk mendapatkan hasil analisis JSON yang akurat.
    """
    target_path = find_original_image(filename)
            
    if not target_path:
        return jsonify({"error": "Gambar asli tidak ditemukan."}), 404
        
    result = process_image(target_path, output_path=None)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
