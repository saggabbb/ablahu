import os
import uuid
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
from detector import process_image

app = Flask(__name__)

# Konfigurasi Upload
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['RESULT_FOLDER'] = os.path.join('static', 'results')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # Max 16MB

# Pastikan folder ada
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    # Cek apakah ada file dalam request
    if 'image' not in request.files:
        return jsonify({'error': 'Tidak ada file gambar yang diunggah.'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'Nama file kosong.'}), 400
        
    if file and allowed_file(file.filename):
        # Buat nama unik agar tidak tertimpa
        ext = file.filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{uuid.uuid4().hex}.{ext}"
        
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        output_path = os.path.join(app.config['RESULT_FOLDER'], unique_filename)
        
        # Simpan file yang diunggah
        file.save(input_path)
        
        # Proses gambar
        try:
            result_data = process_image(input_path, output_path)
            
            if "error" in result_data:
                return jsonify({'error': result_data["error"]}), 500
                
            # Tambahkan URL gambar untuk ditampilkan di frontend
            result_data['original_url'] = url_for('static', filename=f"uploads/{unique_filename}")
            result_data['result_url'] = url_for('static', filename=f"results/{unique_filename}")
            
            return jsonify(result_data)
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
            
    return jsonify({'error': 'Format file tidak diizinkan. Hanya JPG, PNG, WEBP.'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
