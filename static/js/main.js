document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const previewZone = document.getElementById('previewZone');
    const imagePreview = document.getElementById('imagePreview');
    const btnCancel = document.getElementById('btnCancel');
    const btnProcess = document.getElementById('btnProcess');
    
    const uploadPanel = document.getElementById('uploadPanel');
    const loadingState = document.getElementById('loadingState');
    const resultDashboard = document.getElementById('resultDashboard');
    
    const statusBanner = document.getElementById('statusBanner');
    const valKeypoints = document.getElementById('valKeypoints');
    const valMatches = document.getElementById('valMatches');
    const valResolution = document.getElementById('valResolution');
    
    const resOriginal = document.getElementById('resOriginal');
    const resOutput = document.getElementById('resOutput');
    const btnNewScan = document.getElementById('btnNewScan');

    let currentFile = null;

    // --- Drag & Drop Handlers ---
    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    function handleFile(file) {
        // Validasi ekstensi
        const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
        if (!validTypes.includes(file.type)) {
            alert('Format tidak didukung. Harap unggah gambar JPG, PNG, atau WEBP.');
            return;
        }

        currentFile = file;
        
        // Buat URL sementara untuk preview
        const objectUrl = URL.createObjectURL(file);
        imagePreview.src = objectUrl;
        
        // Sembunyikan drop zone, tampilkan preview
        dropZone.classList.add('hidden');
        previewZone.classList.remove('hidden');
    }

    // --- Actions ---
    btnCancel.addEventListener('click', () => {
        resetUploader();
    });

    btnNewScan.addEventListener('click', () => {
        resultDashboard.classList.add('hidden');
        uploadPanel.classList.remove('hidden');
        resetUploader();
    });

    function resetUploader() {
        currentFile = null;
        fileInput.value = '';
        previewZone.classList.add('hidden');
        dropZone.classList.remove('hidden');
    }

    // --- Proses Data ---
    btnProcess.addEventListener('click', async () => {
        if (!currentFile) return;

        // UI State: Loading
        uploadPanel.classList.add('hidden');
        loadingState.classList.remove('hidden');

        // Siapkan Form Data
        const formData = new FormData();
        formData.append('image', currentFile);

        try {
            const response = await fetch('/detect', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Terjadi kesalahan pada server');
            }

            // Sukses: Update UI dengan data
            populateDashboard(data);

            // UI State: Selesai
            loadingState.classList.add('hidden');
            resultDashboard.classList.remove('hidden');

        } catch (error) {
            alert(`Gagal: ${error.message}`);
            // Kembalikan ke panel upload
            loadingState.classList.add('hidden');
            uploadPanel.classList.remove('hidden');
        }
    });

    function populateDashboard(data) {
        // Render Images
        resOriginal.src = data.original_url;
        // Tambah query param agar browser tidak cache hasil yg namanya mungkin sama
        resOutput.src = `${data.result_url}?t=${new Date().getTime()}`;

        // Render Stats
        valKeypoints.textContent = data.keypoints.toLocaleString();
        valMatches.textContent = data.matches.toLocaleString();
        valResolution.textContent = `${data.width}x${data.height}`;

        // Render Status Banner
        if (data.is_forged) {
            statusBanner.className = 'status-banner status-danger';
            statusBanner.innerHTML = `
                <i class="fa-solid fa-triangle-exclamation"></i>
                <div>
                    <div><strong>TERINDIKASI COPY-MOVE FORGERY</strong></div>
                    <div style="font-size: 0.85rem; font-weight: 400; margin-top: 2px;">
                        Algoritma SIFT mendeteksi ${data.matches} pasangan area yang identik. Garis hijau menunjukkan titik yang digandakan.
                    </div>
                </div>
            `;
        } else {
            statusBanner.className = 'status-banner status-safe';
            statusBanner.innerHTML = `
                <i class="fa-solid fa-shield-check"></i>
                <div>
                    <div><strong>TIDAK TERDETEKSI MANIPULASI</strong></div>
                    <div style="font-size: 0.85rem; font-weight: 400; margin-top: 2px;">
                        Gambar terlihat normal. Tidak ditemukan area fitur lokal yang digandakan secara signifikan.
                    </div>
                </div>
            `;
        }
    }
});
