document.addEventListener('DOMContentLoaded', () => {
    
    // DOM Elements
    const galleryGrid = document.getElementById('galleryGrid');
    const loadingGallery = document.getElementById('loadingGallery');
    const totalCount = document.getElementById('totalCount');
    
    const emptyState = document.getElementById('emptyState');
    const resultDashboard = document.getElementById('resultDashboard');
    
    const currentFileName = document.getElementById('currentFileName');
    const resOriginal = document.getElementById('resOriginal');
    const resOutput = document.getElementById('resOutput');
    
    const analysisLoading = document.getElementById('analysisLoading');
    const analysisLoadingText = document.getElementById('analysisLoadingText');
    const analysisContent = document.getElementById('analysisContent');
    const statusBanner = document.getElementById('statusBanner');
    const valKeypoints = document.getElementById('valKeypoints');
    const valMatches = document.getElementById('valMatches');
    const analysisInterpretation = document.getElementById('analysisInterpretation');
    
    // Fetch Data
    fetchGalleryData();
    
    async function fetchGalleryData() {
        try {
            const res = await fetch('/api/results');
            const files = await res.json();
            
            loadingGallery.classList.add('hidden');
            totalCount.textContent = files.length;
            
            if(files.length === 0) {
                galleryGrid.innerHTML = '<p style="grid-column: 1/-1; text-align: center; color: var(--text-muted); font-size: 0.9rem;">Tidak ada hasil ditemukan.</p>';
                return;
            }
            
            // Render Thumbnails
            files.forEach(filename => {
                const thumb = document.createElement('div');
                thumb.className = 'thumbnail-item';
                // Gunakan endpoint output untuk thumbnail agar cepat (bisa diganti jika terlalu berat)
                thumb.innerHTML = `
                    <img src="/image/output/${filename}" alt="${filename}" loading="lazy">
                    <div class="thumbnail-name">${filename}</div>
                `;
                
                thumb.addEventListener('click', () => {
                    // Remove active class from all
                    document.querySelectorAll('.thumbnail-item').forEach(el => el.classList.remove('active'));
                    thumb.classList.add('active');
                    showDetails(filename);
                });
                
                galleryGrid.appendChild(thumb);
            });
            
        } catch (err) {
            console.error("Gagal memuat galeri:", err);
            loadingGallery.innerHTML = '<p class="text-red">Gagal memuat data.</p>';
        }
    }
    
    function showDetails(filename) {
        // Sembunyikan state kosong, tampilkan detail
        emptyState.classList.add('hidden');
        resultDashboard.classList.remove('hidden');
        
        currentFileName.textContent = filename;
        
        // Tampilkan state loading analisis
        analysisContent.classList.add('hidden');
        analysisLoading.classList.remove('hidden');
        analysisLoadingText.classList.remove('hidden');
        
        // Animasi fade in ringan untuk gambar
        resOriginal.style.opacity = 0;
        resOutput.style.opacity = 0;
        
        // Set Source Gambar
        resOriginal.src = `/image/dataset/${filename}`;
        resOutput.src = `/image/output/${filename}`;
        
        // Kembalikan opacity setelah gambar dimuat
        resOriginal.onload = () => { resOriginal.style.opacity = 1; resOriginal.style.transition = "opacity 0.5s ease"; };
        resOutput.onload = () => { resOutput.style.opacity = 1; resOutput.style.transition = "opacity 0.5s ease"; };
        
        // Fetch Analisis SIFT
        fetch(`/api/analyze/${filename}`)
            .then(res => res.json())
            .then(data => {
                analysisLoading.classList.add('hidden');
                analysisLoadingText.classList.add('hidden');
                
                if (data.error) {
                    analysisContent.classList.remove('hidden');
                    analysisContent.innerHTML = `<p class="text-red">Gagal menganalisis: ${data.error}</p>`;
                    return;
                }
                
                analysisContent.classList.remove('hidden');
                
                // Set Nilai Stats
                valKeypoints.textContent = data.keypoints.toLocaleString();
                valMatches.textContent = data.matches.toLocaleString();
                
                // Set Status & Interpretasi
                if (data.is_forged) {
                    statusBanner.className = 'status-banner status-danger';
                    statusBanner.innerHTML = `
                        <i class="fa-solid fa-triangle-exclamation"></i>
                        <div>
                            <div><strong>TERINDIKASI COPY-MOVE FORGERY</strong></div>
                        </div>
                    `;
                    
                    analysisInterpretation.innerHTML = `
                        Algoritma SIFT mendeteksi adanya <strong>${data.keypoints.toLocaleString()}</strong> titik fitur (keypoints) pada gambar ini. 
                        Setelah dicocokkan menggunakan algoritma FLANN dan difilter dengan <em>Lowe's Ratio Test</em> beserta validasi spasial RANSAC, 
                        ditemukan <strong>${data.matches}</strong> pasangan titik yang memiliki kemiripan identik dan posisi yang konsisten secara geometris. 
                        <br><br>
                        Kehadiran ${data.matches} titik identik di lokasi yang berbeda ini <strong>sangat mengindikasikan</strong> bahwa sebuah area pada gambar telah disalin (di-copy) dan ditempel (di-paste) ke bagian lain. Garis hijau pada gambar hasil menghubungkan area sumber dan area salinannya.
                    `;
                } else {
                    statusBanner.className = 'status-banner status-safe';
                    statusBanner.innerHTML = `
                        <i class="fa-solid fa-shield-check"></i>
                        <div>
                            <div><strong>TIDAK TERDETEKSI MANIPULASI</strong></div>
                        </div>
                    `;
                    
                    analysisInterpretation.innerHTML = `
                        Algoritma SIFT mengekstrak <strong>${data.keypoints.toLocaleString()}</strong> titik fitur dari gambar ini. 
                        Berdasarkan pemindaian menyeluruh, sistem hanya menemukan <strong>${data.matches}</strong> kecocokan yang lolos filter (kurang dari batas minimum 4 kecocokan).
                        <br><br>
                        Kesimpulannya, gambar ini terlihat <strong>normal dan bersih</strong>. Tidak ada pola penggandaan fitur lokal yang signifikan yang mengarah pada manipulasi <em>copy-move</em>.
                    `;
                }
            })
            .catch(err => {
                console.error("Analysis Error:", err);
                analysisLoading.classList.add('hidden');
                analysisLoadingText.textContent = "Gagal memuat analisis statis.";
            });
    }
});
