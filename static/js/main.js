document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const galleryGrid = document.getElementById('galleryGrid');
    const loadingGallery = document.getElementById('loadingGallery');
    const totalCount = document.getElementById('totalCount');
    const searchInput = document.getElementById('searchInput');
    const filterStatus = document.getElementById('filterStatus');
    const sortBy = document.getElementById('sortBy');
    const statTotal = document.getElementById('statTotal');
    const statForged = document.getElementById('statForged');
    const statClean = document.getElementById('statClean');
    const statVisible = document.getElementById('statVisible');
    const analysisMeta = document.getElementById('analysisMeta');

    const emptyState = document.getElementById('emptyState');
    const resultDashboard = document.getElementById('resultDashboard');
    const closeDetailBtn = document.getElementById('closeDetailBtn');

    const currentFileName = document.getElementById('currentFileName');
    const resOriginal = document.getElementById('resOriginal');
    const resOutput = document.getElementById('resOutput');

    const analysisLoading = document.getElementById('analysisLoading');
    const analysisContent = document.getElementById('analysisContent');
    const statusBanner = document.getElementById('statusBanner');
    const valKeypoints = document.getElementById('valKeypoints');
    const valMatches = document.getElementById('valMatches');
    const analysisInterpretation = document.getElementById('analysisInterpretation');

    const sidebar = document.querySelector('aside');
    const detailSection = document.querySelector('section');
    const mobileMenuBtn = document.getElementById('mobileMenuBtn');
    const mobileNavPanel = document.getElementById('mobileNavPanel');
    const toastRoot = document.getElementById('toastRoot');
    const themeToggleBtn = document.getElementById('themeToggleBtn');
    const themeToggleIcon = document.getElementById('themeToggleIcon');

    // Theme: dark/light (persisted)
    initTheme();
    if (themeToggleBtn) {
        themeToggleBtn.addEventListener('click', toggleTheme);
    }

    function initTheme() {
        const saved = localStorage.getItem('theme'); // 'dark' | 'light' | null
        if (saved === 'dark' || saved === 'light') {
            applyTheme(saved);
            return;
        }
        const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
        applyTheme(prefersDark ? 'dark' : 'light');
    }

    function toggleTheme() {
        const isDark = document.documentElement.classList.contains('dark');
        applyTheme(isDark ? 'light' : 'dark');
        localStorage.setItem('theme', isDark ? 'light' : 'dark');
    }

    function applyTheme(theme) {
        if (theme === 'dark') {
            document.documentElement.classList.add('dark');
            document.documentElement.classList.remove('light');
            if (themeToggleIcon) {
                themeToggleIcon.className = 'fa-solid fa-moon';
            }
            themeToggleBtn?.setAttribute('aria-label', 'Switch to light mode');
            themeToggleBtn?.setAttribute('title', 'Light mode');
        } else {
            document.documentElement.classList.remove('dark');
            document.documentElement.classList.add('light');
            if (themeToggleIcon) {
                themeToggleIcon.className = 'fa-solid fa-sun';
            }
            themeToggleBtn?.setAttribute('aria-label', 'Switch to dark mode');
            themeToggleBtn?.setAttribute('title', 'Dark mode');
        }
    }

    const galleryState = {
        files: [],
        selectedFilename: null,
        isStatsAnalyzing: false,
        analyzedCount: 0
    };

    function escapeHtml(str) {
        return String(str)
            .replaceAll('&', '&amp;')
            .replaceAll('<', '&lt;')
            .replaceAll('>', '&gt;')
            .replaceAll('"', '&quot;')
            .replaceAll("'", '&#039;');
    }

    function showToast(type, title, message, options = {}) {
        if (!toastRoot) return;
        const variant = ['success', 'error', 'info'].includes(type) ? type : 'info';
        const timeoutMs = Number.isFinite(options.timeoutMs) ? options.timeoutMs : 4000;

        const iconHtml =
            variant === 'success'
                ? '<i class="fa-solid fa-circle-check"></i>'
                : variant === 'error'
                ? '<i class="fa-solid fa-triangle-exclamation"></i>'
                : '<i class="fa-solid fa-circle-info"></i>';

        const toast = document.createElement('div');
        toast.className = `toast toast--${variant}`;
        toast.innerHTML = `
            <div class="toast-icon">${iconHtml}</div>
            <div>
                <div class="toast-title">${escapeHtml(title || 'Info')}</div>
                <div class="toast-msg">${escapeHtml(message || '')}</div>
            </div>
            <button class="toast-close" type="button" aria-label="Close toast">
                <i class="fa-solid fa-xmark"></i>
            </button>
        `;

        const closeBtn = toast.querySelector('.toast-close');
        const close = () => {
            if (toast.classList.contains('toast-leave')) return;
            toast.classList.add('toast-leave');
            setTimeout(() => toast.remove(), 220);
        };

        closeBtn?.addEventListener('click', close);
        toastRoot.appendChild(toast);
        if (timeoutMs > 0) setTimeout(close, timeoutMs);
    }

    if (searchInput) {
        searchInput.addEventListener('input', renderGallery);
    }
    if (filterStatus) {
        filterStatus.addEventListener('change', renderGallery);
    }
    if (sortBy) {
        sortBy.addEventListener('change', renderGallery);
    }

    // Fetch Data
    fetchGalleryData();

    // Mobile Top Navbar Toggle
    if (mobileMenuBtn && mobileNavPanel) {
        mobileMenuBtn.addEventListener('click', () => {
            mobileNavPanel.classList.toggle('open');
        });

        document.addEventListener('click', (event) => {
            const clickedInsidePanel = mobileNavPanel.contains(event.target);
            const clickedToggle = mobileMenuBtn.contains(event.target);
            if (!clickedInsidePanel && !clickedToggle) {
                mobileNavPanel.classList.remove('open');
            }
        });
    }
    // Mobile Close Button
    if (closeDetailBtn) {
        closeDetailBtn.addEventListener('click', () => {
            sidebar.classList.remove('hidden');
            sidebar.classList.add('flex');

            detailSection.classList.add('hidden');
            detailSection.classList.remove('flex');
        });
    }

    // Ensure desktop always shows both panels after mobile transitions
    window.addEventListener('resize', () => {
        if (window.innerWidth >= 1024) {
            sidebar.classList.remove('hidden');
            sidebar.classList.add('flex');
            detailSection.classList.remove('hidden');
            detailSection.classList.add('flex');
            if (mobileNavPanel) {
                mobileNavPanel.classList.remove('open');
            }
        }
    });

    async function fetchGalleryData() {
        try {
            const res = await fetch('/api/results');
            const files = await res.json();

            galleryState.files = files.map((filename) => ({
                filename,
                status: 'unknown',
                matches: null
            }));
            galleryState.isStatsAnalyzing = true;
            galleryState.analyzedCount = 0;

            loadingGallery.classList.add('hidden');
            galleryGrid.classList.remove('hidden');
            totalCount.textContent = files.length;
            updateStats();
            renderGallery();

            if (files.length === 0) {
                galleryGrid.innerHTML = '<p class="col-span-full text-center text-slate-500 text-sm py-8">Tidak ada hasil ditemukan.</p>';
                if (analysisMeta) {
                    analysisMeta.innerHTML = '<i class="fa-solid fa-circle-info"></i> Tidak ada data untuk dianalisis.';
                }
                return;
            }

            // Enrich sidebar stats with existing analysis API response
            await enrichAnalysisStats();

        } catch (err) {
            console.error("Gagal memuat galeri:", err);
            loadingGallery.innerHTML = '<p class="text-rose-400 text-sm">Gagal memuat data.</p>';
            loadingGallery.classList.remove('hidden');
            if (analysisMeta) {
                analysisMeta.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i> Statistik tidak dapat dimuat.';
            }
            showToast('error', 'Gagal memuat galeri', 'Tidak bisa mengambil data dari server. Coba refresh halaman.');
        }
    }

    function getFilteredAndSortedFiles() {
        const q = (searchInput?.value || '').trim().toLowerCase();
        const statusFilter = filterStatus?.value || 'all';
        const sortMode = sortBy?.value || 'name_asc';

        let list = galleryState.files.filter((item) => {
            const passSearch = q.length === 0 || item.filename.toLowerCase().includes(q);
            const passStatus = statusFilter === 'all' || item.status === statusFilter;
            return passSearch && passStatus;
        });

        list = list.sort((a, b) => {
            if (sortMode === 'name_desc') {
                return b.filename.localeCompare(a.filename);
            }
            if (sortMode === 'matches_desc') {
                const aMatches = Number.isFinite(a.matches) ? a.matches : -1;
                const bMatches = Number.isFinite(b.matches) ? b.matches : -1;
                if (bMatches !== aMatches) return bMatches - aMatches;
                return a.filename.localeCompare(b.filename);
            }
            return a.filename.localeCompare(b.filename);
        });

        return list;
    }

    function renderGallery() {
        if (!galleryGrid) return;
        galleryGrid.innerHTML = '';
        const list = getFilteredAndSortedFiles();

        if (list.length === 0) {
            galleryGrid.innerHTML = '<p class="col-span-full text-center text-slate-500 text-sm py-8">Tidak ada file yang cocok dengan filter.</p>';
            updateStats();
            return;
        }

        list.forEach((item, index) => {
            const thumb = document.createElement('div');
            thumb.className = 'gallery-item group relative aspect-square animate-fade-in';
            thumb.style.animationDelay = `${index * 0.03}s`;
            if (galleryState.selectedFilename === item.filename) {
                thumb.classList.add('active');
            }

            const badge = getStatusBadge(item);
            thumb.innerHTML = `
                <div class="absolute inset-0 bg-black/40 group-hover:bg-black/15 transition-colors z-10"></div>
                <img src="/image/output/${item.filename}" alt="${item.filename}" loading="lazy" class="w-full h-full object-cover transition-transform duration-500 group-hover:scale-110">
                <div class="absolute top-1.5 right-1.5 z-30">${badge}</div>
                <div class="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/90 via-black/50 to-transparent p-2 pt-6 z-20">
                    <div class="text-[0.65rem] sm:text-xs font-medium text-slate-200 truncate text-center drop-shadow-md">${item.filename}</div>
                </div>
            `;

            thumb.addEventListener('click', () => {
                galleryState.selectedFilename = item.filename;
                renderGallery();
                showDetails(item.filename);
            });

            galleryGrid.appendChild(thumb);
        });

        updateStats();
    }

    function getStatusBadge(item) {
        if (item.status === 'forged') {
            return '<span class="text-[0.6rem] font-bold px-1.5 py-0.5 rounded bg-rose-500/90 text-white">FORGED</span>';
        }
        if (item.status === 'clean') {
            return '<span class="text-[0.6rem] font-bold px-1.5 py-0.5 rounded bg-emerald-500/90 text-white">CLEAN</span>';
        }
        return '<span class="text-[0.6rem] font-semibold px-1.5 py-0.5 rounded bg-slate-700/90 text-slate-200">...</span>';
    }

    function updateStats() {
        const total = galleryState.files.length;
        const forged = galleryState.files.filter((x) => x.status === 'forged').length;
        const clean = galleryState.files.filter((x) => x.status === 'clean').length;
        const visible = getFilteredAndSortedFiles().length;

        if (statTotal) statTotal.textContent = total.toLocaleString();
        if (statForged) statForged.textContent = forged.toLocaleString();
        if (statClean) statClean.textContent = clean.toLocaleString();
        if (statVisible) statVisible.textContent = visible.toLocaleString();
        if (totalCount) totalCount.textContent = total.toLocaleString();

        if (analysisMeta) {
            if (total === 0) {
                analysisMeta.innerHTML = '<i class="fa-solid fa-circle-info"></i> Belum ada data galeri.';
                return;
            }
            if (galleryState.isStatsAnalyzing) {
                analysisMeta.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Statistik: ${galleryState.analyzedCount}/${total} file dianalisis`;
            } else {
                analysisMeta.innerHTML = `<i class="fa-solid fa-chart-pie"></i> Statistik siap (${galleryState.analyzedCount}/${total} file)`;
            }
        }
    }

    async function enrichAnalysisStats() {
        const tasks = galleryState.files.map(async (item) => {
            try {
                const res = await fetch(`/api/analyze/${item.filename}`);
                const data = await res.json();
                if (!data.error) {
                    item.status = data.is_forged ? 'forged' : 'clean';
                    item.matches = Number.isFinite(data.matches) ? data.matches : null;
                }
            } catch (error) {
                console.error(`Stats analyze failed for ${item.filename}:`, error);
            } finally {
                galleryState.analyzedCount += 1;
                updateStats();
            }
        });

        await Promise.allSettled(tasks);
        galleryState.isStatsAnalyzing = false;
        updateStats();
        renderGallery();

        if (galleryState.files.length > 0) {
            showToast('success', 'Statistik siap', 'Search, filter, dan sort sudah bisa pakai status forged/clean.', { timeoutMs: 2500 });
        }
    }

    function showDetails(filename) {
        if (window.innerWidth < 1024) {
            sidebar.classList.add('hidden');
            sidebar.classList.remove('flex');

            detailSection.classList.remove('hidden');
            detailSection.classList.add('flex');
        }

        emptyState.classList.add('hidden');
        resultDashboard.classList.remove('hidden');
        resultDashboard.classList.add('flex');

        currentFileName.textContent = filename;

        analysisContent.classList.add('hidden');
        analysisLoading.classList.remove('hidden');
        analysisLoading.classList.add('flex');

        resOriginal.style.opacity = 0;
        resOutput.style.opacity = 0;

        resOriginal.src = `/image/dataset/${filename}`;
        resOutput.src = `/image/output/${filename}`;

        resOriginal.onload = () => { resOriginal.style.opacity = 1; };
        resOutput.onload = () => { resOutput.style.opacity = 1; };
        resOriginal.onerror = () => {
            showToast('error', 'Gagal memuat gambar', 'Original image tidak bisa dimuat.');
        };
        resOutput.onerror = () => {
            showToast('error', 'Gagal memuat gambar', 'Detection result tidak bisa dimuat.');
        };

        fetch(`/api/analyze/${filename}`)
            .then(res => res.json())
            .then(data => {
                analysisLoading.classList.add('hidden');
                analysisLoading.classList.remove('flex');

                if (data.error) {
                    analysisContent.classList.remove('hidden');
                    analysisContent.innerHTML = `
                        <div class="p-4 bg-rose-500/10 border border-rose-500/30 rounded-xl text-rose-400 text-sm">
                            <i class="fa-solid fa-circle-exclamation mr-2"></i> Gagal menganalisis: ${data.error}
                        </div>`;
                    showToast('error', 'Analisis gagal', 'Server mengembalikan error saat menganalisis file ini.');
                    return;
                }

                analysisContent.classList.remove('hidden');

                valKeypoints.textContent = data.keypoints.toLocaleString();
                valMatches.textContent = data.matches.toLocaleString();

                if (data.is_forged) {
                    statusBanner.className = 'rounded-xl p-4 sm:p-5 flex items-start gap-4 border shadow-lg transition-all border-rose-500/30 bg-rose-500/10 shadow-rose-500/5';
                    statusBanner.innerHTML = `
                        <div class="p-2.5 bg-rose-500/20 rounded-xl shrink-0 mt-0.5 border border-rose-500/30">
                            <i class="fa-solid fa-triangle-exclamation text-rose-400 text-xl"></i>
                        </div>
                        <div>
                            <h3 class="text-rose-400 font-bold tracking-wide text-sm sm:text-base mb-1 uppercase">TERINDIKASI COPY-MOVE FORGERY</h3>
                            <p class="text-slate-300 text-xs sm:text-sm leading-relaxed">Peringatan: Ditemukan tingkat duplikasi fitur yang tinggi pada citra ini.</p>
                        </div>
                    `;
                    
                    analysisInterpretation.innerHTML = `
                        Algoritma mendeteksi <strong>${data.keypoints.toLocaleString()}</strong> titik fitur (keypoints) SIFT. 
                        Setelah dicocokkan menggunakan <em>FLANN</em> dan <em>RANSAC</em>, 
                        ditemukan <strong>${data.matches}</strong> pasangan titik yang saling berkorespondensi dan terverifikasi stabil secara spasial. 
                        <br><br>
                        Keberadaan ${data.matches} kecocokan ini melampaui ambang batas normal, yang <strong>sangat mengindikasikan</strong> adanya manipulasi gambar berupa penyalinan (copy-move) dari satu area ke area lain. Area yang dicurigai terhubung oleh garis hijau pada panel di bawah.
                    `;
                } else {
                    statusBanner.className = 'rounded-xl p-4 sm:p-5 flex items-start gap-4 border shadow-lg transition-all border-emerald-500/30 bg-emerald-500/10 shadow-emerald-500/5';
                    statusBanner.innerHTML = `
                        <div class="p-2.5 bg-emerald-500/20 rounded-xl shrink-0 mt-0.5 border border-emerald-500/30">
                            <i class="fa-solid fa-shield-check text-emerald-400 text-xl"></i>
                        </div>
                        <div>
                            <h3 class="text-emerald-400 font-bold tracking-wide text-sm sm:text-base mb-1 uppercase">TIDAK TERDETEKSI MANIPULASI</h3>
                            <p class="text-slate-300 text-xs sm:text-sm leading-relaxed">Citra terverifikasi normal tanpa pola duplikasi buatan.</p>
                        </div>
                    `;
                    
                    analysisInterpretation.innerHTML = `
                        Sistem mengekstrak <strong>${data.keypoints.toLocaleString()}</strong> titik fitur (keypoints) dari gambar. 
                        Berdasarkan pemindaian menyeluruh, hanya ditemukan <strong>${data.matches}</strong> kecocokan yang lolos verifikasi spasial.
                        <br><br>
                        Jumlah ini berada di bawah batas ambang anomali, sehingga gambar ini diklasifikasikan sebagai <strong>asli / bersih</strong>. Tidak ada pola penggandaan fitur lokal yang signifikan yang mengarah pada manipulasi <em>copy-move forgery</em>.
                    `;
                }
            })
            .catch(err => {
                console.error("Analysis Error:", err);
                analysisLoading.classList.add('hidden');
                analysisLoading.classList.remove('flex');
                analysisContent.classList.remove('hidden');
                analysisContent.innerHTML = `
                    <div class="p-4 bg-rose-500/10 border border-rose-500/30 rounded-xl text-rose-400 text-sm">
                        <i class="fa-solid fa-wifi mr-2"></i> Gagal menghubungi server analisis.
                    </div>`;
                showToast('error', 'Tidak bisa menghubungi server', 'Coba cek koneksi atau restart server Flask.');
            });
    }
});
