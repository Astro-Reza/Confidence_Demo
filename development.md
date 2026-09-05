# Pipeline Simulasi Sensor Fusion (Berdasarkan `development.ipynb`)

Dokumen ini menjelaskan tahapan lengkap (*step-by-step*) metode simulasi dan algoritma **confidence-weighted quaternion fusion** untuk sistem penunjukan antena (*antenna pointing system*) berdasarkan percobaan yang ada di dalam *notebook* `development.ipynb`.

Pipeline ini berjalan melalui 5 tahap utama secara berurutan:

### 1. Pembuatan Data dan Lingkungan (Data Generation & Environment)
Tahap pertama adalah menyimulasikan lintasan rotasi antena yang sesungguhnya (*ground truth*) dalam satu sumbu (*single-axis*) secara realistis, di mana sudut rotasi dibatasi pada rentang fisik **[-90°, +90°]**.
* **Kecepatan Sudut (*Angular Velocity*):** Dimodelkan menggunakan prinsip **Gerak Brown** (*Brownian motion* atau proses Wiener) dengan tambahan efek redaman ringan (*damping coefficient* $\alpha = 0.998$). Hal ini mensimulasikan percepatan/pergerakan kecepatan yang acak (terdistribusi secara normal) namun realistis sehingga kecepatan tidak akan membesar/melambung tak terkendali seiring waktu.
* **Sudut Rotasi (*Angle*):** Diperoleh dengan mengintegrasikan nilai kecepatan sudut di atas terhadap satuan waktu (*dt*). Agar tetap realistis pada dunia fisik, sistem juga menggunakan batasan pemantulan (*reflective boundaries*) sehingga jika antena mencapai batas mekanis ±90°, arah pergerakan antena (kecepatan) akan terbalik seolah memantul menjauhi batas.

### 2. Pemodelan Sensor dan *Noise* (Sensor Readings & Noise)
Langkah selanjutnya adalah mengambil sudut rotasi sesungguhnya (dari tahap 1) dan menyimulasikan dua jenis sensor independen dengan karakteristik *noise* (gangguan pembacaan) dan masalah yang berbeda:
* **Sensor Encoder:** Sensor ini dimodelkan sebagai sensor dengan tingkat kepresisian tinggi (*base noise* rendah). Namun, kelemahannya adalah disuntikkannya **peristiwa *drift*** (penyimpangan titik 0). *Drift* ini menyimulasikan dua kondisi kegagalan nyata: masalah selip mekanik (gangguan bertahap yang perlahan naik/turun) maupun masalah *glitch* kelistrikan (menyimpang tiba-tiba ke sebuah angka secara instan).
* **Sensor AHRS:** Sensor ini dimodelkan sebagai sensor dengan *base noise* (fluktuasi/getaran bawaan) yang lebih besar dan kasar dibandingkan Encoder secara keseluruhan, tetapi sensor ini stabil dan tidak akan mengalami masalah *drift* atau *offset* seiring waktu.

Asimetri dan kelemahan pada karakteristik kedua sensor inilah yang menjadi fokus perbaikan; sebuah sistem algoritma gabungan harus bisa mengenali dan memprioritaskan "sensor mana yang paling bisa dipercaya" di kondisi tertentu.

### 3. Perhitungan Skor Kepercayaan (Confidence Scoring)
Pada setiap langkah waktu, algoritma akan melakukan evaluasi metrik untuk menghasilkan skor kepercayaan (*confidence score* - $c_i$) pada rentang angka $0$ hingga $1$ untuk masing-masing sensor.
* **Varians Pengukuran Bergulir (*Rolling Measurement Variance* - $\sigma_i^2$):** Dihitung berdasarkan sampel historis pembacaan sensor selama waktu singkat (*sliding window* sebanyak 50 sampel yang setara 0.5 detik). Varians berfungsi untuk mengukur seberapa bergetar atau seberapa berisik (*noisy*) kinerja sebuah sensor secara *real-time*.
* **Kesalahan Residual (*Residual Error* - $\epsilon_i$):** Selisih absolut antara pembacaan sudut sebuah sensor dibandingkan dengan nilai rata-rata (*mean*) seluruh pembacaan sensor secara gabungan. Ini mengukur seberapa jauh sebuah sensor menyimpang sendiri (tidak setuju dengan tren konsensus).

Dengan fungsi $c_i = \frac{1}{1 + \sigma_i^2 + \epsilon_i}$, skor ini secara dinamis dan otomatis **menjatuhkan** tingkat kepercayaan pada sebuah sensor jika di saat tersebut fluktuasi *noise* sangat tinggi, ATAU saat terjadi kejadian penyimpangan *drift* (yang membuat kesalahan residual meningkat tajam).

### 4. Penggabungan Menggunakan Kuaternion (*Confidence-Weighted Quaternion Fusion*)
Ini adalah langkah utama yang mentransformasikan dan menggabungkan data sensor berdasarkan skor di tahap sebelumnya:
* **Konversi ke Kuaternion:** Pertama, pembacaan sudut dari Encoder maupun AHRS dikonversi ke dalam format **unit kuaternion** rotasi 3D: $\mathbf{q}_i = [\cos(\theta_i/2), \sin(\theta_i/2), 0, 0]^T$. Pendekatan kuaternion dipakai karena merupakan basis perputaran spasial paling kuat untuk menolak singularitas matematis (seperti *gimbal lock*).
* **Rata-Rata Berbobot:** Gabungan data (*fused quaternion*) dihitung dengan mencari **rata-rata berbobot** (*weighted sum*) dari komponen vektor kuaternion sensor. Bobot kalinya diambil berdasarkan skor $c_i$ (sehingga sensor dengan kepercayaan tertinggi akan menyumbang nilai dominan pada orientasi akhir).
* **Normalisasi:** Kuaternion gabungan akhir harus **dinormalisasi** kembali ke magnitude/vektor absolut sama dengan $1$ agar rotasi tetap direpresentasikan secara valid dalam fisika 3D. Barulah unit kuaternion kembali diekstrak menjadi sebuah derajat posisi (*angle* final).

### 5. Evaluasi Akhir (Final Preview & Error Analysis)
Langkah penutup berfokus pada demonstrasi dan pembuktian matematis terhadap metode di atas:
* **Visualisasi Penuh:** Notebook menampilkan rangkuman grafik 4-panel (*4-panel view*) yang berisi perbandingan (1) lintasan gerak *ground truth*, (2) data pengukuran sensor kasar yang telah diberi distorsi dan cacat *drift*, (3) respon dinamika *confidence scores* seiring waktu, dan akhirnya (4) keluaran sistem *quaternion fusion*. Di bagian 4, terlihat algoritma secara sempurna mengabaikan cacat mendadak pada Encoder karena skor kepercayaannya disunat oleh fungsi residual pada kejadian tersebut.
* **Metrik RMSE:** Sistem ini juga divalidasi kinerjanya secara terukur dengan metode RMSE (*Root Mean Square Error*), dimana tingkat kesalahan dari keluaran algoritma *fused output* jauh lebih kecil (peningkatan akurasi) dibandingkan jika hanya bergantung pada data satu buah sensor secara mentah, berkat penyaringan skor secara presisi.
