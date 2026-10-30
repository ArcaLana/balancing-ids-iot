# Fundamental Teknik Balancing Data untuk IDS IoT

**Teori, Algoritma, dan Praktikum**

Repositori ini berisi kode pendamping dan hasil percobaan untuk buku *Fundamental Teknik Balancing Data untuk IDS IoT: Teori, Algoritma, dan Praktikum*. Seluruh percobaan memakai dataset publik **RT-IoT2022** dan dijalankan dengan Python di Google Colab.

## Penulis

1. Assoc. Prof. Dr. Ir. Zulhipni Reno Saputra Elsi, S.T., M.Kom.
2. Arif Maulana
3. Muhammad Ihsan, S.T., M.Kom.
4. Dedi Haryanto, S.Kom., M.Kom.

## Isi repositori

| Berkas / folder | Isi | Bab pada buku |
|---|---|---|
| `01_persiapan_dan_distribusi.ipynb` | Membaca dataset, memeriksa kualitas data, membagi data latih dan uji, melihat distribusi kelas | Bab 8 |
| `02_balancing_praktikum.ipynb` | Praktikum teknik balancing (ROS, RUS, SMOTE, Borderline-SMOTE, ADASYN, SMOTE-Tomek, SMOTE-ENN) pada data latih | Bab 9 |
| `03_model_dan_evaluasi.ipynb` | Mengintegrasikan balancing ke dalam pipeline dengan Random Forest dan XGBoost, lalu mengevaluasinya | Bab 10 |
| `04_evaluasi_dan_studi_kasus.ipynb` | Evaluasi lanjutan (matriks konfusi, metrik per kelas, uji ulang dengan beberapa seed, uji statistik, kebutuhan sumber daya) dan studi kasus | Bab 11 dan 12 |
| `kode_bab8/` sampai `kode_bab12/` | Potongan kode per nomor kode pada buku (misalnya Kode 8.2), dalam berkas `.py` | Bab 8 sampai 12 |
| `hasil_nyata/` | Keluaran asli dari menjalankan kode (tabel CSV, teks keluaran, gambar) | Bab 8 sampai 12 |

Nomor kode, tabel, dan gambar pada notebook sama dengan nomor pada buku, sehingga mudah dicocokkan.

## Dataset

Dataset **RT-IoT2022** tidak disertakan di repositori ini. Unduh dari UCI Machine Learning Repository:

<https://archive.ics.uci.edu/dataset/942/rt-iot2022>

Berkas yang dipakai bernama `RT_IOT2022.csv` (123.117 baris dan 85 kolom). Mohon baca ketentuan lisensi dan sitasi pada halaman UCI sebelum memakainya.

## Cara menjalankan di Google Colab

1. Buka notebook dari repositori ini di Colab (**File → Open notebook → GitHub**, lalu tempel alamat repositori), atau unduh notebook dan unggah ke <https://colab.research.google.com>.
2. Di panel **Files** sebelah kiri, buat folder `data`, lalu unggah `RT_IOT2022.csv` ke dalamnya. Pastikan berkasnya utuh (sekitar 52 MB) dan belum pernah disimpan ulang di Excel.
3. Jalankan sel **berurutan dari atas ke bawah**. Mulailah dari notebook `01`, karena notebook berikutnya memakai folder `output/` yang dihasilkan notebook sebelumnya.
4. Urutan yang disarankan: `01` → `02` → `03` → `04`.

Catatan waktu: Kode 10.6 dan 10.7 di notebook `03` memakan beberapa menit pada mesin dua inti. Kode 11.8 (sepuluh seed) di notebook `04` memakan sekitar 15 sampai 30 menit.

Berkas yang diunggah ke panel Files hilang ketika sesi Colab berakhir, jadi unggah ulang dataset setiap kali membuka sesi baru.

## Lingkungan yang dipakai penulis

| Pustaka | Versi |
|---|---|
| Python | 3.13.16 |
| pandas | 3.0.5 |
| NumPy | 2.5.3 |
| scikit-learn | 1.9.1 |
| imbalanced-learn | 0.14.2 |
| XGBoost | 3.4.1 |
| SciPy | 1.18.1 |
| Matplotlib | 3.11.2 |

Jumlah baris, jumlah kelas, dan hitungan per kelas pada keluaran Anda akan sama persis bila berkas dataset sama. Waktu eksekusi dan skor model dapat berbeda sedikit bila versi pustaka atau mesin Anda berbeda.

## Prinsip yang dipegang pada seluruh percobaan

- Data dibagi menjadi data latih dan data uji **sebelum** teknik balancing diterapkan. Balancing hanya dikenakan pada data latih, sehingga data uji tetap mencerminkan distribusi aslinya.
- Skor yang dilaporkan berasal dari hasil menjalankan kode di repositori ini, termasuk hasil yang kurang menguntungkan bagi suatu metode.

## Cara mengutip

Silakan sitasi buku ini dengan format berikut, lalu sesuaikan setelah buku terbit (penerbit, tahun, dan ISBN).

> Elsi, Z. R. S., Maulana, A., Ihsan, M., & Haryanto, D. (2026). *Fundamental Teknik Balancing Data untuk IDS IoT: Teori, Algoritma, dan Praktikum*. [Penerbit].

## Lisensi

Pilih satu lalu hapus bagian ini, misalnya MIT untuk kode: lihat berkas `LICENSE`. Dataset RT-IoT2022 tunduk pada lisensi pemilik aslinya di UCI.

## Kontak

Untuk pertanyaan dan laporan masalah, gunakan menu **Issues** pada repositori ini.
