# UAS-SiskonLanjut
# Segmentasi Citra dengan Metode Otsu dan Genetic Algorithm

Proyek ini bertujuan untuk melakukan segmentasi citra (image segmentation) dengan menggabungkan metode ambang batas Otsu dan algoritma genetika (Genetic Algorithm/GA). Pendekatan ini digunakan untuk mencari nilai ambang batas (threshold) yang optimal dalam membedakan objek dan latar belakang pada gambar grayscale.

## Deskripsi Singkat

Metode Otsu secara manual digunakan untuk menghitung varians antar-kelas dari setiap nilai ambang batas. Nilai threshold terbaik dipilih berdasarkan nilai fitness tertinggi, yaitu varians antar-kelas terbesar. Proses pencarian threshold optimal tidak dilakukan secara langsung, tetapi menggunakan algoritma genetika yang meniru proses evolusi biologis.

## Fitur Program

- Konversi gambar RGB ke grayscale
- Perhitungan histogram gambar grayscale
- Penerapan algoritma genetika:
  - Representasi kromosom dalam bentuk biner 8-bit
  - Seleksi menggunakan metode turnamen
  - Crossover satu titik
  - Mutasi bit secara acak
  - Elitisme dan early stopping (berhenti jika tidak ada peningkatan selama 10 generasi)
- Visualisasi hasil berupa:
  - Histogram intensitas dengan garis ambang batas
  - Gambar hasil segmentasi biner
  - Plot konvergensi fitness

## Cara Menjalankan

1. Pastikan Python sudah terinstal.
2. Instal semua dependensi dengan menjalankan perintah:
pip install -r requirements.txt
3. Jalankan program dan inputkan path gambar ke dalam program main
