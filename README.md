# Laporan Proyek Machine Learning -Kori Isabella Hutabarat

## Domain Proyek

Domain proyek yang dipilih dalam proyek _Machine Learning_ ini adalah mengenai **kesehatan** dengan judul "Klasifikasi Penyakit Jantung".

Jantung merupakan pusat dari bagian tubuh manusia. Jantung berfungsi untuk memompa darah ke seluruh tubuh dan kembali lagi ke jantung melalui pembuluh darah. Jantung terletak pada rongga dada sebelah kiri di antara paru-paru kanan dan paru-paru kiri. Gangguan pada jantung dapat mengakibatkan terganggunya sistem peredaran darah pada tubuh manusia yang mana tentu berdampak sangat fatal bahkan menyebabkan kematian. Menurut [Dr. Tania Savitri](https://hellosehat.com/jantung/penyakit-jantung/pengertian-penyakit-jantung/),  penyakit jantung dan pembuluh darah atau penyakit kardiovaskular adalah berbagai kondisi di mana terjadi penyempitan atau penyumbatan pembuluh darah yang dapat menyebabkan serangan jantung, nyeri dada (angina), atau stroke. Penyakit kardiovaskuler termasuk kondisi kritis yang butuh penanganan segera. Pasalnya, jantung adalah organ vital yang berfungsi untuk memompa darah ke seluruh tubuh. Jika jantung bermasalah, peredaran darah dalam tubuh bisa terganggu. Tanpa pertolongan medis yang sesuai, penyakit kardiovaskuler bisa mengancam jiwa dan menyebabkan kematian. Data [WHO World Health Organization (WHO)](http://p2ptm.kemkes.go.id/uploads/VHcrbkVobjRzUDN3UCs4eUJ0dVBndz09/2018/09/Mengenali_tanda_dan_gejala_serangan_dini_penyakit_jantung_dr_Bambang_Dwiputra_Hari_Jantung_Sedunia_2018.pdf) menunjukkan 70% kematian di dunia disebabkan oleh Penyakit Tidak Menular, 45% diantaranya disebabkan oleh penyakit jantung dan pembuluh darah, yaitu 17.7 juta dari 39,5 juta kematian. Data dari [Kementerian Kesehatan](https://kemkes.go.id/article/view/17073100005/penyakit-jantung-penyebab-kematian-tertinggi-kemenkes-ingatkan-cerdik-.html) menunjukkan bahwa dari seluruh kematian akibat penyakit kardiovaskuler, 7,4 juta (42,3%) diantaranya disebabkan oleh Penyakit Jantung Koroner (PJK). Berdasarkan data dari [Badan Penyelenggara Jaminan Sosial (BPJS)](https://www.bpjs-kesehatan.go.id/bpjs/) pembiayaan penyakit katastropik menghabiskan biaya hampir 14,6 Triliun Rupiah pada 2016 yang mana terjadi peningkatan pembiayaan dibanding tahun 2015, yakni sebesar 6,9 Triliun Rupiah (48,25%) menjadi 7,4 Triliun Rupiah (50,7%) pada 2016 untuk penyakit jantung.

Pada proyek ini, data yang saya pakai adalah data prediksi penyakit jantung dengan 11 fitur klinis sebagai parameter prediksi. Dari data ini, nantinya kita akan dapat mengklasifikasikan apakah seseorang terkena penyakit jantung atau tidak. Dataset ini memiliki 11 fitur yaitu umur, jenis kelamin, jenis nyeri dada, tekanan darah, kolesterol, _fasting blood sugar_, _resting ECG_, denyut jantung maksimal, angina latihan, oldpeak, dan ST Slope. Serta memiliki 1 target yaitu penyakit jantung.

Orang dengan penyakit kardiovaskular atau yang berada pada risiko kardiovaskular tinggi (karena adanya satu atau lebih faktor risiko seperti hipertensi, diabetes, hiperlipidemia atau penyakit yang sudah ada) memerlukan deteksi dan manajemen dini. Sehingga proyek ini bertujuan untuk mengklasifikasikan apakah seseorang terkena penyakit jantung atau tidak dengan 11 fitur klinis yang dijadikan sebagai parameter penentu.

## Business Understanding

### Problem Statements
Berdasarkan latar belakang yang telah diuraikan, berikut adalah rincian masalah yang dapat diselesaikan pada proyek ini :
-   Bagaimana cara membuat model _Machine Learning_ untuk mengklasifikasikan pasien dengan penyakit jantung?
-   Pada rentang usia berapa kebanyakan pasien terkena penyakit jantung?

### Goals

Berikut adalah tujuan proyek ini :
-   Membuat model _Machine Learning_ untuk mengklasifikasikan penyakit jantung pada pasien yang memiliki tingkat akurasi >75%.
-   Membuat grafik yang menunjukkan rentang usia pasien yang terkena penyakit jantung dan yang tidak terkena penyakit jantung.

### Solution statements
Untuk menyelesaikan masalah ini, saya akan mengajukan 4 solusi model _Machine Learning_ yang sederhana karena data ini merupakan data klasifikasi non-linear. Berikut adalah penjelasan model-model machine learning yang akan digunakan untuk masalah ini :
- **Naive Bayes** : Naive Bayes adalah sekumpulan algoritma klasifikasi dalam mesin learning yang menggunakan teori Bayes. Naive Bayes tidak hanya sebuah algoritma, namun sekumpulan algortima yang menggunakan asumsi yang sama yaitu nilai semua fitur independen terhadap fitur yang lain.
- **Random Forest** : Random forest (RF) adalah suatu algoritma yang digunakan pada klasifikasi data dalam jumlah yang besar. Klasifikasi random forest dilakukan melalui penggabungan pohon (tree) dengan melakukan training pada sampel data yang dimiliki. Penggunaan pohon (tree) yang semakin banyak akan mempengaruhi akurasi yang akan didapatkan menjadi lebih baik. Penentuan klasifikasi dengan random forest diambil berdasarkan hasil voting dari tree yang terbentuk. Pemenang dari tree yang terbentuk ditentukan dengan vote terbanyak.
- **XGBoosting** : XGBoosting merupakan salah satu model gradient boosting yang menggunakan beberapa model decision tree secara iteratif dan menghasilkan prediksi yang akurat. Model ini memiliki performa akurasi yang sangat baik dan waktu pelatihan yang cepat. Salah satu kelemahan yang dimiliki oleh XGBoost adalah kolom kategori harus diubah menjadi one-hot encoding sebelum dimasukan ke dalam model ini. Selain itu, nama kolom juga harus diperhatikan karena model ini tidak menerima nama kolom yang memiliki koma atau simbol lainnya.
- **Decision Tree** : Decision Tree merupakan representasi dari berbagai macam opsi. Sama halnya dengan membuat bagan opsi jikalau terjadi sesuatu maka opsi tersebut akan dipilih. Konsep dari decision tree adalah mengubah data menjadi aturan-aturan keputusan. Manfaat utama dari penggunaan decision tree adalah kemampuannya untuk mem-break down proses pengambilan keputusan yang kompleks menjadi lebih simple, sehingga pengambil keputusan akan lebih menginterpretasikan solusi dari permasalahan.
