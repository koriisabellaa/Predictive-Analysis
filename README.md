# Laporan Proyek Machine Learning - Kori Isabella Hutabarat

## Domain Proyek

Domain proyek yang saya pilih dalam proyek _Machine Learning_ ini adalah mengenai **kesehatan** dengan judul "Klasifikasi Penyakit Jantung".

Jantung merupakan pusat dari bagian tubuh manusia. Jantung berfungsi untuk memompa darah ke seluruh tubuh dan kembali lagi ke jantung melalui pembuluh darah. Jantung terletak pada rongga dada sebelah kiri di antara paru-paru kanan dan paru-paru kiri. Gangguan pada jantung dapat mengakibatkan terganggunya sistem peredaran darah pada tubuh manusia yang mana tentu berdampak sangat fatal bahkan menyebabkan kematian. Menurut [Dr. Tania Savitri](https://hellosehat.com/jantung/penyakit-jantung/pengertian-penyakit-jantung/),  penyakit jantung dan pembuluh darah atau penyakit kardiovaskular adalah berbagai kondisi di mana terjadi penyempitan atau penyumbatan pembuluh darah yang dapat menyebabkan serangan jantung, nyeri dada (angina), atau stroke. Penyakit kardiovaskular termasuk kondisi kritis yang butuh penanganan segera. Jika jantung bermasalah, peredaran darah dalam tubuh bisa terganggu. Tanpa pertolongan medis yang sesuai, penyakit kardiovaskular bisa mengancam jiwa dan menyebabkan kematian. Data [World Health Organization (WHO)](http://p2ptm.kemkes.go.id/uploads/VHcrbkVobjRzUDN3UCs4eUJ0dVBndz09/2018/09/Mengenali_tanda_dan_gejala_serangan_dini_penyakit_jantung_dr_Bambang_Dwiputra_Hari_Jantung_Sedunia_2018.pdf) menunjukkan 70% kematian di dunia disebabkan oleh Penyakit Tidak Menular, 45% diantaranya disebabkan oleh penyakit jantung dan pembuluh darah, yaitu 17.7 juta dari 39,5 juta kematian. Data dari [Kementerian Kesehatan](https://kemkes.go.id/article/view/17073100005/penyakit-jantung-penyebab-kematian-tertinggi-kemenkes-ingatkan-cerdik-.html) menunjukkan bahwa dari seluruh kematian akibat penyakit kardiovaskular, 7,4 juta (42,3%) diantaranya disebabkan oleh Penyakit Jantung Koroner (PJK). Berdasarkan data dari [Badan Penyelenggara Jaminan Sosial (BPJS)](https://www.bpjs-kesehatan.go.id/bpjs/) pembiayaan penyakit katastropik menghabiskan biaya hampir 14,6 Triliun Rupiah pada 2016 yang mana terjadi peningkatan pembiayaan dibanding tahun 2015, yakni sebesar 6,9 Triliun Rupiah (48,25%) menjadi 7,4 Triliun Rupiah (50,7%) pada 2016 untuk penyakit jantung.

Pada proyek ini, data yang saya pakai adalah data prediksi penyakit jantung dengan sebelas fitur klinis sebagai parameter prediksi. Dari data ini, nantinya kita akan dapat mengklasifikasikan apakah seseorang terkena penyakit jantung atau tidak. Dataset ini memiliki sebelas fitur yaitu umur, jenis kelamin, jenis nyeri dada, tekanan darah, kolesterol, _fasting blood sugar_, _resting ECG_, denyut jantung maksimal, nyeri dada saat melakukan pergerakan/latihan, oldpeak, dan ST Slope. Serta memiliki satu target yaitu penyakit jantung.

Orang dengan penyakit kardiovaskular atau yang berada pada risiko kardiovaskular tinggi (karena adanya satu atau lebih faktor risiko seperti hipertensi, diabetes, hiperlipidemia atau penyakit yang sudah ada) memerlukan deteksi dan manajemen dini. Sehingga, sangat dibutuhkan suatu pembelajaran mesin untuk dapat mengenali gejala penyakit ini.


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

## Data Understanding

![Web capture_14-10-2021_05246_www kaggle com](https://user-images.githubusercontent.com/87566521/137187045-86202755-ef2a-4aaf-99ab-807abdc51f57.jpeg)

Berikut adalah informasi dataset yang digunakan pada proyek ini :

| Jenis                   | Keterangan                                                                                               |
| ----------------------- | ---------------------------------------------------------------------------------------------------------|
| Sumber                  | [Kaggle Dataset : Heart Failure Prediction](https://www.kaggle.com/fedesoriano/heart-failure-prediction) |
| Lisensi                 | Database : Open Database, Contents : Original Authors                                                    |
| Kategori                | Kesehatan                                                                                                |
| Rating Penggunaan       | 10.0 (Silver)                                                                                            |
| Jenis dan Ukuran Berkas | CSV (36 kb)                                                                                              |

Dataset ini memiliki dua belas fitur dengan sebelas fitur klinis dan satu target. Berikut penjelasannya :

1. `Age` Merupakan usia pada pasien (Years)
2. `Sex` Merupakan jenis kelamin pasien (M: Male, F: Female)
3. `ChestPainType` Merupakan tipe nyeri dada (TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic)
4. `RestingBP` Tekanan darah pada pasien (mm Hg)
5. `Cholesterol` Kolestrol (mm/dl)
6. `FastingBS` Puasa gula darah (1: jika puasa > 120 mg/dl, 0: jika tidak puasa)
7. `RestingECG` Hasil elektrokardiogram (Normal: Normal, ST: memiliki kelainan gelombang ST-T , LVH: menunjukkan kemungkinan atau pasti hipertrofi ventrikel kiri menurut kriteria Estes)
8. `MaxHR` Denyut jantung maksimum yang tercapai (Nilai numerik antara 60 dan 202)
9. `ExerciseAngina` Angina yang diinduksi dari olahraga (Y: Ya, N: Tidak)
10. `Oldpeak` Oldpeak = ST (Nilai numerik diukur dalam depresi)
11. `ST_Slope` Kemiringan puncak latihan segmen ST (Atas: menanjak, Datar: datar, Bawah: menurun)
12. `HeartDisease` Output klasifikasi (1: penyakit jantung, 0: Normal)

Pada proyek ini, terdapat juga visualisasi data untuk memudahkan pembaca mendapatkan informasi. Visualisasi data yang diberi dalam projek ini adalah countplot dari library seaborn. Berikut adalah beberapa visualisasi data dan penjelasannya :

![image](https://user-images.githubusercontent.com/87566521/137190997-f8cd5b25-171b-4100-bbae-a746a59bf461.png)

Dari grafik di atas dapat disimpulkan bahwa lebih banyak pria yang menderita penyakit jantung daripada wanita.

![image](https://user-images.githubusercontent.com/87566521/137191260-7b182b62-8371-4036-b531-8482b940d009.png)

Dari grafik di atas dapat disimpulkan bahwa sebagian besar penderita penyakit jantung adalah pasien dengan nyeri dada tipe Asimtomatik (ASY), dan tanpa penyakit jantung adalah nyeri dada tipe Atipikal Angina (ATA).

![image](https://user-images.githubusercontent.com/87566521/137191460-2c8d74d1-db81-49be-83e8-2bcf1fec4647.png)

Hal menarik yang dapat disimpulkan dari grafik di atas adalah mereka yang memiliki penyakit jantung dan mereka yang tidak memiliki penyakit jantung sebagian besar adalah tipe  Resting ECG Normal.

![image](https://user-images.githubusercontent.com/87566521/137191644-298522bb-1cae-4850-8314-7371f482792c.png)

Dari grafik di atas dapat disimpulkan bahwa mayoritas penderita penyakit jantung adalah mereka yang memiliki Exercise-Induced Angina (AP).

![image](https://user-images.githubusercontent.com/87566521/137191770-17b7a41b-d694-42a0-ad17-7db37188ca4f.png)

Grafik di atas menunjukkan bahwa pria dengan penyakit jantung berusia sekitar 55-60 tahun, dan wanita berusia 57-60 tahun. Sedangkan pria tanpa penyakit jantung berusia sekitar 40-54 tahun, sedangkan wanita berusia sekitar 50 tahun.

## Data Preparation
Pada data preparation, teknik yang saya gunakan adalah one-hot encoding. One-Hot Encoding adalah teknik yang mengubah setiap nilai di dalam kolom menjadi kolom baru dan mengisinya dengan nilai biner yaitu 0 dan 1. Dalam Python Pandas, kita bisa gunakan dummies values di Pandas dengan menggunakan fungsi get_dummies ()

![image](https://user-images.githubusercontent.com/87566521/137195120-c1f1dc41-a4dc-412c-aba2-d310697ee467.png)

Teknik encoding seperti ini sangat penting untuk categorical data sebelum data diproses dalam machine learning. Hal ini dikarenakan machine learning tidak bisa menerima bentuk inputan string dan hanya bisa dalam bentuk data numerik.

Saya juga menggunakan standarisasi pada model ini yaitu melakukan perubahan skala, dimana data yang dimiliki akan diubah sehingga memiliki rata rata = 0 (terpusat) dan standar deviasi = 1. Berikut adalah rumus matematikanya :

![image](https://user-images.githubusercontent.com/87566521/137196495-c5a1e8c9-81a4-4754-ba39-08768b6fa902.png)

Dari rumus ini, X adalah nilai data asli, μ adalah nilai rata rata dari data yang ada, dan σ adalah nilai standar deviasinya.

Selain itu, saya juga melakukan train-test-split yaitu membagi dataset menjadi data latih (train) dan data uji (test). Pada proyek ini, saya mengambil 20% dari total data sebagai data uji (test).


## Modeling
Setelah melakukan _Data Preparation_ , selanjutnya kita akan membuat model _Machine Learning_ . Pada sub-bab sebelumnya, telah disampaikan bahwa ada empat model _Machine Learning_ yang saya gunakan yaitu  **Naive Bayes**, **Random Forest**, **XGBoosting**, dan **Decision Tree**.

Dalam tahap ini, saya membuat suatu fungsi untuk melatih model yang sudah dibuat, lalu melakukan prediksi, dan membandingkan hasil model tersebut dengan menghitung berapa lama model ini dilatih dan bagaimana hasil akurasi dari model tersebut.

Dari hasil pelatihan, dapat kita lihat bahwa **Naive Bayes** merupakan model dengan akurasi tertinggi yaitu 86%.


## Evaluation

Pada bagian ini, saya menguji performa model dengan *classification report* , *confussion matrix* , dan *accuracy score* karena ini adalah masalah klasifikasi. Berikut penjelasannya :

1. `Classification Report` : Merupakan _performance evaluation metric_ dalam _Machine Learning_. Di dalam metrik ini, terdapat **accuracy, precision, recall, dan F1 score**. Berikut penjelasannya :
    - **Accuracy** : Merupakan hasil akurasi dari model yang telah dibuat. Cara menghitungnya adalah dengan menjumlah semua prediksi yang benar dan dibagi dengan total semua prediksi yang benar maupun yang salah.

    ![image](https://user-images.githubusercontent.com/87566521/137200365-463726ef-be79-413d-88cd-5e92aa94a382.png)

    - **Precision** : Dapat didefinisikan sebagai rasio positif benar dengan jumlah _true positive_ dan _false positives_.

    ![image](https://user-images.githubusercontent.com/87566521/137200863-ef2b3bf5-26aa-4e59-a158-5904c72c3774.png)

    - **Recall** : Digunakan untuk menghitung berapa data yang salah dalam melakukan prediksi.

    ![image](https://user-images.githubusercontent.com/87566521/137201245-6b67f220-8cb6-479a-bbaf-2cc60afb6275.png)

    - **F1 Score** : Merupakan _weighted harmonic mean_ dari _precision_ dan _recall_.

    ![image](https://user-images.githubusercontent.com/87566521/137202025-99cc5234-0b32-4eac-bd84-6be57b37d02b.png)

2. `Confussion Matrix` : Merupakan pengukuran performa untuk masalah klasifikasi _Machine Learning_ dengan empat tabel kombinasi berbeda dari nilai prediksi dan nilai aktual. Ada empat istilah yang merupakan representasi hasil proses klasifikasi pada confusion matrix yaitu True Positif, True Negatif, False Positif, dan False Negatif. Berikut penjelasannya :
    - TN / True Negative : kasusnya negatif dan diprediksi negatif
    - TP / True Positive : kasusnya positif dan diprediksi positif
    - FN / False Negative : kasusnya positif tetapi diprediksi negatif
    - FP / False Positive : kasusnya negatif tetapi diprediksi positif

3. `Accuracy Score` : Merupakan hasil dari penjumlahan prediksi yang benar lalu dibagikan dengan prediksi yang benar ditambah prediksi yang salah. Ini sama dengan _accuracy_ pada **Classification Report** namun perbedaannya adalah pada **Classification Report** akurasinya dibulatkan menjadi dua bilangan di belakang 0, yang mana hal ini dapat menyebabkan bias pada akurasi model.

    Untuk _code implementation_, kita hanya perlu import library **sklearn.metric** lalu menuliskan code **accuracy_score(actual, prediction)**.

## Conclusion

Model klasifikasi penyakit jantung telah selesai dibuat dan model ini dapat digunakan untuk mengklasifikasikan data yang sebenarnya. Berdasarkan model tersebut dapat diketahui bahwa algoritma dengan akurasi tertinggi adalah Algoritma Naive Bayes dan akurasi terendah adalah Algoritma Decision Tree. Namun, beberapa pengembangan lain masih dapat dilakukan untuk membuat model yang memiliki akurasi lebih tinggi.


**---Ini adalah bagian akhir laporan---**
