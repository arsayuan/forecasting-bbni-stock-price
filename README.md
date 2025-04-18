# Laporan Proyek Machine Learning - Muhammad Arsayuan Wijaya
## Domain Proyek
Fluktuasi harga saham merupakan salah satu indikator utama yang mencerminkan kondisi pasar modal dan ekonomi secara umum. Bagi investor maupun institusi keuangan seperti bank, memahami pergerakan harga saham menjadi sangat krusial untuk pengambilan keputusan yang tepat dalam investasi, manajemen risiko, hingga perencanaan portofolio keuangan.

Bank Negara Indonesia (BBNI), sebagai salah satu bank terbesar di Indonesia yang terdaftar di Bursa Efek Indonesia (BEI), memiliki pergerakan harga saham yang aktif dan menjadi perhatian banyak investor. Dengan meningkatnya volume transaksi serta dinamika pasar yang cepat, kebutuhan untuk melakukan prediksi harga saham menjadi sangat penting. Prediksi ini memungkinkan pelaku pasar untuk meminimalisasi risiko, memaksimalkan keuntungan, dan merespon kondisi pasar secara lebih proaktif.

Proyek ini bertujuan untuk membangun model prediksi harga saham BBNI selama 10 hari ke depan menggunakan pendekatan time series forecasting berbasis deep learning. Model yang dihasilkan diharapkan dapat membantu investor dan pihak bank untuk memperoleh wawasan lebih dalam tentang tren harga di masa mendatang.

## Business Understanding
### Problem Statememts
1. Bagaimana cara memprediksi harga saham BBNI untuk 10 hari ke depan secara akurat menggunakan data historis?
2. Algoritma time series apa yang paling sesuai untuk memodelkan data harga saham BBNI?

### Goals
1. Mengembangkan model machine learning yang mampu memprediksi harga saham BBNI 10 hari ke depan berbasis data historis.
2. Mengevaluasi performa model dengan metrik yang sesuai (seperti RMSE, MAE, dan MAPE) untuk mengetahui tingkat akurasi prediksi.

### Solution Statements
Untuk dapat meraih Goals, digunakan algoritma Long Short-Term Memory (LSTM) karena kemampuannya dalam memproses dan mengingat pola pada data sekuensial. Model dibangun berdasarkan harga penutupan harian (close price) saham BBNI. Proses pelatihan model dilakukan menggunakan data historis dari tahun 2010 hingga saat ini, April 2025. Model kemudian dievaluasi dengan menggunakan metrik Mean Squared Error (MSE), Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), dan Mean Absolute Percentage Error (MAPE).

## Data Understanding
Dataset yang digunakan adalah data historis harga saham Bank Negara Indonesia (BBNI.JK), yang diperoleh melalui [Yahoo Finance]. Data mencakup periode Januari 2010 hingga April 2025, dengan frekuensi harian. Data memiliki jumlah baris 3763 dan jumlah kolom 5. Setelah melalui proses pengecekan, kondisi data sangatlah baik. Tidak ditemukan missing value, duplikat outlier, dan sebagainya.

Pengambilan data dilakukan secara otomatis menggunakan library yfinance menggunakan Python di Google Colab. Berikut adalah contoh kode yang digunakan untuk mengunduh data:

```sh
import yfinance as yf

# Unduh data harga saham harian BBNI dari Yahoo Finance
data = yf.download('BBNI.JK', start='2010-01-01', end='2025-04-13')
```

### Deskripsi Variabel
Variabel-variabel pada dataset harga saham harian BBNI adalah sebagai berikut:

| Variabel | Deskripsi |
| ------ | ------ |
| Date | Tanggal transaksi saham (index time series) |
| Close | Harga penutupan saham pada hari tersebut |
| High | Harga tertinggi saham pada hari tersebut |
| Low | Harga terendah saham pada hari tersebut |
| Open | Harga pembukaan saham pada hari tersebut |
| Volume | Volume transaksi saham (jumlah saham yang diperdagangkan) |

### Visualisasi Data
![Tren Harga Sahan BBNI](https://github.com/user-attachments/assets/cb204741-11c0-414e-980a-9fa5448ecc1a)

Visualisasi tren harga saham BBNI menunjukkan fluktuasi yang konsisten dengan dinamika pasar, termasuk tren naik dalam beberapa tahun terakhir. Pola ini menjadi penting dalam membangun model prediktif karena LSTM memanfaatkan pola sekuensial dalam data.

## Data Preparation
Pada tahap ini, data harga penutupan saham (Close) dinormalisasi menggunakan MinMaxScaler dari sklearn.preprocessing agar berada dalam rentang 0 hingga 1. Normalisasi bertujuan untuk mempercepat proses training dan meningkatkan performa model LSTM.

```sh
from sklearn.preprocessing import MinMaxScaler

# Normalisasi data harga penutupan
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(close_prices)

scaled_data = pd.DataFrame(scaled_data, columns=close_prices.columns, index=close_prices.index)
```

Setelah data dinormalisasi, data disusun dalam bentuk sequence time series dengan window 60 hari sebagai input (X) dan hari ke-61 sebagai target prediksi (y). Format ini mengikuti pola umum time series forecasting.

```sh
# Membuat sequence time series dengan window size 60
X = []
y = []

for i in range(60, len(scaled_data)):
    X.append(scaled_data.iloc[i-60:i, 0])
    y.append(scaled_data.iloc[i, 0])

X = np.array(X)
y = np.array(y)
```

Dataset dibagi menjadi data latih dan data uji berdasarkan indeks waktu, kemudian input data di-reshape ke format 3 dimensi (samples, time steps, features) sesuai kebutuhan model LSTM.

```sh
# Reshape data agar sesuai dengan input model LSTM (samples, time steps, features)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))
```

## Modeling
Dalam proyek ini, digunakan model Long Short-Term Memory (LSTM), yaitu jenis Recurrent Neural Network (RNN) yang dirancang untuk mengatasi masalah long-term dependency pada data time series. LSTM efektif digunakan untuk memprediksi harga saham karena mampu menangkap pola historis yang kompleks dalam data urut waktu. Adapun model dibangun dengan arsitektur seperti pada kode berikut:

```sh
# Create Model
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(32))
model.add(Dense(1))

# Compile Model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train Model
model.fit(X, y, epochs=20, batch_size=32)
```

## Evaluation
Proyek ini menggunakan beberapa metrik yang sesuai untuk melakukan evaluasi. Adapun metrik yang digunakan sebagai berikut:
1. **Mean Squared Error (MSE)**
MSE mengukur rata-rata kuadrat dari selisih antara nilai aktual dan nilai prediksi. Nilai yang lebih kecil menunjukkan performa model yang lebih baik, tetapi sensitif terhadap outlier.

2. **Root Mean Squared Error (RMSE)**
RMSE adalah akar dari MSE dan memiliki satuan yang sama dengan target (dalam hal ini, harga saham dalam rupiah). Ini memberikan interpretasi yang lebih langsung terhadap seberapa besar kesalahan prediksi.

3. **Mean Absolute Error (MAE)**
MAE mengukur rata-rata dari selisih absolut antara nilai aktual dan prediksi. MAE lebih stabil terhadap outlier dibandingkan MSE.

4. **Mean Absolute Percentage Error (MAPE)**
MAPE mengukur rata-rata kesalahan dalam bentuk persentase relatif terhadap nilai aktual, sehingga memudahkan interpretasi bagi pembaca non-teknis.

### Hasil Evaluasi Model
| Metrik | Nilai | Keterangan |
| ------ | ------ | ------ |
| MSE | 18,230.25 | Rata-rata kuadrat selisih prediksi dan aktual cukup rendah untuk data harga saham harian |
| RMSE | 135.02 | Secara rata-rata, prediksi harga saham hanya meleset sekitar Rp135 dari nilai sebenarnya |
| MAE | 104.95 | Secara umum, selisih absolut prediksi terhadap nilai aktual sangat kecil, terutama jika dibandingkan dengan range harga saham BBNI yang bisa mencapai belasan ribu rupiah |
| MAPE | 2.35% | Model sangat akurat karena rata-rata kesalahan hanya sekitar 2.35% dari harga aktual |

### Keterkaitan dengan Business Understanding
**Apakah berhasil mencapai setiap goals yang diharapkan?**

Seluruh tujuan proyek ini dapat dikatakan telah tercapai. Model LSTM berhasil dikembangkan dan digunakan untuk memprediksi harga saham BBNI selama 10 hari ke depan berdasarkan data historis. Selain itu, performa model juga telah dievaluasi dengan menggunakan metrik yang tepat, yaitu MSE, MAE, RMSE, dan MAPE. Hasil evaluasi menunjukkan bahwa model memiliki akurasi yang sangat baik dan mampu memberikan prediksi yang dapat diandalkan untuk kebutuhan analisis pasar.

**Apakah setiap solution statement yang direncanakan berdampak?**

Penggunaan model LSTM sebagai solusi dalam proyek ini terbukti tepat dan memberikan dampak yang signifikan. LSTM memiliki keunggulan dalam mengenali pola jangka panjang pada data sekuensial seperti harga saham harian, sehingga sangat membantu dalam meningkatkan akurasi prediksi. Langkah-langkah seperti normalisasi data dan pembentukan sequence dengan window 60 hari juga berkontribusi besar dalam memberikan konteks historis yang cukup bagi model. Dengan hasil evaluasi yang sangat memuaskan, model ini dapat memberikan manfaat nyata bagi investor dan pihak bank dalam mengambil keputusan strategis, termasuk dalam hal alokasi aset dan perencanaan investasi jangka pendek.

### Interpretasi Hasil
Model yang digunakan telah menunjukkan performa yang sangat baik berdasarkan nilai-nilai metrik di atas. MAPE < 5% menandakan bahwa prediksi sangat dekat dengan nilai aktual—hal ini sangat penting dalam dunia saham, di mana selisih kecil bisa berdampak besar pada keputusan investasi.

Selain itu, akurasi yang tinggi ini juga berarti model dapat digunakan untuk membantu investor atau analis dalam memproyeksikan pergerakan harga jangka pendek, khususnya untuk saham BBNI.

[Yahoo Finance]: <https://finance.yahoo.com/quote/BBNI.JK/history/>



