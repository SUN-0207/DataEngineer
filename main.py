# Databricks notebook source
# Đọc dữ liệu từ các tệp CSV và tạo DataFrame
prescriptions_df = spark.read.csv("/FileStore/tables/PRESCRIPTIONS_random.csv", header=True, inferSchema=True)
icustays_df = spark.read.csv("/FileStore/tables/ICUSTAYS_random.csv", header=True, inferSchema=True)
admissions_df = spark.read.csv("/FileStore/tables/ADMISSIONS_random.csv", header=True, inferSchema=True)
patients_df = spark.read.csv("/FileStore/tables/PATIENTS_random.csv", header=True, inferSchema=True)
diagnoses_icd_df = spark.read.csv("/FileStore/tables/DIAGNOSES_ICD_random.csv", header=True, inferSchema=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### **1. PHÂN TÍCH TẦN SUẤT VÀ THỜI GIAN TRUNG BÌNH SỬ DỤNG THUỐC TRONG ICU**

# COMMAND ----------

# Chỉ lấy các đơn thuốc dùng trong ICU thông qua ID lần nhập viện trong ICUSTAYS
from pyspark.sql.functions import col
icu_prescriptions_df = prescriptions_df.alias("presc").join(
    icustays_df.alias("icu"), prescriptions_df["HADM_ID"] == icustays_df["HADM_ID"], "inner"
).select(
    col("presc.SUBJECT_ID").alias("SUBJECT_ID"),  
    col("presc.HADM_ID").alias("HADM_ID"),
    col("icu.ICUSTAY_ID").alias("ICUSTAY_ID"),
    col("presc.DRUG").alias("DRUG"),
    col("presc.STARTDATE").alias("STARTDATE"),
    col("presc.ENDDATE").alias("ENDDATE"),
    col("icu.INTIME").alias("INTIME"),
    col("icu.OUTTIME").alias("OUTTIME")
)

# COMMAND ----------

import pyspark.sql.functions as F

# Xóa các bản ghi có giá trị trống trong các cột 'STARTDATE' hoặc 'ENDDATE'
icu_prescriptions_df = icu_prescriptions_df.dropna(subset=['STARTDATE', 'ENDDATE'])

icu_prescriptions_df = icu_prescriptions_df.withColumn(
    "STARTDATE", F.to_timestamp("STARTDATE")
).withColumn(
    "ENDDATE", F.to_timestamp("ENDDATE")
)

# Calculate duration in days
icu_prescriptions_df = icu_prescriptions_df.withColumn(
    "DURATION", F.datediff("ENDDATE", "STARTDATE")
)

drug_usage_summary = icu_prescriptions_df.groupby('DRUG') \
    .agg(
        F.count('DRUG').alias('usage_count'),  # Đếm số lần thuốc được kê đơn
        F.floor(F.mean('DURATION')).alias('avg_duration')  # Tính thời gian sử dụng trung bình làm tròn đến đơn vị ngày
    )

# Hiển thị kết quả phân tích
drug_usage_summary.show()

# Lấy top 10 loại thuốc có tần suất sử dụng cao nhất
top_drugs = drug_usage_summary.sort("usage_count", ascending=False).limit(10)
top_drugs.show()

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt

# Chuyển đổi thành Pandas DataFrame
pandas_df = top_drugs.toPandas()

# Tạo figure và các axes
fig, ax1 = plt.subplots(figsize=(20,15))
ax2 = ax1.twinx()  # Tạo trục y thứ hai chia sẻ trục x

# Vẽ biểu đồ cột cho tần suất
ax1.bar(pandas_df['DRUG'], pandas_df['usage_count'], label='Tần suất sử dụng')
ax1.set_ylabel('Tần suất')

# Vẽ đường biểu diễn thời gian trung bình
ax2.plot(pandas_df['DRUG'],pandas_df['avg_duration'], color='red', marker='o', label='Thời gian trung bình')
ax2.set_ylabel('Thời gian (ngày)')

# Tùy chỉnh biểu đồ
plt.title('Tần suất và thời gian sử dụng trung bình của 10 loại thuốc được sử dụng nhiều nhất')
plt.xlabel('Tên thuốc')
plt.xticks(rotation=45)  # Xoay nhãn trục x
plt.legend(loc='upper left')

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### **2. GOM CỤM BỆNH NHÂN BẰNG GIẢI THUẬT K-MEANS**

# COMMAND ----------

# MAGIC %md
# MAGIC ### **Tính toán các đặc trưng sử dụng để gom cụm**
# MAGIC 1. Số lượng thuốc mỗi bệnh nhân sử dụng
# MAGIC 2. Thời gian bệnh nhân nằm ICU
# MAGIC 3. Số lần vào ICU
# MAGIC 4. Số lần thay đổi thuốc của bệnh nhân
# MAGIC 5. Số mã chẩn đoán

# COMMAND ----------

from pyspark.sql.window import Window

# 1. Tính số lượng thuốc mỗi bệnh nhân đã dùng
num_drugs_df = icu_prescriptions_df.groupBy("SUBJECT_ID") \
    .agg(F.countDistinct("DRUG").alias("num_drugs"))

# 2. Tính thời gian lưu trú trong ICU
icu_length_df = icu_prescriptions_df.withColumn(
    "length_of_stay",
    F.datediff(F.col("OUTTIME"), F.col("INTIME"))
).groupBy("SUBJECT_ID").agg(F.sum("length_of_stay").alias("length_of_stay"))

# Do thời gian lưu trú có vài giá trị ngoại lai lớn nên cần loại bỏ
# Tính IQR (Interquartile Range) để xác định ngoại lai
Q1 = icu_length_df.approxQuantile("length_of_stay", [0.25], 0)[0]
Q3 = icu_length_df.approxQuantile("length_of_stay", [0.75], 0)[0]
IQR = Q3 - Q1

# Loại bỏ ngoại lai (bệnh nhân có length_of_stay ngoài phạm vi 1.5 IQR)
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
icu_length_df = icu_length_df.filter((F.col("length_of_stay") >= lower_bound) & (F.col("length_of_stay") <= upper_bound))

# 3. Tính số lần vào ICU
num_icu_df = icustays_df.groupBy("SUBJECT_ID").agg(
    F.countDistinct("ICUSTAY_ID").alias("num_icu")  # Đếm số lần nhập viện ICU khác nhau
)

# 4. Tính số lần thay đổi thuốc của bệnh nhân
drugs_change_df = icu_prescriptions_df.withColumn(
    "drug_change", F.when(F.lag("DRUG").over(Window.partitionBy("SUBJECT_ID", "ICUSTAY_ID").orderBy("STARTDATE")) != F.col("DRUG"), 1).otherwise(0)
)
drugs_change_df = drugs_change_df.groupBy("SUBJECT_ID").agg(
    F.sum("drug_change").alias("drug_changes"))

# 5. Tính số lượng mã chẩn đoán từ DIAGNOSES_ICD
num_diagnoses_df = diagnoses_icd_df.groupBy("SUBJECT_ID").agg(
    F.countDistinct("ICD9_CODE").alias("num_diagnoses")
)

# Kết hợp các DataFrame trên thành một DataFrame duy nhất
features_df = num_drugs_df \
    .join(icu_length_df, "SUBJECT_ID", "inner") \
    .join(num_icu_df, "SUBJECT_ID", "inner")\
    .join(drugs_change_df, "SUBJECT_ID", "inner")\
    .join(num_diagnoses_df, "SUBJECT_ID", "inner")


# COMMAND ----------

count_distinct_subjects = features_df.select("SUBJECT_ID").distinct().count()

print("Số lượng bệnh nhân:", count_distinct_subjects)

# COMMAND ----------

# MAGIC %md
# MAGIC **Chuẩn hóa các đặc trưng và chuẩn bị dữ liệu cho mô hình**

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler

# Chuyển các đặc điểm thành cột feature vector
assembler = VectorAssembler(
    inputCols=["num_drugs", "length_of_stay", "num_icu", "drug_changes", "num_diagnoses"],
    outputCol="features"
)
data = assembler.transform(features_df)

# Bước 2: Chuẩn hóa các đặc điểm với StandardScaler
scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withMean=True, withStd=True)
scaler_model = scaler.fit(data)
scaled_data = scaler_model.transform(data)

# COMMAND ----------

# MAGIC %md
# MAGIC **Áp dụng K-Means để gom cụm**

# COMMAND ----------

from pyspark.ml.clustering import KMeans

# Khởi tạo mô hình K-means với số cụm là 3
kmeans = KMeans(featuresCol="scaled_features", k=3, seed=1)
model = kmeans.fit(scaled_data)

# Áp dụng mô hình K-means lên dữ liệu để phân cụm
clustered_data = model.transform(scaled_data)

# COMMAND ----------

# MAGIC %md
# MAGIC **Kết quả gom cụm**

# COMMAND ----------

# Thống kê số lượng bệnh nhân trong mỗi cụm
clustered_data.groupBy("prediction").count().show()

# Tính trung bình các đặc điểm cho từng cụm
cluster_summary = clustered_data.groupBy("prediction") \
    .agg(
        F.avg("num_drugs").alias("avg_num_drugs"),
        F.avg("length_of_stay").alias("avg_length_of_stay"),
        F.avg("num_icu").alias("avg_num_icu"),
        F.avg("drug_changes").alias("avg_drug_changes"),
        F.avg("num_diagnoses").alias("avg_num_diagnoses")
    )
cluster_summary.show()


# COMMAND ----------

# MAGIC %md
# MAGIC **Biểu đồ radar**

# COMMAND ----------

import numpy as np
from math import pi
from sklearn.preprocessing import StandardScaler

# Chuyển đổi thành Pandas DataFrame
cluster_summary_df = cluster_summary.toPandas()

# Các giá trị trung bình từ bảng cluster_summary_df
cluster_summary_values = cluster_summary_df.drop("prediction", axis=1).values

# Chuẩn hóa dữ liệu bằng StandardScaler
scaler = StandardScaler()
cluster_summary_values_scaled = scaler.fit_transform(cluster_summary_values)

# Tạo danh sách các đặc điểm cần vẽ
categories = ['avg_num_drugs', 'avg_length_of_stay', 'avg_num_icu', 'avg_drug_changes', 'avg_num_diagnoses']

# Số lượng đặc điểm
num_vars = len(categories)

# Tạo góc cho các đặc trưng trên radar chart
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]

# Vẽ radar chart cho từng cụm
for num in range(len(cluster_summary_values_scaled)):
    scaled_values = cluster_summary_values_scaled[num].tolist()
    scaled_values += scaled_values[:1]

    # Tạo radar chart
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, scaled_values, color='blue', alpha=0.25)
    ax.plot(angles, scaled_values, color='blue', linewidth=2)

    # Thiết lập nhãn cho trục x và đặt giới hạn trục y
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
    ax.set_ylim(-2, 2)  # Đặt giới hạn trục y

    # Ẩn các nhãn của trục y
    ax.set_yticklabels([])

    # Tiêu đề của biểu đồ
    plt.title(f'Cluster {cluster_summary_df.iloc[num]["prediction"]} Radar Chart', size=14, color='blue', fontweight='bold')
    plt.show()

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, datediff, to_date, year

# Khởi tạo Spark session (nếu chưa khởi tạo)
spark = SparkSession.builder.appName("ICU Analysis").getOrCreate()

# Đọc dữ liệu từ bảng PRESCRIPTIONS và PATIENTS
prescriptions_df = spark.read.csv("/FileStore/tables/PRESCRIPTIONS_random.csv", header=True, inferSchema=True)
icustays_df = spark.read.csv("/FileStore/tables/ICUSTAYS_random.csv", header=True, inferSchema=True)
patients_df = spark.read.csv("/FileStore/tables/PATIENTS_random.csv", header=True, inferSchema=True)
admissions_df = spark.read.csv("/FileStore/tables/ADMISSIONS_random.csv", header=True, inferSchema=True)


# Kiểm tra schema
prescriptions_df.printSchema()
icustays_df.printSchema()
patients_df.printSchema()
admissions_df.printSchema()

# COMMAND ----------

prescriptions_df = prescriptions_df.dropna()
icustays_df = icustays_df.dropna()
patients_df = patients_df.dropna()
admissions_df = admissions_df.dropna()

# COMMAND ----------

prescriptions_df = prescriptions_df.withColumn("STARTDATE", to_date(col("STARTDATE")))
prescriptions_df = prescriptions_df.withColumn("ENDDATE", to_date(col("ENDDATE")))
prescriptions_df = prescriptions_df.filter(col("ENDDATE") >= col("STARTDATE"))

# COMMAND ----------

# MAGIC %md
# MAGIC # **3. Phân tích thời gian và tần suất kê đơn thuốc từ Prescriptions**

# COMMAND ----------



# COMMAND ----------

prescriptions_with_icustays = prescriptions_df.join(
    icustays_df, on="HADM_ID", how="inner"
).select(
    col("DRUG"),
    col("ENDDATE"),  
    col("STARTDATE")
)

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

from pyspark.sql import functions as F

prescriptions_with_days = prescriptions_with_icustays.withColumn(
    "DAYS_PRESCRIBED", F.datediff(F.col("ENDDATE"), F.col("STARTDATE"))
)

# COMMAND ----------

prescriptions_days = prescriptions_with_days.select("DRUG", "DAYS_PRESCRIBED")

# Chuyển sang Pandas DataFrame để vẽ biểu đồ
prescriptions_days_pd = prescriptions_days.toPandas()

# COMMAND ----------

drug_counts = prescriptions_days_pd['DRUG'].value_counts()

# Lấy tên thuốc có tần suất xuất hiện nhiều nhất
most_frequent_drug = drug_counts.idxmax()

print(f"Thuốc có tần suất xuất hiện nhiều nhất: {most_frequent_drug}")

# COMMAND ----------

most_frequent_drug_data = prescriptions_days_pd[prescriptions_days_pd['DRUG'] == most_frequent_drug]

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

# Vẽ boxplot cho loại thuốc có tần suất xuất hiện nhiều nhất
plt.figure(figsize=(10, 6))
most_frequent_drug_data.boxplot(column='DAYS_PRESCRIBED', grid=False)
plt.title(f'Phân phối thời gian kê đơn cho thuốc: {most_frequent_drug}')
plt.xlabel('Thời gian kê đơn (ngày)')
plt.show()


# COMMAND ----------

top_10_most_frequent_drugs = prescriptions_days_pd['DRUG'].value_counts().head(10).index

# Lọc dữ liệu chỉ lấy các loại thuốc có tần suất cao nhất
top_10_drugs_data = prescriptions_days_pd[prescriptions_days_pd['DRUG'].isin(top_10_most_frequent_drugs)]

# Vẽ boxplot cho các loại thuốc có tần suất xuất hiện nhiều nhất
plt.figure(figsize=(12, 8))
top_10_drugs_data.boxplot(column='DAYS_PRESCRIBED', by='DRUG', vert=False, patch_artist=True, grid=False)
plt.title('Phân phối thời gian kê đơn cho 10 loại thuốc phổ biến nhất')
plt.suptitle('')
plt.xlabel('Thời gian kê đơn (ngày)')
plt.xticks(rotation=90)
plt.show()

# COMMAND ----------

from pyspark.sql import functions as F

# Tính toán thời gian kê đơn ngắn nhất, dài nhất và trung bình của từng loại thuốc
prescription_stats = prescriptions_with_days.groupBy("DRUG").agg(
    F.min("DAYS_PRESCRIBED").alias("MIN_DAYS_PRESCRIBED"),
    F.max("DAYS_PRESCRIBED").alias("MAX_DAYS_PRESCRIBED"),
    F.avg("DAYS_PRESCRIBED").alias("AVG_DAYS_PRESCRIBED")
)

# Hiển thị kết quả
prescription_stats.show()

# COMMAND ----------

prescription_stats_pd = prescription_stats.toPandas()

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 8))

# Vẽ cột cho MIN, MAX, và AVG (10 loại thuốc)
# prescription_stats_pd[prescription_stats_pd['DRUG'].isin(top_10_most_frequent_drugs)].set_index('DRUG')[['MIN_DAYS_PRESCRIBED', 'MAX_DAYS_PRESCRIBED', 'AVG_DAYS_PRESCRIBED']].plot(kind='bar', width=0.8, figsize=(12, 8))
prescription_stats_pd[:10].set_index('DRUG')[['MIN_DAYS_PRESCRIBED', 'MAX_DAYS_PRESCRIBED', 'AVG_DAYS_PRESCRIBED']].plot(kind='bar', width=0.8, figsize=(12, 8))
plt.title('Thời gian kê đơn ngắn nhất, dài nhất và trung bình của từng loại thuốc')
plt.ylabel('Số ngày')
plt.xlabel('Loại thuốc')
plt.xticks(rotation=90)  # Xoay nhãn trục X nếu cần
plt.legend(title="Thống kê", loc="upper left")
plt.tight_layout()
plt.show()


# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

top_4_drugs = prescriptions_days_pd['DRUG'].value_counts().head(4).index.tolist()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))  # 2 rows, 2 columns

# Flatten axes array để dễ dàng truy cập
axes = axes.flatten()

# Lặp qua 4 loại thuốc và vẽ từng distribution plot vào các subplot tương ứng
for i, drug in enumerate(top_4_drugs):
    drug_data = prescriptions_days_pd[prescriptions_days_pd['DRUG'] == drug]
    
    sns.histplot(drug_data['DAYS_PRESCRIBED'], kde=True, bins=20, color='skyblue', stat='density', ax=axes[i])
    axes[i].set_title(f'Phân bố thời gian kê đơn cho {drug}')
    axes[i].set_xlabel('Thời gian kê đơn (ngày)')
    axes[i].set_ylabel('Mật độ')

# Điều chỉnh khoảng cách giữa các subplots
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # **4. Phân tích độ tuổi và giới tính của bệnh nhân sử dụng thuốc trong ICU**

# COMMAND ----------

from pyspark.sql import functions as F

# Kết hợp PRESCRIPTIONS với ICUSTAYS để lọc các thuốc kê đơn trong ICU
prescriptions_with_demographics = prescriptions_df \
    .join(admissions_df, prescriptions_df["HADM_ID"] == admissions_df["HADM_ID"], "inner") \
    .join(patients_df, admissions_df["SUBJECT_ID"] == patients_df["SUBJECT_ID"], "inner") \
    # .join(icustays_df, prescriptions_df["HADM_ID"] == icustays_df["HADM_ID"], "inner") \

# Tính độ tuổi khi nhập viện
prescriptions_with_demographics = prescriptions_with_demographics.withColumn(
    "AGE", F.year(admissions_df["ADMITTIME"]) - F.year(patients_df["DOB"])
)

# Loại bỏ những dòng có độ tuổi > 100
prescriptions_with_demographics = prescriptions_with_demographics.filter(F.col("AGE") <= 100)

# COMMAND ----------

prescriptions_with_demographics_pd = prescriptions_with_demographics.toPandas()

# COMMAND ----------

top_10_drugs = (
    prescriptions_with_demographics_pd['DRUG']
    .value_counts()
    .head(10)
    .index.tolist()
)

# Lọc dữ liệu cho top 10 loại thuốc
top_10_data = prescriptions_with_demographics_pd[prescriptions_with_demographics_pd['DRUG'].isin(top_10_drugs)]

# COMMAND ----------

import seaborn as sns
import matplotlib.pyplot as plt

# Vẽ Violin Plot từ dữ liệu age_gender_stats_pd
plt.figure(figsize=(12, 6))
sns.violinplot(
    data=top_10_data,
    x='DRUG', y='AGE', hue='GENDER', split=True, palette='coolwarm'
)
plt.title('Phân bố độ tuổi sử dụng thuốc theo giới tính')
plt.xlabel('Loại thuốc')
plt.ylabel('Độ tuổi')
plt.xticks(rotation=45)
plt.legend(title='Giới tính', loc='upper right')
plt.show()

# COMMAND ----------

# Tính toán độ tuổi nhỏ nhất, lớn nhất, trung bình cho mỗi loại thuốc, phân chia theo giới tính
age_gender_stats = prescriptions_with_demographics.groupBy("DRUG", "GENDER") \
    .agg(
        F.min("AGE").alias("MIN_AGE"),
        F.max("AGE").alias("MAX_AGE"),
        F.avg("AGE").alias("AVG_AGE")
    )

# Hiển thị kết quả
age_gender_stats.show()

# COMMAND ----------

age_gender_stats_pd = age_gender_stats.toPandas()

lactulose_rows = age_gender_stats_pd[age_gender_stats_pd['DRUG'] == 'Insulin']
# Hiển thị kết quả
lactulose_rows

# COMMAND ----------

# Lọc 10 loại thuốc phổ biến nhất từ age_gender_stats_pd
top_10_drugs = (
    age_gender_stats_pd['DRUG']
    .value_counts()
    .head(10)
    .index.tolist()
)

# Lọc dữ liệu cho top 10 loại thuốc
top_10_data = age_gender_stats_pd[age_gender_stats_pd['DRUG'].isin(top_10_drugs)]
top_10_data


# COMMAND ----------

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
sns.scatterplot(
    data=top_10_data,
    x='DRUG',
    y='AVG_AGE',
    hue='GENDER',
    palette='coolwarm',
    style='GENDER',
    s=100,
    alpha=1
)
plt.title('Phân bố độ tuổi trung bình theo giới tính cho 10 loại thuốc phổ biến nhất')
plt.xlabel('Loại thuốc')
plt.ylabel('Độ tuổi trung bình')
plt.xticks(rotation=45)
plt.legend(title='Giới tính', loc='upper right')
plt.tight_layout()
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC # 4. Phân tích mối quan hệ giữa loại thuốc và tỷ lệ tử vong

# COMMAND ----------

# Đọc dữ liệu từ các tệp CSV và tạo DataFrame
prescriptions_df = spark.read.csv("/FileStore/tables/PRESCRIPTIONS_random.csv", header=True, inferSchema=True)
icustays_df = spark.read.csv("/FileStore/tables/ICUSTAYS_random.csv", header=True, inferSchema=True)
admissions_df = spark.read.csv("/FileStore/tables/ADMISSIONS_random.csv", header=True, inferSchema=True)
diagnoses_df = spark.read.csv("/FileStore/tables/DIAGNOSES_ICD_random.csv", header=True, inferSchema=True)
patients_df = spark.read.csv("/FileStore/tables/PATIENTS_random.csv", header=True, inferSchema=True)
d_icd_diagnoses_df = spark.read.csv("/FileStore/tables/D_ICD_DIAGNOSES.csv", header=True, inferSchema=True) 


# COMMAND ----------

#Tính loại thuốc thường được sử dụng dựa trên loại bệnh
# Import cần thiết
from pyspark.sql import functions as F

# Kết hợp ba DataFrames PRESCRIPTIONS, DIAGNOSES_ICD và ICUSTAYS dựa trên HADM_ID
combined_df = prescriptions_df \
    .join(diagnoses_df, "HADM_ID", "inner") \
    .join(icustays_df, "HADM_ID", "inner")

# Nhóm dữ liệu theo ICD9_CODE và DRUG, và tính số lượng thuốc đã sử dụng
drug_count_df = combined_df.groupBy("ICD9_CODE", "DRUG") \
    .agg(F.count("DRUG").alias("DRUG_COUNT"))

# Sắp xếp theo DRUG_COUNT giảm dần
drug_count_df_sorted = drug_count_df.orderBy("DRUG_COUNT", ascending=False)


# map ICD9_CODE vào tên bệnh
mapped_df = combined_df \
    .join(d_icd_diagnoses_df, "ICD9_CODE", "left")  # Thực hiện join với bảng d_icd_diagnoses_df

#  Nhóm theo ICD9_CODE và DRUG, tính số lượng thuốc và lấy SHORT_TITLE từ bảng d_icd_diagnoses_df
drug_count_with_title_df = mapped_df.groupBy("ICD9_CODE", "DRUG", "SHORT_TITLE") \
    .agg(F.count("DRUG").alias("DRUG_COUNT"))

# Sắp xếp theo số lượng thuốc giảm dần
drug_count_with_title_sorted = drug_count_with_title_df.orderBy("DRUG_COUNT", ascending=False)

#Hiển thị kết quả
drug_count_with_title_sorted.show()
# sắp xếp theo số lượng thuốc dựa trên loại bệnh
drug_count_pd = drug_count_with_title_sorted.limit(20).toPandas()  

# Import seaborn để vẽ biểu đồ
import seaborn as sns

# Bước 2: Tạo biểu đồ cột với tên bệnh và số lượng thuốc
import matplotlib.pyplot as plt
plt.figure(figsize=(14, 10))
sns.barplot(data=drug_count_pd, x="DRUG_COUNT", y="SHORT_TITLE", hue="DRUG", dodge=False, palette="viridis")

# Thêm nhãn và tiêu đề
plt.xlabel("Drug Count")
plt.ylabel("Disease (Short Title)")
plt.title("Most Commonly Used Drugs for Different Diseases")
plt.legend(title="Drug", bbox_to_anchor=(1.05, 1), loc="upper left")  # Di chuyển chú thích ra ngoài biểu đồ
plt.show()

# COMMAND ----------

# Tính tỷ lệ tử vong cho từng loại thuốc (DRUG)
drug_mortality_df = combined_df.groupBy("DRUG") \
    .agg(
        F.count("DRUG").alias("TOTAL_PATIENTS"),
        F.sum("HOSPITAL_EXPIRE_FLAG").alias("TOTAL_DEATHS")
    ) \
    .withColumn("MORTALITY_RATE", F.col("TOTAL_DEATHS") / F.col("TOTAL_PATIENTS"))

# Chuyển dữ liệu sang Pandas để vẽ biểu đồ
drug_mortality_pd = drug_mortality_df.limit(20).toPandas()
# Vẽ Bubble Chart cho tỷ lệ tử vong theo loại thuốc
plt.figure(figsize=(12, 8))
plt.scatter(
    drug_mortality_pd["DRUG"], 
    drug_mortality_pd["MORTALITY_RATE"], 
    s=drug_mortality_pd["TOTAL_PATIENTS"]*10,  # điều chỉnh kích thước bubble
    alpha=0.3,
    c=drug_mortality_pd["MORTALITY_RATE"], cmap='coolwarm'
)

plt.colorbar(label='Mortality Rate')
plt.xlabel("Drug")
plt.ylabel("Mortality Rate")
plt.title("Relationship Between Drug Usage and Mortality Rate")
plt.xticks(rotation=90)
plt.show()
## Lọc sang biểu đồ cột 
drug_mortality_pd = drug_mortality_df.limit(20).toPandas()

# Lọc 20 loại thuốc có tỷ lệ tử vong cao nhất
top_20_drugs = drug_mortality_pd.sort_values(by="MORTALITY_RATE", ascending=False).head(100)

#Vẽ biểu đồ cột
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 8))
plt.barh(top_100_drugs["DRUG"], top_20_drugs["MORTALITY_RATE"], color="tomato")
plt.xlabel("Mortality Rate")
plt.ylabel("Drug")
plt.title("Top 20 Drugs with Highest Mortality Rate")
plt.gca().invert_yaxis()  # Đảo ngược trục Y để loại có tỉ lệ tử vong cao nhất ở trên cùng
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # 5. Phân tích mối quan hệ giữa loại thuốc và tỷ lệ tử vong

# COMMAND ----------

# Chỉ lấy các đơn thuốc dùng trong ICU thông qua ID lần nhập viện trong ICUSTAYS
from pyspark.sql.functions import col
icu_prescriptions_df = prescriptions_df.alias("presc").join(
    icustays_df.alias("icu"), prescriptions_df["HADM_ID"] == icustays_df["HADM_ID"], "inner"
).select(
    col("presc.SUBJECT_ID").alias("SUBJECT_ID"),  
    col("presc.HADM_ID").alias("HADM_ID"),
    col("icu.ICUSTAY_ID").alias("ICUSTAY_ID"),
    col("presc.DRUG").alias("DRUG"),
    col("presc.DOSE_VAL_RX").alias("DOSE_VAL_RX"),
    col("presc.STARTDATE").alias("STARTDATE"),
    col("presc.ROUTE").alias("ROUTE")
)

# COMMAND ----------

# Lấy dữ liệu về tình trạng tử vong từ bảng ADMISSIONS
mortality_df = admissions_df.filter(admissions_df.HOSPITAL_EXPIRE_FLAG == 1) \
                            .select("SUBJECT_ID", "HADM_ID", "HOSPITAL_EXPIRE_FLAG")

# Kết hợp với bảng đơn thuốc trong ICU
icu_mortality_drug_df = icu_prescriptions_df.join(mortality_df, ["SUBJECT_ID", "HADM_ID"], "inner")

# COMMAND ----------

from pyspark.sql import functions as F

# Đếm số bệnh nhân tử vong cho từng loại thuốc trong ICU
death_count_df = icu_mortality_drug_df.groupBy("DRUG") \
    .agg(
        F.countDistinct(icu_mortality_drug_df["SUBJECT_ID"]).alias("death_count")
    )

# Đếm tổng số bệnh nhân sử dụng mỗi loại thuốc trong ICU
total_patient_count_df = icu_prescriptions_df.groupBy("DRUG") \
    .agg(
        F.countDistinct(icu_prescriptions_df["SUBJECT_ID"]).alias("total_patient_count")
    )

# Kết hợp dữ liệu tử vong và tổng số bệnh nhân sử dụng thuốc để tính tỷ lệ tử vong
drug_mortality_rate_df = death_count_df.join(total_patient_count_df, "DRUG", "inner") \
    .withColumn("mortality_rate", F.col("death_count") / F.col("total_patient_count"))

# COMMAND ----------

from pyspark.sql import functions as F

# Đếm số bệnh nhân tử vong cho từng loại thuốc trong ICU
death_count_df = icu_mortality_drug_df.groupBy("DRUG") \
    .agg(
        F.countDistinct(icu_mortality_drug_df["SUBJECT_ID"]).alias("death_count")
    )

# Đếm tổng số bệnh nhân sử dụng mỗi loại thuốc trong ICU
total_patient_count_df = icu_prescriptions_df.groupBy("DRUG") \
    .agg(
        F.countDistinct(icu_prescriptions_df["SUBJECT_ID"]).alias("total_patient_count")
    )

# Kết hợp dữ liệu tử vong và tổng số bệnh nhân sử dụng thuốc để tính tỷ lệ tử vong
drug_mortality_rate_df = death_count_df.join(total_patient_count_df, "DRUG", "inner") \
    .withColumn("mortality_rate", F.col("death_count") / F.col("total_patient_count"))

# COMMAND ----------

from pyspark.sql.window import Window
from pyspark.sql.functions import lag
# Sắp xếp đơn thuốc của mỗi bệnh nhân theo thời gian STARTDATE
prescriptions_sorted_df = icu_prescriptions_df.orderBy("SUBJECT_ID", "ICUSTAY_ID", "STARTDATE")

# Tạo cột xác định sự thay đổi thuốc (so sánh thuốc giữa các đơn thuốc của cùng bệnh nhân)
prescriptions_with_change_df = prescriptions_sorted_df.withColumn(
    "drug_change", F.when(F.lag("DRUG").over(Window.partitionBy("SUBJECT_ID", "ICUSTAY_ID").orderBy("STARTDATE")) != F.col("DRUG"), 1).otherwise(0)
)

# Tạo cột xác định sự thay đổi liều dùng
prescriptions_with_change_df = prescriptions_with_change_df.withColumn(
    "dose_change", F.when(F.lag("DOSE_VAL_RX").over(Window.partitionBy("SUBJECT_ID", "ICUSTAY_ID").orderBy("STARTDATE")) != F.col("DOSE_VAL_RX"), 1).otherwise(0)
)

# Tạo cột xác định sự thay đổi đường dùng
prescriptions_with_change_df = prescriptions_with_change_df.withColumn(
    "route_change", F.when(F.lag("ROUTE").over(Window.partitionBy("SUBJECT_ID", "ICUSTAY_ID").orderBy("STARTDATE")) != F.col("ROUTE"), 1).otherwise(0)
)

# COMMAND ----------

# Tính tổng số lần thay đổi thuốc, liều, và đường dùng trên mỗi bệnh nhân
change_summary_df = prescriptions_with_change_df.groupBy("SUBJECT_ID").agg(
    F.sum("drug_change").alias("drug_change_count"),
    F.sum("dose_change").alias("dose_change_count"),
    F.sum("route_change").alias("route_change_count")
)

# Tính số lần thay đổi trên toàn bộ bệnh nhân
total_changes_df = change_summary_df.select(
    F.sum("drug_change_count").alias("total_drug_changes"),
    F.sum("dose_change_count").alias("total_dose_changes"),
    F.sum("route_change_count").alias("total_route_changes"),
    F.countDistinct("SUBJECT_ID").alias("total_patients")
)

# COMMAND ----------

# Tính trung bình số lần thay đổi trên mỗi bệnh nhân
total_changes = total_changes_df.collect()[0]
total_patients = total_changes["total_patients"]
total_changes_count = total_changes["total_drug_changes"] + total_changes["total_dose_changes"] + total_changes["total_route_changes"]
avg_changes_per_patient = total_changes_count / total_patients

# Tính tỷ lệ thay đổi cho từng yếu tố
drug_change_ratio = total_changes["total_drug_changes"] / total_changes_count * 100
dose_change_ratio = total_changes["total_dose_changes"] / total_changes_count * 100
route_change_ratio = total_changes["total_route_changes"] / total_changes_count * 100

print(f"Trong số {total_patients} bệnh nhân ICU, có tổng cộng {total_changes_count} lần thay đổi thuốc, với trung bình {avg_changes_per_patient:.2f} lần thay đổi trên mỗi bệnh nhân.")
print(f"Các lý do thay đổi phổ biến bao gồm:")
print(f"- Thay đổi thuốc: {drug_change_ratio:.2f}%")
print(f"- Thay đổi liều dùng: {dose_change_ratio:.2f}%")
print(f"- Thay đổi đường dùng: {route_change_ratio:.2f}%")

# COMMAND ----------

# Nhóm theo loại thuốc và tính tổng số lần thay đổi thuốc
top_drugs_changes_df = prescriptions_with_change_df.groupBy("DRUG") \
    .agg(F.sum("drug_change").alias("total_drug_changes")) \
    .orderBy(F.desc("total_drug_changes")) \
    .limit(10)

# Hiển thị top 10 loại thuốc thay đổi thường xuyên nhất
top_drugs_changes_df.show()
