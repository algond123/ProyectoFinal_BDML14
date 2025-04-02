from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, round as spark_round, mean
from pyspark.sql.functions import stddev

# Crear sesión de Spark (agrega el path real del driver JDBC si es necesario)
spark = SparkSession.builder \
    .appName("SpotifyFromDB_Metrics") \
    .master("local[*]") \
    .config("spark.security.manager", "false") \
    .config("spark.hadoop.security.manager", "None") \
    .config("spark.driver.extraClassPath", "./libs/sqlite-jdbc.jar") \
    .getOrCreate()

# Leer desde SQLite
jdbc_url = "jdbc:sqlite:./Data/spotify_data.db"

df = spark.read \
    .format("jdbc") \
    .option("url", jdbc_url) \
    .option("dbtable", "tracks") \
    .load()

# Confirmar estructura
df.printSchema()
df.select("track_name", "energy", "tempo", "loudness", "danceability", "valence").show(10)

# Filtrar columnas necesarias y eliminar nulos
features = ['energy', 'tempo', 'loudness', 'danceability', 'valence']
df = df.select(*features).na.drop()

# Clipping
df = df.withColumn("tempo", when(col("tempo") < 20, 20).when(col("tempo") > 200, 200).otherwise(col("tempo")))
df = df.withColumn("loudness", when(col("loudness") < -60, -60).when(col("loudness") > 0, 0).otherwise(col("loudness")))
df = df.withColumn("energy", when(col("energy") < 0, 0).when(col("energy") > 1, 1).otherwise(col("energy")))
df = df.withColumn("danceability", when(col("danceability") < 0, 0).when(col("danceability") > 1, 1).otherwise(col("danceability")))

# Escalado Min-Max
df = df.withColumn("tempo", (col("tempo") - 20) / (200 - 20))
df = df.withColumn("loudness", (col("loudness") + 60) / 60)

# Cálculo de 'arousal'
alpha, beta, gamma, delta = 0.5, 0.25, 0.2, 0.05

df = df.withColumn("arousal",
    spark_round(
        alpha * col("energy") +
        beta * col("tempo") +
        gamma * col("loudness") +
        delta * col("danceability"),
        3
    )
)

# Redondear columnas
for feature in ['energy', 'tempo', 'loudness', 'danceability', 'arousal', 'valence']:
    df = df.withColumn(feature, spark_round(col(feature), 3))

# Nueva estructura
df.printSchema()
df.select("arousal", "valence", "energy", "tempo", "loudness", "danceability", "valence").show(10)

# 1. **Metrics for All Columns**
print("\nMetrics for All Columns:")
df.describe().show()

# 2. **Metrics for Selected Features**
print("\nMetrics for Selected Features:")
df.select("energy", "tempo", "loudness", "danceability", "arousal", "valence").describe().show()

# 3. **Correlation between energy and arousal**
correlation_matrix = df.select("energy", "arousal").stat.corr("energy", "arousal")
print(f"Correlation between energy and arousal: {correlation_matrix}\n")

# 4. **Skewness of each feature**
skewness = df.select("energy", "tempo", "loudness", "danceability", "arousal", "valence").agg(
    {"energy": "skewness", "tempo": "skewness", "loudness": "skewness", "danceability": "skewness", "arousal": "skewness", "valence": "skewness"}
)
print("\nSkewness of each feature:")
skewness.show()

# 5. **Quantiles of each feature**
quantiles = df.select("energy", "tempo", "loudness", "danceability", "arousal", "valence").approxQuantile(
    "energy", [0.25, 0.5, 0.75], 0.05
)
print(f"Energy quantiles: {quantiles}\n")

# 6. **Standard Deviation of each feature**
stddev = df.select("energy", "tempo", "loudness", "danceability", "arousal", "valence").agg(
    stddev("energy").alias("stddev_energy"),
    stddev("tempo").alias("stddev_tempo"),
    stddev("loudness").alias("stddev_loudness"),
    stddev("danceability").alias("stddev_danceability"),
    stddev("arousal").alias("stddev_arousal"),
    stddev("valence").alias("stddev_valence")
)
print("\nStandard Deviation of each feature:")
stddev.show()

# 7. **Kurtosis of each feature**
kurtosis = df.select("energy", "tempo", "loudness", "danceability", "arousal", "valence").agg(
    {"energy": "kurtosis", "tempo": "kurtosis", "loudness": "kurtosis", "danceability": "kurtosis", "arousal": "kurtosis", "valence": "kurtosis"}
)
print("\nKurtosis of each feature:")
kurtosis.show()

# 8. **Variance of each feature**
variance = df.select("energy", "tempo", "loudness", "danceability", "arousal", "valence").agg(
    {"energy": "variance", "tempo": "variance", "loudness": "variance", "danceability": "variance", "arousal": "variance", "valence": "variance"}
)
print("\nVariance of each feature:")
variance.show()

# 9. **Min and Max values of each feature**
min_max_values = df.select("energy", "tempo", "loudness", "danceability", "arousal", "valence").agg(
    {"energy": "min", "energy": "max", "tempo": "min", "tempo": "max", "loudness": "min", "loudness": "max", 
     "danceability": "min", "danceability": "max", "arousal": "min", "arousal": "max", "valence": "min", "valence": "max"}
)
print("\nMin and Max values of each feature:")
min_max_values.show()

# 10. **Final Statistics for arousal and valence**
print("\nFinal Statistics for Arousal and Valence:")
df.select("arousal", "valence").describe().show()

print("Save on spark_arousal_valence_output.csv")

# Guardar (opcional)
df.select("arousal", "valence") \
    .write.mode("overwrite") \
    .option("header", True) \
    .csv("./Code/Data/spark_arousal_valence_output.csv")

# Detener Spark
spark.stop()
