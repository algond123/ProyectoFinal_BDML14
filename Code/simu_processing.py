from pyspark.sql import SparkSession

# Crear sesión Spark
spark = SparkSession.builder \
    .appName("SpotifyFromSQL") \
    .config("spark.driver.extraClassPath", "/path/to/sqlite-jdbc.jar") \
    .getOrCreate()

# Leer desde SQLite (requiere el JDBC driver de SQLite)
jdbc_url = "jdbc:sqlite:data/spotify_data.db"

df = spark.read \
    .format("jdbc") \
    .option("url", jdbc_url) \
    .option("dbtable", "tracks") \
    .load()

df.printSchema()
df.show(5)

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, round as spark_round

# Crear sesión de Spark
spark = SparkSession.builder \
    .appName("SpotifyCleanNormalize") \
    .getOrCreate()

# Ruta al dataset original (ajusta si usas otra carpeta)
dataset_path = "data/spotify_dataset.csv"

# Cargar el dataset
df = spark.read.option("header", True).option("inferSchema", True).csv(dataset_path)

# Filtrar columnas de interés y eliminar valores nulos
features = ['energy', 'tempo', 'loudness', 'danceability', 'valence']
df = df.select(*features).na.drop()

# Clipping de valores
df = df.withColumn("tempo", when(col("tempo") < 20, 20).when(col("tempo") > 200, 200).otherwise(col("tempo")))
df = df.withColumn("loudness", when(col("loudness") < -60, -60).when(col("loudness") > 0, 0).otherwise(col("loudness")))
df = df.withColumn("energy", when(col("energy") < 0, 0).when(col("energy") > 1, 1).otherwise(col("energy")))
df = df.withColumn("danceability", when(col("danceability") < 0, 0).when(col("danceability") > 1, 1).otherwise(col("danceability")))

# Escalado manual Min-Max
df = df.withColumn("tempo", (col("tempo") - 20) / (200 - 20))
df = df.withColumn("loudness", (col("loudness") + 60) / 60)

# Calcular 'arousal' con ponderaciones
alpha = 0.5
beta = 0.25
gamma = 0.2
delta = 0.05

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
for feature in ['energy', 'tempo', 'loudness', 'danceability']:
    df = df.withColumn(feature, spark_round(col(feature), 3))

# Mostrar estadísticas finales
df.select("arousal", "valence").describe().show()

# Guardar en CSV (opcional)
df.select("arousal", "valence") \
    .write.mode("overwrite") \
    .option("header", True) \
    .csv("data/arousal_valence_output")

# Cerrar sesión Spark
spark.stop()


spark.stop()
