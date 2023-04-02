#### Big Data Project 3
#### Ahbab Ashraf

from pyspark.sql import SparkSession
from pyspark.sql.context import SQLContext
from pyspark.sql.types import StructType
from pyspark.sql.functions import lit

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorIndexer, VectorAssembler
from pyspark.ml.regression import LinearRegression 

from scipy.stats import spearmanr
from scipy.stats import pearsonr

import pandas as pd



# Make spark session
spark = SparkSession.builder.appName('spark-sql').master('local').getOrCreate()
sqlContext = SQLContext(spark)
filepath = 'dataset.csv'

# Read csv into dataframe
data = sqlContext.read.csv(filepath, header=True, inferSchema="true")

# Drop songs that appear more than once
data = data.dropDuplicates(['artists', 'track_name'])

# Cast all columns to double
for col in data.columns:
    data = data.withColumn(col, data[col].cast('double'))

# Drop string columns
data = data.drop('track_id', 'artists', 'album_name', 'track_name', 'track_genre')
# Drop binary columns
data = data.drop('mode', 'time_signature', 'explicit')

# Get all features
features = (data.drop('popularity')).columns

# Initialize vector assembler to put features into a single column
featAssembler = VectorAssembler(inputCols=features, outputCol='features')

# Assemble features into a single column
output = featAssembler.setHandleInvalid("skip").transform(data)

# Place feature data and label data in the same dataframe
vectorData = output.select('features', 'popularity')

# Split up the data into thirds based on popularity
lowPop = vectorData[vectorData.popularity <= 24]
midPop = vectorData[ ((vectorData.popularity > 24) & (vectorData.popularity <= 43)) ]
highPop = vectorData[vectorData.popularity > 43]

# Split each division randomly into 5 parts
lows = lowPop.randomSplit(weights=[.2, .2, .2, .2, .2])
mids = midPop.randomSplit(weights=[.2, .2, .2, .2, .2])
highs = highPop.randomSplit(weights=[.2, .2, .2, .2, .2])

# Make 5 splits of data with an equal distribution of low, mid, and high popularity in each split
splits = [0] * 5
for i in range(5):
    splits[i] = lows[i]
    splits[i] = splits[i].unionAll(mids[i])
    splits[i] = splits[i].unionAll(highs[i])

# Initialize linear regression model
model = LinearRegression(featuresCol='features', labelCol='popularity', regParam = 0.3)

# Avg across folds
avgPear = 0
avgSpear = 0

# Perform model analysis on all 5 folds
for i in range(5):
    
    # Initialize train and test data variables
    train_data = spark.createDataFrame([], StructType([]))
    train_data = train_data.withColumn('features', lit(None))
    train_data = train_data.withColumn('popularity', lit(None))
    test_data = None
    
    # Unify splits to make data for fold
    for j in range(5):
        # Use 1 split for testing
        if (i == j):
            test_data = splits[j]
        # Use 4 splits for training
        else:
            train_data = train_data.unionAll(splits[j])
    
    # Data is ready
    print('\n> Fold {}'.format(i))
    
    # Fit the model to the training data
    ss = model.fit(train_data)

    # Make predictions for the test data
    pred = ss.evaluate(test_data)

    # Place predictions inside pandas DataFrame
    pred_df = pred.predictions.toPandas()

    # Get predicted and actual popularity as arrays
    pred_pop = pred_df['prediction'].values
    actual_pop = pred_df['popularity'].values

    # Calculate values for evaulation metrics
    pearson = pearsonr(pred_pop, actual_pop)
    spearman = spearmanr(pred_pop, actual_pop)

    avgPear += pearson[0]
    avgSpear += spearman[0]

    # Print Pearson and spearman correlation
    print('Pearson\'s Correlation: {}\nSpearman\'s Correlation: {}\n'.format(pearson[0], spearman[0]))
    
# Get avg metrics
avgPear /= 5
avgSpear /= 5
print('> Average Across Folds')
print('Pearson\'s Correlation: {}\nSpearman\'s Correlation: {}\n'.format(avgPear, avgSpear))