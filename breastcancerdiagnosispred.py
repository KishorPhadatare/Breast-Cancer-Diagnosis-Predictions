from pyspark import SparkContext
from pyspark.sql import SparkSession

sc= SparkContext(appName="DecisionTrees")
! echo $PYSPARK_SUBMIT_ARGS
spark= SparkSession.Builder().getOrCreate()

#Downloading Dataset for Breast Cancer Diagonsis:
! wget https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data
!head -n 3 wdbc.data
!wc -l wdbc.data

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import StringIndexer

#Loading the data 
data[]
with open("wdbc.data") as infile:
	for line in infile:
		tokens=line.rstrip("\n").split(",")
		y=tokens[1]
		features= Vector.dense([float (x) for x in 	tokens[2:]])
		data.append((y,features))
		
input=spark.createDataFrame(data,["lable", "features"])
input.show()

stringIndexer= StringIndexer(inputCol="lable", outputCol="lableIndex")
si_model=stringIndexer.fit(input)
input=si_model.transform(input)
# lableIndex=1.0 ==Malignant & 0.0== B

train, test=input.randomSplit([0.7,0.3], seed=100)

#Implementing DecisionTrees:
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

df= DecisionTreeClassifier(lableCol="lableIndex")
model=df.fit(train)
model.numNodes
model.depth
model.featureImportances
model.numFeatures

#Predicting The Output:
pred=model.transform(test)
pred.select('lable', 'lableIndex','probability', 'prediction').show()

#Evaluating Model:
evaluator=MulticlassClassificationEvaluator(lableCol="lableIndex", predictionCol="prediction", metricName="accuracy")
accuracy=evaluator.evaluate(pred)
print("Error= %g" %(1-accuracy))


#Implementing GradientBoost Algorithm:
 
from pyspark.ml.Classification import GBTClassifier
dataFrame= GBTClassifier(lableCol="lableIndex", featuresCol="features", maxIter=100, stepSize=0.1)
model=dataFrame.fit(train)
model.FeatureImportances()

#Predicting The Output
pred=model.transform(test)
pred.select('lable', 'lableIndex','probability', 'prediction').show()

#Evaluating Model:
evaluator=MulticlassClassificationEvaluator(lableCol="lableIndex", predictionCol="prediction", metricName="accuracy")
accuracy=evaluator.evaluate(pred)
print("Error= %g" %(1-accuracy))


#Implementing Random Forest:
from pyspark.ml.Classification import RandomForestClassifier
dframe=RandomForestClassifier(lableCol="lableIndex",numTrees=100)
model=dframe.fit(train)
model.FeatureImportances

#Predicting The Output
pred=model.transform(test)
pred.select('lable', 'lableIndex','probability', 'prediction').show()

#Evaluating Model:
evaluator=MulticlassClassificationEvaluator(lableCol="lableIndex", predictionCol="prediction", metricName="accuracy")
accuracy=evaluator.evaluate(pred)
print("Error= %g" %(1-accuracy))


#Implementing Cross Validation for DecisionTrees:
from pyspark.ml import Pipeline
pipeline= Pipeline(stages=[decisionTree])

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
paramGrid= ParamGridBuilder().addGrid(decisionTree.maxDepth,[1,2,3,4,5,6,7,8]).build()

#paramGrid= ParamGridBuilder().addGrid(decisionTree.maxDepth,[1,2,3,4,5,6,7,8]).addGrid(decisionTree.minInstancesPerNode,[1,2,3,4,5,6,7,8]).build()
evaluator=MulticlassClassificationEvaluator(lableCol="lableIndex", predictionCol="prediction", metricName="accuracy")
crossVal=CrossValidator(estimator=pipeline, estimatorParamMaps=paramGrid, evaluator=evaluator,numFolds=10)
cvmodel= crossVal.fit(train)
cvmodel.avgMetrics
print cvmodel.bestModel.stages[0]

#Implementing Cross Validation for GradientBoost:
pipeline= Pipeline(stages=[gradientBoost])

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
paramGrid= ParamGridBuilder().addGrid(gradientBoost.maxDepth,[1,2,3,4,5,6,7,8]).build()

#paramGrid= ParamGridBuilder().addGrid(gradientBoost.maxDepth,[1,2,3,4,5,6,7,8]).addGrid(gradientBoost.minInstancesPerNode,[1,2,3,4,5,6,7,8]).build()
evaluator=MulticlassClassificationEvaluator(lableCol="lableIndex", predictionCol="prediction", metricName="accuracy")
crossVal=CrossValidator(estimator=pipeline, estimatorParamMaps=paramGrid, evaluator=evaluator,numFolds=10)
cvmodel= crossVal.fit(train)
cvmodel.avgMetrics
print cvmodel.bestModel.stages[0]

#Implementing Cross Validation for RandomForestClassifier:
pipeline= Pipeline(stages=[randomForest])

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
paramGrid= ParamGridBuilder().addGrid(randomForest.maxDepth,[1,2,3,4,5,6,7,8]).build()

#paramGrid= ParamGridBuilder().addGrid(randomForest.maxDepth,[1,2,3,4,5,6,7,8]).addGrid(randomForest.minInstancesPerNode,[1,2,3,4,5,6,7,8]).build()
evaluator=MulticlassClassificationEvaluator(lableCol="lableIndex", predictionCol="prediction", metricName="accuracy")
crossVal=CrossValidator(estimator=pipeline, estimatorParamMaps=paramGrid, evaluator=evaluator,numFolds=10)
cvmodel= crossVal.fit(train)
cvmodel.avgMetrics
print cvmodel.bestModel.stages[0]


