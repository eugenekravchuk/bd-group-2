# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # France price prediction
# MAGIC Run `train_ab` first. After A/B is frozen, run the unchanged ETL for Bordeaux,
# MAGIC then use `evaluate_c` with the same run ID.

# COMMAND ----------

import sys, os, json, uuid
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
import sklearn
from pyspark.sql import functions as F
sys.path.insert(0, os.path.abspath('../model'))
from price_model import FEATURES, fit, score, metrics

dbutils.widgets.dropdown('mode', 'train_ab', ['train_ab', 'evaluate_c'])
dbutils.widgets.text('run_id', '')
dbutils.widgets.text('catalog', 'airbnb_lakehouse')
mode = dbutils.widgets.get('mode')
catalog = dbutils.widgets.get('catalog')
assert catalog.replace('_', '').isalnum(), 'Use a simple catalog identifier'
run_id = dbutils.widgets.get('run_id') or str(uuid.uuid4())
assert all(c.isalnum() or c in '-_' for c in run_id)
gold = f'{catalog}.gold'
spark.sql(f'CREATE VOLUME IF NOT EXISTS {gold}.model_artifacts')
folder = f'/Volumes/{catalog}/gold/model_artifacts/{run_id}'

def load(cities):
    source = spark.table(f'{gold}.listing_features').filter(F.col('city').isin(cities))
    required = ['city', 'listing_id', 'snapshot_date', 'price', 'split', 'neighbourhood'] + FEATURES
    missing = set(required) - set(source.columns)
    assert not missing, f'Missing ETL fields: {missing}'
    frame = source.select(*required).toPandas()
    assert set(frame.city) == set(cities), 'Missing city; run ETL first'
    assert not frame.duplicated(['city', 'listing_id']).any(), 'Duplicate listing keys'
    frame['price'] = pd.to_numeric(frame.price, errors='coerce')
    frame.loc[~np.isfinite(frame.price) | (frame.price <= 0), 'price'] = np.nan
    assert frame.split.isin(['train', 'test']).all()
    return frame

def publish(frame, table):
    # Append preserves first-run evidence and prior model versions.
    frame = frame.copy()
    frame['run_id'] = run_id
    frame['evaluation_stage'] = mode
    spark.createDataFrame(frame).withColumn('created_at', F.current_timestamp()).write.mode('append').saveAsTable(f'{gold}.{table}')

# COMMAND ----------

if mode == 'train_ab':
    assert not os.path.exists(folder), 'Run ID already exists; use a fresh ID'
    data = load(['paris', 'lyon'])
    train = data[(data.split == 'train') & data.price.notna()]
    test = data[(data.split == 'test') & data.price.notna()]
    assert set(train.city) == set(test.city) == {'paris', 'lyon'}
    model, baseline = fit(train)
    predictions = score(data, model, baseline)
    predictions['evaluation_set'] = np.where(predictions.split == 'test', 'ab_test', 'ab_train')
    report = metrics(predictions[predictions.evaluation_set == 'ab_test'])
    os.makedirs(folder)
    joblib.dump((model, baseline), folder + '/frozen_model.joblib')
    manifest = {'run_id': run_id, 'frozen_at': datetime.now(timezone.utc).isoformat(),
                'sklearn_version': sklearn.__version__, 'features': FEATURES,
                'snapshots': data.groupby('city').snapshot_date.apply(lambda x: sorted(set(map(str, x)))).to_dict(),
                'training_rows': len(train), 'test_rows': len(test), 'currency': 'EUR'}
    with open(folder + '/manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    report.to_csv(folder + '/ab_test_metrics.csv', index=False)
else:
    assert os.path.exists(folder + '/manifest.json'), 'Provide the frozen A/B run ID'
    assert not os.path.exists(folder + '/c_first_metrics.csv'), 'First Bordeaux run is already recorded'
    with open(folder + '/manifest.json') as f:
        manifest = json.load(f)
    assert manifest['sklearn_version'] == sklearn.__version__, 'Use the original sklearn version'
    model, baseline = joblib.load(folder + '/frozen_model.joblib')
    data = load(['bordeaux'])
    predictions = score(data, model, baseline)
    predictions['evaluation_set'] = 'c_unseen'
    report = metrics(predictions)
    report.to_csv(folder + '/c_first_metrics.csv', index=False)
    ab = pd.read_csv(folder + '/ab_test_metrics.csv').query("city == 'combined'").iloc[0]
    print('Bordeaux minus A/B test MAE (EUR):', report.iloc[0].model_mae - ab.model_mae)
    print('Bordeaux snapshot(s):', sorted(set(map(str, data.snapshot_date))))

# COMMAND ----------

publish(predictions[['city', 'listing_id', 'snapshot_date', 'neighbourhood', 'room_type',
                     'accommodates', 'price', 'predicted_price', 'baseline_price', 'residual',
                     'absolute_error', 'baseline_absolute_error', 'evaluation_set']], 'price_predictions')
publish(report, 'model_metrics')
display(report)
print('Frozen model run ID:', run_id)
print('Label coverage:', data.groupby('city').price.agg(['count', 'size']))
display(predictions[predictions.evaluation_set.isin(['ab_test', 'c_unseen'])]
        .groupby(['city', 'room_type']).agg(n=('absolute_error', 'count'),
        model_mae=('absolute_error', 'mean'), baseline_mae=('baseline_absolute_error', 'mean')).reset_index())