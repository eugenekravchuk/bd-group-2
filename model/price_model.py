"""Training-only preprocessing and baseline; no Spark dependency."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

NUMERIC = ['accommodates', 'bedrooms', 'beds', 'bathrooms', 'amenities_count',
           'minimum_nights', 'host_is_superhost', 'host_listings_count',
           'host_years', 'instant_bookable']
CATEGORICAL = ['room_type', 'property_type', 'bathroom_type']
FEATURES = NUMERIC + CATEGORICAL

def features(df):
    out = df[FEATURES].copy()
    for c in NUMERIC:
        out[c] = pd.to_numeric(out[c], errors='coerce').replace([np.inf, -np.inf], np.nan)
    for c in CATEGORICAL:
        out[c] = out[c].fillna('__missing__').astype(str)
    return out

def fit(train):
    """Fixed, explainable configuration: never tune using final test or Bordeaux."""
    x = features(train)
    preprocess = ColumnTransformer([
        ('numeric', SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True), NUMERIC),
        ('category', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL),
    ])
    model = Pipeline([('prepare', preprocess), ('regressor', RandomForestRegressor(
        n_estimators=120, max_depth=16, min_samples_leaf=5,
        criterion='squared_error', random_state=42, n_jobs=-1))])
    model.fit(x, train.price)
    keys = train.assign(room_type=train.room_type.fillna('__missing__'),
                        accommodates=train.accommodates.fillna(-1))
    baseline = {
        'pairs': keys.groupby(['room_type', 'accommodates']).price.median().to_dict(),
        'rooms': keys.groupby('room_type').price.median().to_dict(),
        'global': float(train.price.median()),
    }
    return model, baseline

def baseline_predict(df, baseline):
    rooms = df.room_type.fillna('__missing__')
    guests = df.accommodates.fillna(-1)
    return np.array([baseline['pairs'].get((r, a), baseline['rooms'].get(r, baseline['global']))
                     for r, a in zip(rooms, guests)])

def score(df, model, baseline):
    out = df.copy()
    out['predicted_price'] = model.predict(features(df))
    out['baseline_price'] = baseline_predict(df, baseline)
    out['residual'] = out.price - out.predicted_price
    out['absolute_error'] = abs(out.residual)
    out['baseline_absolute_error'] = abs(out.price - out.baseline_price)
    return out

def metrics(scored):
    rows = []
    groups = [(c, g) for c, g in scored.groupby('city')] + [('combined', scored)]
    for city, g in groups:
        g = g.loc[np.isfinite(g.price) & (g.price > 0)]
        if g.empty:
            raise ValueError('No valid evaluation labels for ' + city)
        model_mae, baseline_mae = g.absolute_error.mean(), g.baseline_absolute_error.mean()
        rows.append(dict(city=city, n=len(g), model_mae=float(model_mae),
                         baseline_mae=float(baseline_mae),
                         improvement_eur=float(baseline_mae-model_mae)))
    return pd.DataFrame(rows)
