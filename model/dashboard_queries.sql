SELECT * FROM airbnb_lakehouse.gold.model_metrics
WHERE run_id = 'france_v1'
ORDER BY evaluation_stage, city;

SELECT city, room_type, COUNT(absolute_error) AS labeled_listings,
       AVG(absolute_error) AS model_mae_eur,
       AVG(baseline_absolute_error) AS baseline_mae_eur
FROM airbnb_lakehouse.gold.price_predictions
WHERE run_id = 'france_v1' AND evaluation_set IN ('ab_test', 'c_unseen')
GROUP BY city, room_type;

SELECT city, listing_id, price, predicted_price, residual, neighbourhood
FROM airbnb_lakehouse.gold.price_predictions
WHERE run_id = 'france_v1' AND evaluation_set IN ('ab_test', 'c_unseen')
  AND price IS NOT NULL
ORDER BY ABS(residual) DESC LIMIT 30;
