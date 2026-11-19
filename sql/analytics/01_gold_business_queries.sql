-- 1. Monthly Revenue & Order Volume Trend
SELECT
    d.year,
    d.month,
    d.month_name,
    COUNT(DISTINCT f.order_id) AS total_orders,
    ROUND(SUM(f.gross_order_value)::numeric, 2) AS total_revenue,
    ROUND(AVG(f.avg_review_score)::numeric, 2) AS avg_review_score
FROM gold.factorders f
JOIN gold.dimdate d ON f.order_date_key = d.date_key
WHERE f.order_status NOT IN ('canceled', 'unavailable')
GROUP BY d.year, d.month, d.month_name
ORDER BY d.year, d.month;

-- 2. Top 10 Product Categories by Gross Sales
SELECT
    p.product_category_name_english AS category,
    COUNT(*) AS items_sold,
    ROUND(SUM(fi.price)::numeric, 2) AS gross_sales
FROM gold.factorderitems fi
JOIN gold.dimproduct p ON fi.product_sk = p.product_sk
GROUP BY p.product_category_name_english
ORDER BY gross_sales DESC
LIMIT 10;

-- 3. Late Delivery Impact on Customer Satisfaction
SELECT
    is_late_delivery,
    COUNT(*) AS delivered_orders,
    ROUND(AVG(delivery_days)::numeric, 1) AS avg_delivery_days,
    ROUND(AVG(avg_review_score)::numeric, 2) AS avg_review_score
FROM gold.factorders
WHERE order_status = 'delivered' AND avg_review_score IS NOT NULL
GROUP BY is_late_delivery;
