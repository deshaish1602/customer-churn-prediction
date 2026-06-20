
-- ================================================
-- Customer Churn Analysis Queries
-- ================================================

-- 1. Overall Churn Rate
SELECT 
    Churn,
    COUNT(*) as total_customers,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage
FROM customers
GROUP BY Churn;

-- 2. Churn by Contract Type
SELECT 
    Contract,
    COUNT(*) as total_customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) as churned,
    ROUND(SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as churn_rate
FROM customers
GROUP BY Contract
ORDER BY churn_rate DESC;

-- 3. High Risk Segments
SELECT 
    Contract,
    InternetService,
    PaymentMethod,
    COUNT(*) as total_customers,
    ROUND(SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as churn_rate
FROM customers
WHERE Contract = 'Month-to-month'
  AND InternetService = 'Fiber optic'
GROUP BY PaymentMethod
ORDER BY churn_rate DESC;

-- 4. Revenue Lost to Churn
SELECT 
    ROUND(SUM(MonthlyCharges), 2) as monthly_revenue_lost,
    ROUND(SUM(MonthlyCharges) * 12, 2) as annual_revenue_lost
FROM customers
WHERE Churn = 'Yes';

-- 5. Customer Lifetime Value
SELECT 
    Contract,
    ROUND(AVG(tenure * MonthlyCharges), 2) as avg_lifetime_value
FROM customers
GROUP BY Contract
ORDER BY avg_lifetime_value DESC;
