# Lesson 1: SQL & Data Manipulation 🗄️

**Module 3: Data Engineering | Lesson 1 of 4**

Master SQL - the essential skill for data extraction and manipulation in ML!

---

## Why SQL for ML?

**80% of ML work is data preparation:**
- Data stored in databases (SQL)
- ETL pipelines use SQL
- Feature engineering often starts with SQL
- Most companies use SQL databases

**Reality:** You'll spend more time writing SQL than training models!

---

## 1. SQL Basics 📚

### SELECT - Read Data

```sql
-- Basic select
SELECT * FROM customers;

-- Select specific columns
SELECT customer_id, name, email FROM customers;

-- Select with alias
SELECT
    customer_id AS id,
    name AS customer_name
FROM customers;

-- Limit results
SELECT * FROM customers LIMIT 10;
```

### WHERE - Filter Data

```sql
-- Basic filtering
SELECT * FROM customers
WHERE age > 30;

-- Multiple conditions
SELECT * FROM customers
WHERE age > 30 AND city = 'NYC';

-- OR condition
SELECT * FROM customers
WHERE city = 'NYC' OR city = 'LA';

-- IN operator
SELECT * FROM customers
WHERE city IN ('NYC', 'LA', 'SF');

-- LIKE pattern matching
SELECT * FROM customers
WHERE name LIKE 'John%';  -- Starts with John

-- NULL handling
SELECT * FROM customers
WHERE email IS NOT NULL;
```

### ORDER BY - Sort Results

```sql
-- Ascending (default)
SELECT * FROM customers
ORDER BY age;

-- Descending
SELECT * FROM customers
ORDER BY age DESC;

-- Multiple columns
SELECT * FROM customers
ORDER BY city, age DESC;
```

---

## 2. Aggregations 📊

### Basic Aggregations

```sql
-- Count
SELECT COUNT(*) FROM customers;

-- Sum
SELECT SUM(revenue) FROM orders;

-- Average
SELECT AVG(age) FROM customers;

-- Min/Max
SELECT MIN(age), MAX(age) FROM customers;

-- Distinct count
SELECT COUNT(DISTINCT city) FROM customers;
```

### GROUP BY

```sql
-- Group by single column
SELECT city, COUNT(*) as customer_count
FROM customers
GROUP BY city;

-- Multiple aggregations
SELECT
    city,
    COUNT(*) as count,
    AVG(age) as avg_age,
    MIN(age) as min_age,
    MAX(age) as max_age
FROM customers
GROUP BY city;

-- Multiple grouping columns
SELECT
    city,
    age_group,
    COUNT(*) as count
FROM customers
GROUP BY city, age_group;
```

### HAVING - Filter After Aggregation

```sql
-- Filter groups (not rows)
SELECT city, COUNT(*) as count
FROM customers
GROUP BY city
HAVING COUNT(*) > 100;

-- HAVING with WHERE
SELECT city, AVG(revenue) as avg_revenue
FROM orders
WHERE order_date >= '2023-01-01'
GROUP BY city
HAVING AVG(revenue) > 1000;
```

---

## 3. JOINs - Combine Tables 🔗

**Most important SQL skill for ML feature engineering!**

### INNER JOIN

```sql
-- Match rows from both tables
SELECT
    c.customer_id,
    c.name,
    o.order_id,
    o.amount
FROM customers c
INNER JOIN orders o ON c.customer_id = o.customer_id;

-- Only returns customers who have orders
```

### LEFT JOIN

```sql
-- All rows from left table + matching from right
SELECT
    c.customer_id,
    c.name,
    COUNT(o.order_id) as order_count,
    SUM(o.amount) as total_spent
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id, c.name;

-- Returns all customers (even with 0 orders)
```

### Multiple JOINs

```sql
SELECT
    c.customer_id,
    c.name,
    o.order_id,
    p.product_name,
    oi.quantity
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
LEFT JOIN order_items oi ON o.order_id = oi.order_id
LEFT JOIN products p ON oi.product_id = p.product_id;
```

---

## 4. Window Functions 🪟

**Powerful for feature engineering!**

### ROW_NUMBER

```sql
-- Rank rows within groups
SELECT
    customer_id,
    order_date,
    amount,
    ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date) as order_number
FROM orders;

-- Result: 1st order, 2nd order, etc. per customer
```

### RANK and DENSE_RANK

```sql
SELECT
    customer_id,
    amount,
    RANK() OVER (ORDER BY amount DESC) as rank,
    DENSE_RANK() OVER (ORDER BY amount DESC) as dense_rank
FROM orders;

-- RANK: 1, 2, 2, 4 (skips 3)
-- DENSE_RANK: 1, 2, 2, 3 (no skip)
```

### LAG and LEAD

```sql
-- Previous/next row values
SELECT
    customer_id,
    order_date,
    amount,
    LAG(amount) OVER (PARTITION BY customer_id ORDER BY order_date) as prev_amount,
    LEAD(amount) OVER (PARTITION BY customer_id ORDER BY order_date) as next_amount
FROM orders;

-- Use for time series features!
```

### Running Totals

```sql
SELECT
    customer_id,
    order_date,
    amount,
    SUM(amount) OVER (PARTITION BY customer_id ORDER BY order_date) as cumulative_spending
FROM orders;
```

### Moving Averages

```sql
-- 7-day moving average
SELECT
    order_date,
    amount,
    AVG(amount) OVER (
        ORDER BY order_date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) as ma_7day
FROM orders;
```

---

## 5. Subqueries & CTEs 🎯

### Subqueries

```sql
-- Subquery in WHERE
SELECT * FROM customers
WHERE customer_id IN (
    SELECT DISTINCT customer_id
    FROM orders
    WHERE amount > 1000
);

-- Subquery in SELECT
SELECT
    customer_id,
    name,
    (SELECT COUNT(*) FROM orders o WHERE o.customer_id = c.customer_id) as order_count
FROM customers c;
```

### CTEs (Common Table Expressions) - Preferred!

```sql
-- More readable than subqueries
WITH high_value_customers AS (
    SELECT customer_id, SUM(amount) as total_spent
    FROM orders
    GROUP BY customer_id
    HAVING SUM(amount) > 10000
)
SELECT
    c.name,
    h.total_spent
FROM customers c
JOIN high_value_customers h ON c.customer_id = h.customer_id;
```

### Multiple CTEs

```sql
WITH
order_stats AS (
    SELECT
        customer_id,
        COUNT(*) as order_count,
        SUM(amount) as total_spent
    FROM orders
    GROUP BY customer_id
),
customer_segments AS (
    SELECT
        customer_id,
        CASE
            WHEN total_spent > 10000 THEN 'VIP'
            WHEN total_spent > 5000 THEN 'Premium'
            ELSE 'Standard'
        END as segment
    FROM order_stats
)
SELECT * FROM customer_segments;
```

---

## 6. ML-Specific SQL Patterns 🤖

### Feature Engineering Queries

```sql
-- Customer features for ML model
WITH customer_features AS (
    SELECT
        c.customer_id,
        c.age,
        c.city,
        COUNT(o.order_id) as total_orders,
        SUM(o.amount) as total_spent,
        AVG(o.amount) as avg_order_value,
        MAX(o.order_date) as last_order_date,
        MIN(o.order_date) as first_order_date,
        DATEDIFF(CURRENT_DATE, MAX(o.order_date)) as days_since_last_order,
        COUNT(DISTINCT o.product_category) as unique_categories
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id
    GROUP BY c.customer_id, c.age, c.city
)
SELECT
    *,
    CASE
        WHEN days_since_last_order < 30 THEN 'Active'
        WHEN days_since_last_order < 90 THEN 'At Risk'
        ELSE 'Churned'
    END as customer_status
FROM customer_features;
```

### Time-Based Features

```sql
-- Extract date features
SELECT
    order_id,
    order_date,
    EXTRACT(YEAR FROM order_date) as year,
    EXTRACT(MONTH FROM order_date) as month,
    EXTRACT(DOW FROM order_date) as day_of_week,
    EXTRACT(HOUR FROM order_timestamp) as hour,
    CASE
        WHEN EXTRACT(DOW FROM order_date) IN (0, 6) THEN 1
        ELSE 0
    END as is_weekend
FROM orders;
```

### Pivot Tables

```sql
-- Pivot: Categories as columns
SELECT
    customer_id,
    SUM(CASE WHEN category = 'Electronics' THEN amount ELSE 0 END) as electronics_spent,
    SUM(CASE WHEN category = 'Clothing' THEN amount ELSE 0 END) as clothing_spent,
    SUM(CASE WHEN category = 'Food' THEN amount ELSE 0 END) as food_spent
FROM orders
GROUP BY customer_id;
```

### Train/Test Split

```sql
-- Random split for ML
SELECT
    *,
    CASE
        WHEN RANDOM() < 0.8 THEN 'train'
        ELSE 'test'
    END as dataset_split
FROM features;

-- Time-based split
SELECT
    *,
    CASE
        WHEN order_date < '2023-10-01' THEN 'train'
        ELSE 'test'
    END as dataset_split
FROM features;
```

---

## 7. Python + SQL Integration 🐍

### SQLite (Local Database)

```python
import sqlite3
import pandas as pd

# Connect to database
conn = sqlite3.connect('data.db')

# Read SQL query into DataFrame
query = """
SELECT
    customer_id,
    COUNT(*) as order_count,
    SUM(amount) as total_spent
FROM orders
GROUP BY customer_id
"""

df = pd.read_sql_query(query, conn)
print(df.head())

# Execute query
conn.execute("""
CREATE TABLE IF NOT EXISTS customers (
    customer_id INTEGER PRIMARY KEY,
    name TEXT,
    age INTEGER,
    city TEXT
)
""")

# Insert data
df.to_sql('orders', conn, if_exists='replace', index=False)

# Close connection
conn.close()
```

### PostgreSQL

```python
import psycopg2
import pandas as pd
from sqlalchemy import create_engine

# SQLAlchemy (recommended)
engine = create_engine('postgresql://user:password@localhost:5432/dbname')

# Read query
df = pd.read_sql_query("""
SELECT * FROM customers WHERE age > 30
""", engine)

# Write DataFrame to table
df.to_sql('processed_data', engine, if_exists='replace', index=False)

# Direct psycopg2
conn = psycopg2.connect(
    host="localhost",
    database="mydb",
    user="user",
    password="password"
)

cursor = conn.cursor()
cursor.execute("SELECT * FROM customers LIMIT 10")
rows = cursor.fetchall()

cursor.close()
conn.close()
```

### MySQL

```python
import mysql.connector
import pandas as pd
from sqlalchemy import create_engine

# SQLAlchemy
engine = create_engine('mysql+pymysql://user:password@localhost/dbname')
df = pd.read_sql_query("SELECT * FROM customers", engine)

# Direct connector
conn = mysql.connector.connect(
    host="localhost",
    user="user",
    password="password",
    database="mydb"
)

cursor = conn.cursor()
cursor.execute("SELECT * FROM customers")
results = cursor.fetchall()
```

---

## 8. Practical ML Workflow 🔄

### Complete Feature Engineering Pipeline

```python
import pandas as pd
import sqlite3

# Connect
conn = sqlite3.connect('ecommerce.db')

# Feature engineering query
feature_query = """
WITH customer_orders AS (
    SELECT
        c.customer_id,
        c.age,
        c.gender,
        c.city,
        COUNT(o.order_id) as order_count,
        SUM(o.amount) as total_spent,
        AVG(o.amount) as avg_order_value,
        MAX(o.order_date) as last_order_date,
        JULIANDAY('now') - JULIANDAY(MAX(o.order_date)) as days_since_last_order,
        COUNT(DISTINCT o.product_category) as unique_categories_purchased,
        -- Recent activity (last 30 days)
        SUM(CASE WHEN JULIANDAY('now') - JULIANDAY(o.order_date) <= 30 THEN 1 ELSE 0 END) as orders_last_30_days,
        SUM(CASE WHEN JULIANDAY('now') - JULIANDAY(o.order_date) <= 30 THEN o.amount ELSE 0 END) as spent_last_30_days
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id
    GROUP BY c.customer_id, c.age, c.gender, c.city
),
customer_features AS (
    SELECT
        *,
        -- Churn label (didn't order in 90 days)
        CASE
            WHEN days_since_last_order > 90 THEN 1
            ELSE 0
        END as churned,
        -- Customer value segment
        CASE
            WHEN total_spent > 10000 THEN 'High'
            WHEN total_spent > 5000 THEN 'Medium'
            ELSE 'Low'
        END as value_segment
    FROM customer_orders
)
SELECT * FROM customer_features
WHERE order_count > 0;  -- Only customers with orders
"""

# Load features
df = pd.read_sql_query(feature_query, conn)

print(f"Features shape: {df.shape}")
print(f"\nFeature columns:\n{df.columns.tolist()}")
print(f"\nChurn rate: {df['churned'].mean():.2%}")

# Now ready for ML!
X = df.drop(['customer_id', 'churned', 'value_segment'], axis=1)
y = df['churned']

# Continue with preprocessing and modeling...
```

### Temporal Feature Engineering

```python
temporal_query = """
WITH order_sequence AS (
    SELECT
        customer_id,
        order_id,
        order_date,
        amount,
        ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date) as order_number,
        LAG(amount) OVER (PARTITION BY customer_id ORDER BY order_date) as prev_order_amount,
        LAG(order_date) OVER (PARTITION BY customer_id ORDER BY order_date) as prev_order_date,
        -- Days between orders
        JULIANDAY(order_date) - JULIANDAY(LAG(order_date) OVER (PARTITION BY customer_id ORDER BY order_date)) as days_since_prev_order,
        -- Running average
        AVG(amount) OVER (
            PARTITION BY customer_id
            ORDER BY order_date
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ) as ma_3_orders
    FROM orders
)
SELECT
    *,
    -- Increasing/decreasing spending trend
    CASE
        WHEN amount > prev_order_amount THEN 'Increasing'
        WHEN amount < prev_order_amount THEN 'Decreasing'
        ELSE 'Stable'
    END as spending_trend
FROM order_sequence
WHERE order_number > 1;  -- Skip first order (no previous)
"""

df_temporal = pd.read_sql_query(temporal_query, conn)
```

---

## Quick Reference 📖

**Essential SQL Operations:**

```sql
-- SELECT
SELECT col1, col2 FROM table;

-- FILTER
WHERE condition AND/OR condition

-- AGGREGATE
SELECT col, COUNT(*), AVG(value) FROM table GROUP BY col

-- JOIN
FROM table1 t1
LEFT JOIN table2 t2 ON t1.id = t2.id

-- WINDOW
SELECT col,
       ROW_NUMBER() OVER (PARTITION BY group ORDER BY date) as row_num
FROM table

-- CTE
WITH temp AS (
    SELECT ...
)
SELECT * FROM temp
```

**Python Connection Strings:**

```python
# SQLite
'sqlite:///path/to/database.db'

# PostgreSQL
'postgresql://user:password@host:port/database'

# MySQL
'mysql+pymysql://user:password@host:port/database'
```

---

## Practice Exercises 🏋️

1. Write query to get top 10 customers by total spent
2. Create features: order frequency, recency, monetary value (RFM)
3. Use window functions to calculate customer growth over time
4. Build complete feature set for churn prediction

<details>
<summary>Solutions</summary>

```sql
-- 1. Top 10 customers
SELECT
    customer_id,
    SUM(amount) as total_spent
FROM orders
GROUP BY customer_id
ORDER BY total_spent DESC
LIMIT 10;

-- 2. RFM features
WITH rfm AS (
    SELECT
        customer_id,
        MAX(order_date) as last_order_date,
        COUNT(*) as frequency,
        SUM(amount) as monetary
    FROM orders
    GROUP BY customer_id
)
SELECT
    customer_id,
    JULIANDAY('now') - JULIANDAY(last_order_date) as recency_days,
    frequency,
    monetary
FROM rfm;

-- 3. Customer growth
WITH monthly_customers AS (
    SELECT
        strftime('%Y-%m', order_date) as month,
        COUNT(DISTINCT customer_id) as customer_count
    FROM orders
    GROUP BY month
)
SELECT
    month,
    customer_count,
    LAG(customer_count) OVER (ORDER BY month) as prev_month_count,
    customer_count - LAG(customer_count) OVER (ORDER BY month) as growth
FROM monthly_customers;

-- 4. Churn prediction features
WITH customer_features AS (
    SELECT
        c.customer_id,
        c.age,
        c.city,
        COUNT(o.order_id) as order_count,
        SUM(o.amount) as total_spent,
        AVG(o.amount) as avg_order_value,
        JULIANDAY('now') - JULIANDAY(MAX(o.order_date)) as days_since_last_order,
        MIN(o.order_date) as first_order_date,
        JULIANDAY(MAX(o.order_date)) - JULIANDAY(MIN(o.order_date)) as customer_lifetime_days,
        COUNT(DISTINCT o.product_category) as unique_categories
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id
    GROUP BY c.customer_id, c.age, c.city
)
SELECT
    *,
    CAST(order_count AS FLOAT) / NULLIF(customer_lifetime_days, 0) as order_frequency,
    CASE WHEN days_since_last_order > 90 THEN 1 ELSE 0 END as churned
FROM customer_features;
```
</details>

---

## Key Takeaways 💡

1. **SQL is essential** for ML data preparation
2. **JOINs** critical for combining data sources
3. **Window functions** powerful for time series features
4. **CTEs** make complex queries readable
5. **Feature engineering in SQL** faster than Python for large data
6. **Use pandas.read_sql_query()** for seamless integration
7. **Always aggregate at correct level** for your ML problem

---

**Next:** [Lesson 2 - Pandas Mastery & ETL Basics →](Lesson%202%20-%20Pandas%20Mastery%20and%20ETL%20Basics.md)

---

**Congratulations!** You now know SQL for ML! 🎉
