# Lesson 2: Pandas Mastery & ETL Basics 🐼

**Module 3: Data Engineering | Lesson 2 of 4**

Master advanced Pandas operations and ETL pipelines for production ML!

---

## Why Pandas Mastery?

**Pandas is the workhorse of ML data processing:**
- 95% of data manipulation before modeling
- Fast, vectorized operations
- Seamless integration with ML libraries
- Essential for data cleaning and feature engineering

---

## 1. Advanced Data Loading 📥

### Multiple File Formats

```python
import pandas as pd
import numpy as np

# CSV
df = pd.read_csv('data.csv')
df = pd.read_csv('data.csv', sep='\t', encoding='utf-8', na_values=['NA', 'missing'])

# Excel
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')

# JSON
df = pd.read_json('data.json')
df = pd.read_json('data.json', lines=True)  # JSONL format

# Parquet (fast, compressed)
df = pd.read_parquet('data.parquet')

# SQL
import sqlite3
conn = sqlite3.connect('database.db')
df = pd.read_sql_query("SELECT * FROM table", conn)

# Multiple files at once
import glob
files = glob.glob('data/*.csv')
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
```

### Chunked Reading (Large Files)

```python
# Read in chunks (memory-efficient)
chunk_size = 10000
chunks = []

for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    # Process chunk
    chunk_processed = chunk[chunk['age'] > 18]
    chunks.append(chunk_processed)

df = pd.concat(chunks, ignore_index=True)

# Or process iteratively
for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    # Aggregate on-the-fly
    print(f"Chunk mean: {chunk['value'].mean()}")
```

---

## 2. Data Inspection & Quality Checks 🔍

### Quick Overview

```python
# Basic info
print(df.shape)           # (rows, cols)
print(df.dtypes)          # Column types
print(df.info())          # Memory usage, non-null counts
print(df.describe())      # Statistics
print(df.head(10))        # First 10 rows
print(df.sample(5))       # Random 5 rows

# Missing values
print(df.isnull().sum())
print(df.isnull().mean() * 100)  # Percentage

# Duplicates
print(f"Duplicates: {df.duplicated().sum()}")
print(df[df.duplicated(keep=False)])  # Show all duplicates

# Value counts
print(df['column'].value_counts())
print(df['column'].value_counts(normalize=True))  # Percentages
```

### Data Quality Report

```python
def data_quality_report(df):
    """Comprehensive data quality check"""
    report = pd.DataFrame({
        'Column': df.columns,
        'Type': df.dtypes,
        'Non-Null': df.notnull().sum(),
        'Null': df.isnull().sum(),
        'Null%': (df.isnull().sum() / len(df) * 100).round(2),
        'Unique': df.nunique(),
        'Unique%': (df.nunique() / len(df) * 100).round(2)
    }).reset_index(drop=True)

    return report

print(data_quality_report(df))
```

---

## 3. Advanced Indexing & Selection 🎯

### Boolean Indexing

```python
# Single condition
df[df['age'] > 30]

# Multiple conditions
df[(df['age'] > 30) & (df['city'] == 'NYC')]
df[(df['age'] < 20) | (df['age'] > 65)]

# NOT condition
df[~df['city'].isin(['NYC', 'LA'])]

# String operations
df[df['name'].str.contains('John', case=False)]
df[df['email'].str.endswith('@gmail.com')]

# Query method (more readable)
df.query('age > 30 and city == "NYC"')
df.query('age > @min_age')  # Use variables
```

### loc vs iloc

```python
# loc: label-based
df.loc[0:5, 'name']                    # Rows 0-5, column 'name'
df.loc[df['age'] > 30, ['name', 'age']]  # Conditional selection

# iloc: position-based
df.iloc[0:5, 0]                        # First 5 rows, first column
df.iloc[:, [0, 2, 4]]                  # All rows, columns 0, 2, 4

# at/iat: Fast scalar access
df.at[0, 'name']                       # Single value by label
df.iat[0, 0]                           # Single value by position
```

---

## 4. Data Transformation 🔄

### Apply, Map, Transform

```python
# apply: Apply function to column/row
df['age_squared'] = df['age'].apply(lambda x: x ** 2)

# Apply to multiple columns
df[['col1', 'col2']].apply(np.sum, axis=0)  # Column-wise
df.apply(lambda row: row['col1'] + row['col2'], axis=1)  # Row-wise

# map: Element-wise mapping (Series only)
mapping = {'M': 'Male', 'F': 'Female'}
df['gender_full'] = df['gender'].map(mapping)

# transform: Return same-shaped result
df['age_normalized'] = df.groupby('city')['age'].transform(
    lambda x: (x - x.mean()) / x.std()
)

# applymap: Element-wise on entire DataFrame (deprecated, use map)
df = df.map(lambda x: x.lower() if isinstance(x, str) else x)
```

### String Operations

```python
# Vectorized string methods
df['name_upper'] = df['name'].str.upper()
df['name_lower'] = df['name'].str.lower()
df['first_name'] = df['name'].str.split().str[0]

# Extract patterns
df['email_domain'] = df['email'].str.extract(r'@(\w+\.\w+)')
df['phone_area'] = df['phone'].str.extract(r'(\d{3})-')

# Replace
df['text'] = df['text'].str.replace('old', 'new')
df['text'] = df['text'].str.replace(r'\d+', '', regex=True)  # Remove digits

# String checks
df['has_number'] = df['text'].str.contains(r'\d', regex=True)
df['is_email'] = df['text'].str.match(r'^[\w\.-]+@[\w\.-]+\.\w+$')
```

### DateTime Operations

```python
# Convert to datetime
df['date'] = pd.to_datetime(df['date_str'])
df['date'] = pd.to_datetime(df['date_str'], format='%Y-%m-%d')

# Extract components
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['day_of_week'] = df['date'].dt.dayofweek
df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6])
df['quarter'] = df['date'].dt.quarter

# Time differences
df['days_since'] = (pd.Timestamp.now() - df['date']).dt.days

# Resampling time series
df.set_index('date').resample('D').mean()   # Daily
df.set_index('date').resample('W').sum()    # Weekly
df.set_index('date').resample('M').count()  # Monthly
```

---

## 5. GroupBy Mastery 📊

### Basic Aggregations

```python
# Single aggregation
df.groupby('city')['age'].mean()

# Multiple aggregations
df.groupby('city').agg({
    'age': ['mean', 'median', 'std'],
    'income': ['sum', 'mean'],
    'customer_id': 'count'
})

# Named aggregations
df.groupby('city').agg(
    avg_age=('age', 'mean'),
    total_income=('income', 'sum'),
    customer_count=('customer_id', 'count')
).reset_index()

# Multiple grouping columns
df.groupby(['city', 'gender'])['income'].mean()
```

### Advanced GroupBy

```python
# Custom aggregation functions
def range_func(x):
    return x.max() - x.min()

df.groupby('city')['age'].agg([np.mean, np.std, range_func])

# Transform (keep original shape)
df['age_group_mean'] = df.groupby('city')['age'].transform('mean')
df['age_deviation'] = df['age'] - df['age_group_mean']

# Filter groups
df.groupby('city').filter(lambda x: len(x) > 100)  # Only cities with >100 customers

# Apply (most flexible)
df.groupby('city').apply(lambda x: x.nlargest(3, 'income'))
```

### Window Functions (Rolling)

```python
# Rolling statistics
df['ma_7'] = df['value'].rolling(window=7).mean()
df['std_7'] = df['value'].rolling(window=7).std()

# Rolling with time window
df.set_index('date')['value'].rolling('7D').mean()

# Expanding (cumulative)
df['cumsum'] = df['value'].expanding().sum()
df['cummean'] = df['value'].expanding().mean()

# Shift (lagged values)
df['prev_value'] = df['value'].shift(1)
df['next_value'] = df['value'].shift(-1)
df['diff'] = df['value'] - df['value'].shift(1)
```

---

## 6. Merging & Joining 🔗

### Different Join Types

```python
# Inner join (default)
merged = pd.merge(df1, df2, on='customer_id', how='inner')

# Left join
merged = pd.merge(df1, df2, on='customer_id', how='left')

# Outer join (all rows from both)
merged = pd.merge(df1, df2, on='customer_id', how='outer')

# Different column names
merged = pd.merge(df1, df2, left_on='id', right_on='customer_id')

# Multiple keys
merged = pd.merge(df1, df2, on=['customer_id', 'date'])

# Suffix for overlapping columns
merged = pd.merge(df1, df2, on='id', suffixes=('_left', '_right'))
```

### Concat & Append

```python
# Concatenate vertically (stack rows)
combined = pd.concat([df1, df2, df3], ignore_index=True)

# Concatenate horizontally (add columns)
combined = pd.concat([df1, df2], axis=1)

# With keys
combined = pd.concat([df1, df2], keys=['batch1', 'batch2'])
```

---

## 7. Pivot Tables & Reshaping 🔄

### Pivot Tables

```python
# Basic pivot
pivot = df.pivot_table(
    values='amount',
    index='customer_id',
    columns='product_category',
    aggfunc='sum',
    fill_value=0
)

# Multiple aggregations
pivot = df.pivot_table(
    values='amount',
    index='customer_id',
    columns='month',
    aggfunc=['sum', 'count', 'mean']
)

# Multiple values
pivot = df.pivot_table(
    values=['amount', 'quantity'],
    index='customer_id',
    columns='category',
    aggfunc='sum'
)
```

### Melting (Wide to Long)

```python
# Wide format
wide_df = pd.DataFrame({
    'customer_id': [1, 2, 3],
    'jan_sales': [100, 200, 150],
    'feb_sales': [120, 180, 160],
    'mar_sales': [140, 220, 170]
})

# Melt to long format
long_df = wide_df.melt(
    id_vars=['customer_id'],
    value_vars=['jan_sales', 'feb_sales', 'mar_sales'],
    var_name='month',
    value_name='sales'
)
```

---

## 8. Performance Optimization ⚡

### Efficient Data Types

```python
# Reduce memory usage
def optimize_dtypes(df):
    # Integers
    for col in df.select_dtypes(include=['int']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')

    # Floats
    for col in df.select_dtypes(include=['float']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')

    # Categories (for low-cardinality string columns)
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() / len(df) < 0.5:  # <50% unique
            df[col] = df[col].astype('category')

    return df

# Before
print(f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

df = optimize_dtypes(df)

# After
print(f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
```

### Vectorization over Loops

```python
# BAD: Loop
result = []
for i, row in df.iterrows():
    result.append(row['a'] + row['b'])
df['c'] = result

# GOOD: Vectorized
df['c'] = df['a'] + df['b']

# GOOD: NumPy
df['c'] = np.where(df['a'] > 0, df['a'] + df['b'], 0)
```

### Using eval and query

```python
# Faster for complex expressions
df['result'] = df.eval('a + b * c - d')

# Multiple operations
df.eval("""
    result1 = a + b
    result2 = c * d
    final = result1 - result2
""", inplace=True)
```

---

## 9. ETL Pipeline Basics 🔄

### Extract-Transform-Load Pattern

```python
def extract_data(source):
    """Extract data from various sources"""
    if source.endswith('.csv'):
        return pd.read_csv(source)
    elif source.endswith('.parquet'):
        return pd.read_parquet(source)
    elif source.startswith('postgresql://'):
        return pd.read_sql_query("SELECT * FROM table", source)

def transform_data(df):
    """Apply transformations"""
    # 1. Data cleaning
    df = df.drop_duplicates()
    df = df.dropna(subset=['essential_column'])

    # 2. Type conversions
    df['date'] = pd.to_datetime(df['date'])
    df['category'] = df['category'].astype('category')

    # 3. Feature engineering
    df['month'] = df['date'].dt.month
    df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6])

    # 4. Aggregations
    df = df.groupby('customer_id').agg({
        'amount': ['sum', 'mean', 'count'],
        'date': 'max'
    }).reset_index()

    return df

def load_data(df, destination):
    """Load data to destination"""
    if destination.endswith('.csv'):
        df.to_csv(destination, index=False)
    elif destination.endswith('.parquet'):
        df.to_parquet(destination, index=False)
    elif destination.startswith('postgresql://'):
        df.to_sql('table_name', destination, if_exists='replace')

# Complete ETL pipeline
def etl_pipeline(source, destination):
    print("Extracting...")
    df = extract_data(source)

    print(f"Extracted {len(df)} rows")

    print("Transforming...")
    df = transform_data(df)

    print(f"Transformed to {len(df)} rows")

    print("Loading...")
    load_data(df, destination)

    print("ETL Complete!")

# Run pipeline
etl_pipeline('data/raw.csv', 'data/processed.parquet')
```

### Incremental ETL

```python
def incremental_etl(source, destination, checkpoint_file='checkpoint.txt'):
    """Process only new data since last run"""

    # Load checkpoint
    try:
        with open(checkpoint_file, 'r') as f:
            last_processed = pd.to_datetime(f.read().strip())
    except FileNotFoundError:
        last_processed = pd.Timestamp('1900-01-01')

    print(f"Last processed: {last_processed}")

    # Extract only new data
    df = pd.read_csv(source)
    df['date'] = pd.to_datetime(df['date'])
    new_data = df[df['date'] > last_processed]

    if len(new_data) == 0:
        print("No new data")
        return

    print(f"Processing {len(new_data)} new rows")

    # Transform
    new_data = transform_data(new_data)

    # Load (append mode)
    if os.path.exists(destination):
        existing = pd.read_parquet(destination)
        combined = pd.concat([existing, new_data], ignore_index=True)
        combined.to_parquet(destination, index=False)
    else:
        new_data.to_parquet(destination, index=False)

    # Update checkpoint
    with open(checkpoint_file, 'w') as f:
        f.write(str(df['date'].max()))

    print("Incremental ETL Complete!")
```

### Data Validation

```python
def validate_data(df, schema):
    """Validate data against schema"""
    errors = []

    # Check required columns
    missing_cols = set(schema['required_columns']) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")

    # Check data types
    for col, expected_type in schema['types'].items():
        if col in df.columns:
            if df[col].dtype != expected_type:
                errors.append(f"{col}: expected {expected_type}, got {df[col].dtype}")

    # Check value ranges
    for col, (min_val, max_val) in schema.get('ranges', {}).items():
        if col in df.columns:
            invalid = df[(df[col] < min_val) | (df[col] > max_val)]
            if len(invalid) > 0:
                errors.append(f"{col}: {len(invalid)} values out of range [{min_val}, {max_val}]")

    # Check nulls
    for col in schema.get('not_null', []):
        if col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                errors.append(f"{col}: {null_count} null values found")

    return errors

# Define schema
schema = {
    'required_columns': ['customer_id', 'amount', 'date'],
    'types': {
        'customer_id': 'int64',
        'amount': 'float64',
        'date': 'datetime64[ns]'
    },
    'ranges': {
        'amount': (0, 100000),
        'age': (0, 120)
    },
    'not_null': ['customer_id', 'date']
}

# Validate
errors = validate_data(df, schema)
if errors:
    print("Validation errors:")
    for error in errors:
        print(f"  - {error}")
else:
    print("Validation passed!")
```

---

## Quick Reference 📖

**Essential Pandas Operations:**

```python
# Loading
df = pd.read_csv('file.csv')

# Inspection
df.shape, df.info(), df.describe()

# Selection
df['col'], df[['col1', 'col2']], df[df['col'] > 5]

# Transformation
df['new'] = df['col'].apply(func)
df.groupby('col').agg({'col2': 'mean'})

# Joining
pd.merge(df1, df2, on='key')

# Saving
df.to_csv('output.csv', index=False)
df.to_parquet('output.parquet')
```

---

## Practice Exercises 🏋️

1. Load CSV, clean missing values, optimize dtypes
2. Create RFM features using groupby and transform
3. Build ETL pipeline with validation
4. Optimize slow pandas code using vectorization

<details>
<summary>Solutions</summary>

```python
# 1. Load and optimize
df = pd.read_csv('data.csv')
df = df.drop_duplicates()
df = df.fillna({'age': df['age'].median(), 'city': 'Unknown'})
df = optimize_dtypes(df)

# 2. RFM features
from datetime import datetime

current_date = pd.Timestamp.now()
rfm = df.groupby('customer_id').agg({
    'order_date': lambda x: (current_date - x.max()).days,  # Recency
    'order_id': 'count',                                     # Frequency
    'amount': 'sum'                                          # Monetary
}).rename(columns={
    'order_date': 'recency',
    'order_id': 'frequency',
    'amount': 'monetary'
})

# 3. ETL with validation
def etl_with_validation(source, dest):
    df = pd.read_csv(source)
    errors = validate_data(df, schema)
    if errors:
        raise ValueError(f"Validation failed: {errors}")
    df = transform_data(df)
    df.to_parquet(dest, index=False)

# 4. Optimize
# Slow
result = []
for i, row in df.iterrows():
    if row['a'] > 0:
        result.append(row['a'] * 2)
    else:
        result.append(0)

# Fast
result = np.where(df['a'] > 0, df['a'] * 2, 0)
```
</details>

---

## Key Takeaways 💡

1. **Master groupby** - most powerful Pandas operation
2. **Vectorize** instead of loops (100x faster)
3. **Optimize dtypes** to reduce memory
4. **Use categories** for low-cardinality strings
5. **ETL pipelines** should include validation
6. **Incremental processing** for large data
7. **Read data in chunks** if memory-constrained

---

**Next:** [Lesson 3 - Class Imbalance & SMOTE →](Lesson%203%20-%20Class%20Imbalance%20and%20SMOTE.md)

---

**Congratulations!** You've mastered Pandas for ML! 🎉
