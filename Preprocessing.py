import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load the dataset
data = pd.read_csv("/content/Loan_Default_data.csv")
data = data.dropna(subset=['repay_fail'])

# Define target and features
y = data['repay_fail']
X = data.drop(columns=['repay_fail'])

#  NUMERICAL FEATURES 
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# If dataset contains an unnecessary index column:
if 'Unnamed: 0' in numerical_features:
    numerical_features.remove('Unnamed: 0')
print(numerical_features)
# CATEGORICAL FEATURES 
ordinal_term = ['term']
onehot_features = ['home_ownership', 'purpose', 'verification_status', 'loan_status']

# PREPROCESSOR PIPELINE (with missing value handling) 
preprocessor = ColumnTransformer(
    transformers=[
        # Numerical --> Impute median --> Scale
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', MinMaxScaler())
        ]), numerical_features),

        # Ordinal category --> Impute most frequent --> Encode
        ('ordinal_term', Pipeline(steps=[
        ('encoder', OrdinalEncoder(categories=[['36 months', '60 months']]))
        ]), ordinal_term),

        # One-hot categories --> Impute most frequent --> OneHot
        ('onehot', Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), onehot_features),
    ],
    remainder='drop'
)

# Create final pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

# Fit and transform data
X_preprocessed = pipeline.fit_transform(X)

# CREATE DATAFRAME WITH COLUMN NAMES 
onehot_names = pipeline.named_steps['preprocessor'].named_transformers_['onehot'].named_steps['encoder'].get_feature_names_out(onehot_features)
ordinal_names = ['term_encoded']
all_feature_names = numerical_features + ordinal_names + list(onehot_names)

X_final = pd.DataFrame(X_preprocessed, columns=all_feature_names)

X_final.head()
 # for check null value
print(y.isna().sum())