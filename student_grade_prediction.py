# ===================================================================
# STUDENT GRADE PREDICTION - COMPLETE PYTHON CODE
# Extracted from: Оюутны эцсийн дүнг урьдчилан таамаглах
# Team: Д.Буянжаргал, М.Төгөлдөр, Т.Мөнгөнхишиг, Б.Мөнхсаруул, Б.Анхбаяр
# ===================================================================

# ===================================================================
# SECTION 2.3 - Data Loading and Initial Exploration
# ===================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('student-mat.csv')

# Display dataset shape
print(f"Өгөгдлийн хэмжээ: {df.shape[0]} мөр, {df.shape[1]} багана")
print()

# Display first few rows
print("Эхний 5 мөр:")
print(df.head())
print()

# Statistical summary
print("Статистик үзүүлэлтүүд:")
print(df.describe())
print()

# Check for missing values
print("Дутуу утгын тоо:")
print(df.isnull().sum())
print()


# ===================================================================
# SECTION 4.1 - Target Variable Analysis (G3)
# ===================================================================

# G3 statistics
print("=" * 60)
print("G3 (Эцсийн дүн) статистик:")
print(f"Дундаж: {df['G3'].mean():.2f}")
print(f"Медиан: {df['G3'].median():.2f}")
print(f"Стандарт хазайлт: {df['G3'].std():.2f}")
print(f"Хамгийн бага: {df['G3'].min()}")
print(f"Хамгийн их: {df['G3'].max()}")
print()

# Histogram and boxplot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Histogram
axes[0].hist(df['G3'], bins=20, color='skyblue', edgecolor='black')
axes[0].set_xlabel('Эцсийн дүн (G3)')
axes[0].set_ylabel('Давтамж')
axes[0].set_title('G3-ийн тархалт')
axes[0].axvline(df['G3'].mean(), color='red', linestyle='--', 
                label=f"Дундаж: {df['G3'].mean():.2f}")
axes[0].legend()

# Boxplot
axes[1].boxplot(df['G3'])
axes[1].set_ylabel('Эцсийн дүн (G3)')
axes[1].set_title('G3-ийн boxplot')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('01_g3_distribution.png', dpi=300, bbox_inches='tight')
plt.show()


# ===================================================================
# SECTION 4.2 - Correlation Analysis
# ===================================================================

# Correlation matrix for grades
grades_corr = df[['G1', 'G2', 'G3']].corr()
print("=" * 60)
print("Дүнгийн корреляцийн матриц:")
print(grades_corr)
print()

# Correlation heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(grades_corr, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('G1, G2, G3-ийн корреляци')
plt.savefig('02_grades_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# Scatter plots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# G1 vs G3
axes[0].scatter(df['G1'], df['G3'], alpha=0.5, color='blue')
axes[0].set_xlabel('Эхний улирлын дүн (G1)')
axes[0].set_ylabel('Эцсийн дүн (G3)')
axes[0].set_title(f'G1 ба G3 (корреляци: {df["G1"].corr(df["G3"]):.2f})')
axes[0].grid(True, alpha=0.3)

# G2 vs G3
axes[1].scatter(df['G2'], df['G3'], alpha=0.5, color='green')
axes[1].set_xlabel('Хоёрдугаар улирлын дүн (G2)')
axes[1].set_ylabel('Эцсийн дүн (G3)')
axes[1].set_title(f'G2 ба G3 (корреляци: {df["G2"].corr(df["G3"]):.2f})')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('03_grades_scatter_plots.png', dpi=300, bbox_inches='tight')
plt.show()


# ===================================================================
# SECTION 4.3 - Categorical Variables Analysis
# ===================================================================

# Categorical variables boxplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Sex effect
axes[0, 0].boxplot([df[df['sex'] == 'F']['G3'], df[df['sex'] == 'M']['G3']], 
                     labels=['Эмэгтэй', 'Эрэгтэй'])
axes[0, 0].set_ylabel('Эцсийн дүн (G3)')
axes[0, 0].set_title('Хүйсний дагуух G3')
axes[0, 0].grid(True, alpha=0.3)

# School effect
axes[0, 1].boxplot([df[df['school'] == 'GP']['G3'], df[df['school'] == 'MS']['G3']], 
                     labels=['Gabriel Pereira', 'Mousinho da Silveira'])
axes[0, 1].set_ylabel('Эцсийн дүн (G3)')
axes[0, 1].set_title('Сургуулийн дагуух G3')
axes[0, 1].grid(True, alpha=0.3)

# Internet effect
axes[1, 0].boxplot([df[df['internet'] == 'no']['G3'], df[df['internet'] == 'yes']['G3']], 
                     labels=['Үгүй', 'Тийм'])
axes[1, 0].set_ylabel('Эцсийн дүн (G3)')
axes[1, 0].set_title('Интернэт холболтын дагуух G3')
axes[1, 0].grid(True, alpha=0.3)

# Higher education desire
axes[1, 1].boxplot([df[df['higher'] == 'no']['G3'], df[df['higher'] == 'yes']['G3']], 
                     labels=['Үгүй', 'Тийм'])
axes[1, 1].set_ylabel('Эцсийн дүн (G3)')
axes[1, 1].set_title('Дээд боловсрол хүсэх дагуух G3')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('04_categorical_variables.png', dpi=300, bbox_inches='tight')
plt.show()


# ===================================================================
# SECTION 4.4 - Numeric Variables Correlation
# ===================================================================

# Select numeric columns
numeric_cols = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 
                'failures', 'famrel', 'freetime', 'goout', 'Dalc', 
                'Walc', 'health', 'absences', 'G1', 'G2']

# Calculate correlation with G3
correlations = df[numeric_cols].corrwith(df['G3']).sort_values(ascending=False)

print("=" * 60)
print("G3-тай корреляци:")
print(correlations)
print()

# Bar chart
plt.figure(figsize=(10, 6))
colors = ['green' if x > 0 else 'red' for x in correlations]
plt.barh(correlations.index, correlations.values, color=colors, alpha=0.7)
plt.xlabel('Корреляцийн коэффициент')
plt.title('Тоон хувьсагчдын G3-тай корреляци')
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('05_numeric_correlations.png', dpi=300, bbox_inches='tight')
plt.show()


# ===================================================================
# SECTION 5.1 - Import Scikit-learn Libraries
# ===================================================================

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("Scikit-learn сангууд амжилттай импортлогдлоо")
print()


# ===================================================================
# SECTION 5.2 - Data Preprocessing
# ===================================================================

# Create a copy of the dataframe
df_encoded = df.copy()

# List of categorical columns
categorical_cols = ['school', 'sex', 'address', 'famsize', 'Pstatus', 
                    'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup', 
                    'famsup', 'paid', 'activities', 'nursery', 'higher', 
                    'internet', 'romantic']

# Label Encoding
le = LabelEncoder()
for col in categorical_cols:
    df_encoded[col] = le.fit_transform(df[col])

print("=" * 60)
print("Категори хувьсагчид амжилттай кодлогдлоо")
print(f"\nЖишээ: 'school' хувьсагч")
print(f"Өмнөх утгууд: {df['school'].unique()}")
print(f"Кодлогдсон утгууд: {df_encoded['school'].unique()}")
print()

# Separate features (X) and target (y)
X = df_encoded.drop('G3', axis=1)
y = df_encoded['G3']

print(f"Шинж чанаруудын тоо: {X.shape[1]}")
print(f"Оюутнуудын тоо: {X.shape[0]}")
print(f"Зорилтот хувьсагч (G3): {y.shape[0]} утга")
print()


# ===================================================================
# SECTION 5.3 - Train-Test Split
# ===================================================================

# 80% training, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("=" * 60)
print("Өгөгдөл амжилттай хуваагдлаа:")
print(f"Сургалтын олонлог: {X_train.shape[0]} оюутан")
print(f"Тестийн олонлог: {X_test.shape[0]} оюутан")
print(f"\nХувь харьцаа:")
print(f"Сургалт: {X_train.shape[0] / len(X) * 100:.1f}%")
print(f"Тест: {X_test.shape[0] / len(X) * 100:.1f}%")
print()


# ===================================================================
# SECTION 5.4 - Model Training
# ===================================================================

# Create the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

print("=" * 60)
print("Загвар амжилттай сургагдлаа!")
print(f"\nЗагварын параметрүүд:")
print(f"Intercept (β₀): {model.intercept_:.4f}")
print(f"Коэффициентүүдийн тоо: {len(model.coef_)}")
print()


# ===================================================================
# SECTION 5.5 - Feature Coefficients
# ===================================================================

# Create DataFrame with coefficients
coef_df = pd.DataFrame({
    'Хувьсагч': X.columns,
    'Коэффициент': model.coef_
}).sort_values('Коэффициент', ascending=False)

print("=" * 60)
print("Топ 10 эерэг коэффициентүүд:")
print(coef_df.head(10))
print("\nТоп 10 сөрөг коэффициентүүд:")
print(coef_df.tail(10))
print()

# Bar chart of top 15 coefficients
plt.figure(figsize=(10, 8))
top_15 = pd.concat([coef_df.head(8), coef_df.tail(7)])
colors = ['green' if x > 0 else 'red' for x in top_15['Коэффициент']]
plt.barh(top_15['Хувьсагч'], top_15['Коэффициент'], color=colors, alpha=0.7)
plt.xlabel('Коэффициентийн утга')
plt.title('Шинж чанаруудын коэффициентүүд (Топ 15)')
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('06_feature_coefficients.png', dpi=300, bbox_inches='tight')
plt.show()


# ===================================================================
# SECTION 6.1 - Predictions
# ===================================================================

# Make predictions
y_pred = model.predict(X_test)

# Compare actual vs predicted
comparison_df = pd.DataFrame({
    'Бодит утга (y_test)': y_test.values,
    'Таамаглал (y_pred)': y_pred,
    'Алдаа': y_test.values - y_pred
})

print("=" * 60)
print("Эхний 10 таамаглал:")
print(comparison_df.head(10))
print()

print(f"Дундаж алдаа: {comparison_df['Алдаа'].mean():.4f}")
print(f"Алдааны стандарт хазайлт: {comparison_df['Алдаа'].std():.4f}")
print()


# ===================================================================
# SECTION 6.2 - Model Performance
# ===================================================================

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("=" * 60)
print("Загварын үнэлгээний үзүүлэлтүүд:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# Training set R²
y_train_pred = model.predict(X_train)
r2_train = r2_score(y_train, y_train_pred)
print(f"\nСургалтын R²: {r2_train:.4f}")
print(f"Тестийн R²: {r2:.4f}")
print()

# Actual vs Predicted scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='k', linewidth=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', lw=2, label='Төгс таамаглал')
plt.xlabel('Бодит утга (y_test)')
plt.ylabel('Таамагласан утга (y_pred)')
plt.title(f'Бодит vs Таамагласан утга (R² = {r2:.3f})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('07_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
plt.show()


# ===================================================================
# SECTION 6.3 - Residual Analysis
# ===================================================================

# Calculate residuals
residuals = y_test - y_pred

# Residual plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Residuals vs Predicted values
axes[0].scatter(y_pred, residuals, alpha=0.6, edgecolors='k', linewidth=0.5)
axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[0].set_xlabel('Таамагласан утга')
axes[0].set_ylabel('Үлдэгдэл (y_test - y_pred)')
axes[0].set_title('Үлдэгдлийн тархалт')
axes[0].grid(True, alpha=0.3)

# Residuals histogram
axes[1].hist(residuals, bins=20, color='skyblue', edgecolor='black')
axes[1].set_xlabel('Үлдэгдэл')
axes[1].set_ylabel('Давтамж')
axes[1].set_title('Үлдэгдлийн histogram')
axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('08_residual_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("=" * 60)
print(f"Үлдэгдлийн дундаж: {residuals.mean():.4f}")
print(f"Үлдэгдлийн стандарт хазайлт: {residuals.std():.4f}")
print()


# ===================================================================
# SECTION 6.4 - Cross-Validation
# ===================================================================

# 5-fold cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')

print("=" * 60)
print("5-fold Cross-Validation үр дүн:")
print(f"Fold 1 R²: {cv_scores[0]:.4f}")
print(f"Fold 2 R²: {cv_scores[1]:.4f}")
print(f"Fold 3 R²: {cv_scores[2]:.4f}")
print(f"Fold 4 R²: {cv_scores[3]:.4f}")
print(f"Fold 5 R²: {cv_scores[4]:.4f}")
print(f"\nДундаж R²: {cv_scores.mean():.4f}")
print(f"Стандарт хазайлт: {cv_scores.std():.4f}")
print()

# Cross-validation scores bar chart
plt.figure(figsize=(8, 5))
folds = [f'Fold {i+1}' for i in range(5)]
plt.bar(folds, cv_scores, color='steelblue', alpha=0.7, edgecolor='black')
plt.axhline(y=cv_scores.mean(), color='r', linestyle='--', 
            linewidth=2, label=f'Дундаж: {cv_scores.mean():.3f}')
plt.xlabel('Cross-Validation Fold')
plt.ylabel('R² оноо')
plt.title('5-Fold Cross-Validation үр дүн')
plt.ylim(0, 1)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('09_cross_validation.png', dpi=300, bbox_inches='tight')
plt.show()


# ===================================================================
# FINAL SUMMARY
# ===================================================================

print("=" * 60)
print("ҮНДСЭН ҮР ДҮН:")
print("=" * 60)
print(f"1. Test R²: {r2:.4f} (Загвар эцсийн дүний {r2*100:.1f}%-ийг тайлбарлана)")
print(f"2. Cross-validation дундаж R²: {cv_scores.mean():.4f}")
print(f"3. Mean Absolute Error: {mae:.4f} оноо")
print(f"4. Хамгийн чухал хувьсагч: {coef_df.iloc[0]['Хувьсагч']} (коэффициент: {coef_df.iloc[0]['Коэффициент']:.4f})")
print(f"5. Загвар тогтвортой: CV std = {cv_scores.std():.4f}")
print("=" * 60)
print("\nБҮХ ГРАФИК ХАДГАЛАГДЛАА:")
print("01_g3_distribution.png")
print("02_grades_correlation_heatmap.png")
print("03_grades_scatter_plots.png")
print("04_categorical_variables.png")
print("05_numeric_correlations.png")
print("06_feature_coefficients.png")
print("07_actual_vs_predicted.png")
print("08_residual_analysis.png")
print("09_cross_validation.png")
print("=" * 60)
print("АМЖИЛТТАЙ ДУУСЛАА! ✅")
