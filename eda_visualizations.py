# =========================================
# DigiCow Challenge - Exploratory Data Analysis
# =========================================

import pandas as pd
import numpy as np
# Set matplotlib backend to non-interactive
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")

# ------------------------
# Load data
# ------------------------
train = pd.read_csv("data/Train.csv")

print("=" * 60)
print("DIGICOW FARMER TRAINING ADOPTION - EDA VISUALIZATIONS")
print("=" * 60)

# ------------------------
# Target variable analysis
# ------------------------
print("\n🎯 TARGET VARIABLE ANALYSIS:")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('DigiCow Farmer Training Adoption - EDA Analysis', fontsize=16, fontweight='bold')

# Target distribution
target_counts = train['adopted_within_07_days'].value_counts()
axes[0, 0].pie(target_counts.values, labels=['Not Adopted (0)', 'Adopted (1)'], 
               autopct='%1.1f%%', startangle=90, colors=['#ff9999', '#66b3ff'])
axes[0, 0].set_title('Target Distribution')

# Adoption rate by gender
adoption_by_gender = train.groupby('gender')['adopted_within_07_days'].mean() * 100
sns.barplot(x=adoption_by_gender.index, y=adoption_by_gender.values, ax=axes[0, 1])
axes[0, 1].set_title('Adoption Rate by Gender')
axes[0, 1].set_ylabel('Adoption Rate (%)')

# Adoption rate by age
adoption_by_age = train.groupby('age')['adopted_within_07_days'].mean() * 100
sns.barplot(x=adoption_by_age.index, y=adoption_by_age.values, ax=axes[1, 0])
axes[1, 0].set_title('Adoption Rate by Age Group')
axes[1, 0].set_ylabel('Adoption Rate (%)')

# Adoption rate by registration type
adoption_by_reg = train.groupby('registration')['adopted_within_07_days'].mean() * 100
sns.barplot(x=adoption_by_reg.index, y=adoption_by_reg.values, ax=axes[1, 1])
axes[1, 1].set_title('Adoption Rate by Registration Type')
axes[1, 1].set_ylabel('Adoption Rate (%)')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('target_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# ------------------------
# Numeric features distribution
# ------------------------
print("\n📊 NUMERIC FEATURES DISTRIBUTION:")

numeric_features = [
    'belong_to_cooperative', 'num_trainings_30d', 'num_trainings_60d', 
    'num_total_trainings', 'num_repeat_trainings', 'days_to_second_training',
    'num_unique_trainers', 'has_second_training'
]

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('Numeric Features Distribution by Adoption Status', fontsize=16, fontweight='bold')

for i, feature in enumerate(numeric_features):
    row, col = i // 4, i % 4
    if feature in train.columns:
        # Plot distribution for adopted vs not adopted
        adopted = train[train['adopted_within_07_days'] == 1][feature].dropna()
        not_adopted = train[train['adopted_within_07_days'] == 0][feature].dropna()
        
        axes[row, col].hist([adopted, not_adopted], bins=20, alpha=0.7, 
                           label=['Adopted', 'Not Adopted'], color=['#66b3ff', '#ff9999'])
        axes[row, col].set_title(feature.replace('_', ' ').title())
        axes[row, col].legend()
        axes[row, col].set_xlabel('Value')
        axes[row, col].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('numeric_distributions.png', dpi=300, bbox_inches='tight')
plt.close()

# ------------------------
# Categorical features analysis
# ------------------------
print("\n🏷️ CATEGORICAL FEATURES ANALYSIS:")

# Top counties by adoption rate
county_adoption = train.groupby('county')['adopted_within_07_days'].agg(['mean', 'count'])
county_adoption = county_adoption[county_adoption['count'] >= 50]  # Filter counties with sufficient data
county_adoption = county_adoption.sort_values('mean', ascending=False).head(10)

plt.figure(figsize=(12, 6))
sns.barplot(x=county_adoption.index, y=county_adoption['mean'] * 100)
plt.title('Top 10 Counties by Adoption Rate (min 50 farmers)')
plt.ylabel('Adoption Rate (%)')
plt.xlabel('County')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('county_adoption.png', dpi=300, bbox_inches='tight')
plt.close()

# Top trainers by adoption rate
trainer_adoption = train.groupby('trainer')['adopted_within_07_days'].agg(['mean', 'count'])
trainer_adoption = trainer_adoption[trainer_adoption['count'] >= 20]  # Filter trainers with sufficient data
trainer_adoption = trainer_adoption.sort_values('mean', ascending=False).head(10)

plt.figure(figsize=(12, 6))
sns.barplot(x=trainer_adoption.index, y=trainer_adoption['mean'] * 100)
plt.title('Top 10 Trainers by Adoption Rate (min 20 farmers)')
plt.ylabel('Adoption Rate (%)')
plt.xlabel('Trainer')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('trainer_adoption.png', dpi=300, bbox_inches='tight')
plt.close()

# ------------------------
# Correlation analysis
# ------------------------
print("\n🔗 CORRELATION ANALYSIS:")

# Create correlation matrix for numeric features
numeric_cols = ['belong_to_cooperative', 'num_trainings_30d', 'num_trainings_60d', 
                'num_total_trainings', 'num_repeat_trainings', 'days_to_second_training',
                'num_unique_trainers', 'has_second_training', 'adopted_within_07_days']

correlation_data = train[numeric_cols].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_data, annot=True, cmap='coolwarm', center=0, 
            square=True, fmt='.2f')
plt.title('Correlation Matrix - Numeric Features')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# Correlation with target
target_correlation = correlation_data['adopted_within_07_days'].drop('adopted_within_07_days').sort_values(ascending=False)

plt.figure(figsize=(10, 6))
target_correlation.plot(kind='bar')
plt.title('Feature Correlation with Target (Adoption)')
plt.ylabel('Correlation Coefficient')
plt.xlabel('Features')
plt.xticks(rotation=45)
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.tight_layout()
plt.savefig('target_correlation.png', dpi=300, bbox_inches='tight')
plt.close()

# ------------------------
# Topics analysis
# ------------------------
print("\n📚 TOPICS LIST ANALYSIS:")

# Extract number of topics
train['num_topics'] = train['topics_list'].apply(lambda x: len(str(x).split(',')) if pd.notnull(x) else 0)

# Adoption rate by number of topics
topics_adoption = train.groupby('num_topics')['adopted_within_07_days'].mean() * 100

plt.figure(figsize=(12, 6))
sns.barplot(x=topics_adoption.index, y=topics_adoption.values)
plt.title('Adoption Rate by Number of Topics Covered')
plt.ylabel('Adoption Rate (%)')
plt.xlabel('Number of Topics')
plt.tight_layout()
plt.savefig('topics_adoption.png', dpi=300, bbox_inches='tight')
plt.close()

# ------------------------
# Key observations summary
# ------------------------
print("\n" + "=" * 60)
print("KEY OBSERVATIONS FOR FEATURE ENGINEERING:")
print("=" * 60)

print("\n🎯 Target Imbalance:")
print(f"- Only {target_counts[1]/len(train)*100:.1f}% of farmers adopt within 7 days")
print("- Need to handle class imbalance in modeling")

print("\n📊 Numeric Feature Insights:")
for feat, corr in target_correlation.items():
    direction = "positive" if corr > 0 else "negative"
    strength = "strong" if abs(corr) > 0.3 else "moderate" if abs(corr) > 0.1 else "weak"
    print(f"- {feat}: {strength} {direction} correlation ({corr:.3f})")

print("\n🏷️ Categorical Feature Insights:")
print("- Gender shows different adoption patterns")
print("- Age groups have varying adoption rates")
print("- Registration type impacts adoption")
print("- County and trainer effects are significant")

print("\n📚 Topics Analysis:")
print(f"- Farmers receive {train['num_topics'].mean():.1f} topics on average")
print("- More topics may correlate with higher adoption")

print("\n🔗 Feature Relationships:")
print("- Training counts are highly correlated with each other")
print("- Days to second training shows relationship with adoption")
print("- Number of unique trainers matters")

print("\n💡 Feature Engineering Recommendations:")
print("1. Create interaction features (county × trainer, age × gender)")
print("2. Extract date features from first_training_date")
print("3. Use number of topics as a feature")
print("4. Consider target encoding for high-cardinality features")
print("5. Handle missing values in days_to_second_training appropriately")

print("=" * 60)
