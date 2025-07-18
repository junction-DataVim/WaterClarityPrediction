# Water Quality Prediction System

A machine learning system for predicting water quality based on physicochemical
parameters. This project uses various ML algorithms to classify water quality
into three categories: Excellent, Good, and Poor.

## 📁 Project Structure

```
Water clarity/
├── README.md                   # This file
├── Water-Clarity-DS.csv       # Dataset containing water quality measurements
├── test.ipynb                 # Jupyter notebook for model development and training
├── water_quality_model.pkl    # Trained machine learning model (serialized)
├── feature_names.json         # List of feature names used by the model
├── class_labels.json          # Mapping of class indices to quality labels
├── predict.py                 # Standalone prediction script
├── api.py                     # API endpoint for model serving
└── __pycache__/              # Python cache files
    └── api.cpython-311.pyc
```

## 🔬 Features

The model uses the following water quality parameters to make predictions:

- **Temperature** (°C)
- **Turbidity** (cm)
- **Dissolved Oxygen** (mg/L)
- **BOD** (Biological Oxygen Demand, mg/L)
- **pH** value
- **Ammonia** concentration (mg/L)
- **Nitrite** concentration (mg/L)

## 🎯 Model Output

The system predicts water quality in three categories:

- **0**: Excellent water quality
- **1**: Good water quality
- **2**: Poor water quality

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn joblib matplotlib seaborn
```

### Quick Start

1. **Using the prediction script directly:**

```python
from predict import predict_water_quality

# Example prediction
result = predict_water_quality(
    temp=67.45,
    turbidity=10.13,
    do=0.208,
    bod=7.474,
    ph=4.752,
    ammonia=0.286,
    nitrite=4.355
)

print(f"Water Quality: {result['quality']}")
print(f"Confidence: {result['probabilities']}")
```

2. **Running the API server:**

```bash
python api.py
```

3. **Training your own model:**
   - Open `test.ipynb` in Jupyter Notebook
   - Run all cells to train and evaluate different models
   - The best model will be automatically saved

## 📊 Dataset Information

The dataset (`Water-Clarity-DS.csv`) contains water quality measurements with:

- European decimal format (comma-separated)
- Multiple physicochemical parameters
- Balanced sampling across quality categories

## 🤖 Model Development

### Available Models

The system evaluates multiple machine learning algorithms:

- Random Forest
- Gradient Boosting
- Support Vector Machine (SVM)
- Logistic Regression
- K-Nearest Neighbors

### Model Selection Process

1. **Data Preprocessing**: Handle European decimal format, balance classes
2. **Model Evaluation**: Cross-validation with stratified sampling
3. **Hyperparameter Tuning**: Grid search for optimal parameters
4. **Performance Analysis**: Classification reports and confusion matrices

### Key Functions in `test.ipynb`

- `load_and_preprocess_data()`: Load and clean the dataset
- `balance_classes()`: Balance the dataset for fair training
- `evaluate_models()`: Compare different ML algorithms
- `optimize_best_model()`: Hyperparameter tuning for best model
- `analyze_feature_importance()`: Understand feature contributions

## 📈 Model Performance

The system automatically selects the best performing model based on
cross-validation scores. Typical performance metrics include:

- Accuracy scores for each class
- F1-scores per quality category
- Confusion matrix analysis
- Feature importance rankings

## 🔧 API Usage

The API provides a REST endpoint for making predictions:

```python
# Example API call structure
POST /predict
{
    "temp": 67.45,
    "turbidity": 10.13,
    "do": 0.208,
    "bod": 7.474,
    "ph": 4.752,
    "ammonia": 0.286,
    "nitrite": 4.355
}
```

## 📝 Model Artifacts

The system generates several important files:

- `water_quality_model.pkl`: Serialized trained model
- `feature_names.json`: Feature names in correct order
- `class_labels.json`: Quality label mappings

## 🔍 Feature Importance

The model analyzes which parameters are most important for water quality
prediction. This helps understand:

- Which measurements have the strongest impact on water quality
- How different parameters contribute to the final classification
- Insights for water quality monitoring priorities

## 🎨 Visualization

The Jupyter notebook includes:

- Data distribution plots
- Model performance comparisons
- Feature importance visualizations
- Confusion matrix heatmaps

## 📋 Usage Examples

### 1. Batch Predictions

```python
import pandas as pd
from predict import predict_water_quality

# Load your data
data = pd.read_csv('new_water_samples.csv')

# Make predictions for each row
predictions = []
for _, row in data.iterrows():
    result = predict_water_quality(
        temp=row['temp'],
        turbidity=row['turbidity'],
        do=row['do'],
        bod=row['bod'],
        ph=row['ph'],
        ammonia=row['ammonia'],
        nitrite=row['nitrite']
    )
    predictions.append(result['quality'])

data['predicted_quality'] = predictions
```

### 2. Real-time Monitoring

```python
# Integrate with sensor data
def monitor_water_quality(sensor_data):
    result = predict_water_quality(**sensor_data)

    if result['prediction'] == 2:  # Poor quality
        send_alert(f"Poor water quality detected: {result['quality']}")

    return result
```

## 🛠️ Development

### Extending the Model

1. Add new features to the dataset
2. Update `feature_names.json` accordingly
3. Retrain the model using `test.ipynb`
4. Test with new prediction scripts

### Improving Performance

1. Collect more training data
2. Experiment with feature engineering
3. Try advanced algorithms (XGBoost, Neural Networks)
4. Implement ensemble methods

## 📊 Model Evaluation Metrics

The system provides comprehensive evaluation:

- **Cross-validation scores**: Generalization performance
- **Test accuracy**: Final model performance
- **Per-class metrics**: Precision, recall, F1-score
- **Confusion matrix**: Classification details

## 🔄 Retraining

To retrain the model with new data:

1. Update `Water-Clarity-DS.csv` with new samples
2. Run the complete pipeline in `test.ipynb`
3. New model artifacts will be automatically saved
4. Update prediction scripts if needed

## 💡 Best Practices

1. **Data Quality**: Ensure measurements are accurate and consistent
2. **Feature Scaling**: The system handles scaling automatically
3. **Class Balance**: Dataset balancing is implemented for fair training
4. **Validation**: Always validate predictions with ground truth when possible

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Add tests for new functionality
4. Update documentation
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🆘 Support

For issues or questions:

1. Check the Jupyter notebook for detailed examples
2. Review the prediction script for usage patterns
3. Examine the API code for integration examples

---

**Note**: This system is designed for educational and research purposes. For
production water quality monitoring, please validate against certified
laboratory measurements and follow local regulations.
