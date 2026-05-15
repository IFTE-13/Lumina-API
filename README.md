# Lumina Malware Detection API

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)](https://fastapi.tiangolo.com/)

Lumina is a machine learning-powered malware detection system that analyzes Windows PE files. It extracts 50+ static features and uses a LightGBM classifier to detect malware with 99.9% accuracy.

## Model Architecture
<img width="1042" height="761" alt="Integrated System Workflow of the Proposed Suspicious File Detection System drawio" src="https://github.com/user-attachments/assets/4abf2f35-3d90-41f1-874e-393a338819ab" />

## Features

- **Fast Analysis**: < 2 seconds per file
- **Static Analysis**: No file execution required
- **REST API**: Easy integration
- **Batch Processing**: Analyze multiple files
- **Detailed Results**: Confidence scores & probability breakdowns

## Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/IFTE-13/Lumina-API.git
cd Lumina-API

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Dataset
```bash
# Collect benign Windows executables
python setup_benign_data.py

# Download malicious dataset from:
# https://data.mendeley.com/datasets/vnj7sxkt53/1
# Place Header.csv in dataset/

# Build balanced dataset
python build_dataset.py
```

### 3. Train Model
```bash
python train_model.py
```

This creates:
- malware_model.pkl - Trained model
- scaler.pkl - Feature scaler
- feature_columns.pkl - Feature names

### 4. Run API Server
```bash
python app.py
```
Server runs at http://localhost:8000

## API Endpoints
| Method | Endpoint   | Description         |
| ------ | ---------- | ------------------- |
| `GET`  | `/`        | API info            |
| `GET`  | `/health`  | Health check        |
| `POST` | `/predict` | Analyze `.exe` file |

### Test the API
```bash
# Health check
curl http://localhost:8000/health

# Analyze a file
curl -X POST http://localhost:8000/predict -F "file=@sample.exe"
```

### Response Example
```bash
{
  "filename": "sample.exe",
  "verdict": "MALICIOUS",
  "confidence": 95.5,
  "probability_benign": 0.045,
  "probability_malicious": 0.955
}
```

### Testing
```bash
# Test with EICAR (rename to .exe first)
curl -X POST http://localhost:8000/predict -F "file=@eicar.exe"

# Test with legitimate Windows executable
curl -X POST http://localhost:8000/predict -F "file=@notepad.exe"
```

### Model Performance
| Metric    | Score |
| --------- | ----- |
| Accuracy  | 97.1% |
| Precision | 95.4% |
| Recall    | 97.8% |
| F1 Score  | 97.1% |

## Requirements
- Python 3.10+
- 4GB RAM minimum
- Windows OS (for benign sample collection)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (git checkout -b feature/AmazingFeature)
3. Commit your changes (git commit -m 'Add some AmazingFeature')
4. Push to the branch (git push origin feature/AmazingFeature)
5. Open a Pull Request

## Credits
- PEFile - PE parsing
- LightGBM - ML framework
- Mendeley Dataset - Training data

## Author
MOHAMMED IFTEKHAR

GitHub: [@IFTE-13](https://github.com/IFTE-13)
