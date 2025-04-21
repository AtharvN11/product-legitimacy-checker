# Product Legitimacy Checker

A web application that analyzes Amazon product reviews to determine the legitimacy of products and detect potential scams.

## Features

- Amazon product review scraping
- ML-based review analysis
- Legitimacy scoring
- Risk assessment
- User-friendly interface

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/product-legitimacy-checker.git
cd product-legitimacy-checker
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On macOS/Linux
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python app.py
```

5. Open your browser and navigate to `http://localhost:5000`

## Usage

1. Enter an Amazon product URL in the input field
2. Click "Analyze"
3. View the detailed analysis results including:
   - Legitimacy score
   - Risk assessment
   - Product details
   - Expert analysis

## Project Structure

```
product-legitimacy-checker/
├── data/                          # Data storage
├── models/                        # Trained ML models
├── notebooks/                     # Jupyter Notebooks
├── utils/                         # Utilities
├── src/                           # Source code
├── frontend/                      # Web interface
└── api_docs/                      # API documentation
```

## Dependencies

- Flask
- BeautifulSoup4
- Scikit-learn
- Requests
- Pandas
- Joblib

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This tool is for educational purposes only. Please respect Amazon's terms of service and robots.txt when using this application. 