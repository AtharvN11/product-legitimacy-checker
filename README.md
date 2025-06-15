# Product Legitimacy Checker

A web application that analyzes Amazon product reviews to determine the legitimacy of products and detect potential scams using a fine-tuned BERT model.

## Features

- Amazon product review scraping  
- BERT-based review classification  
- Legitimacy scoring & scam detection  
- Risk assessment per product  
- User-friendly web interface

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/product-legitimacy-checker.git
cd product-legitimacy-checker
```

2. **Create and activate a virtual environment:**
```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On macOS/Linux
source .venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Train the model (Required before first run):**
```bash
python bert_fake_review_classifier.py
```

This will train and save the BERT model used to classify reviews as real or fake.

5. **Run the application:**
```bash
python app.py
```

6. **Open your browser and navigate to:**
```
http://localhost:5000
```

## Usage

1. Paste an Amazon product URL into the input field  
2. Click **"Analyze"**  
3. Get your results:
   - Legitimacy score (0–100%)
   - Risk assessment
   - Product metadata
   - AI-based review insights

## Project Structure

```
product-legitimacy-checker/
├── data/                          # Data storage
├── models/                        # Trained BERT model and tokenizer
├── notebooks/                     # Jupyter Notebooks for experimentation
├── utils/                         # Utility functions
├── src/                           # Core logic and scripts
├── frontend/                      # HTML/CSS for the web interface
├── bert_fake_review_classifier.py# Training script for BERT
├── app.py                         # Flask application
└── api_docs/                      # Optional API documentation
```

## Dependencies

- Flask  
- BeautifulSoup4  
- Scikit-learn  
- HuggingFace Transformers  
- Requests  
- Pandas  
- Joblib  
- Torch  

## Contributing

1. Fork the repository  
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)  
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)  
4. Push to the branch (`git push origin feature/AmazingFeature`)  
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

> This tool is for **educational and research** purposes only. Please use it responsibly and respect [Amazon's terms of service](https://www.amazon.in/gp/help/customer/display.html?nodeId=508088) and `robots.txt` policies when scraping data.
