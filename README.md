This project files include Django backend configurations, templates for frontend rendering, and resources like datasets, visualizations, and custom modules for scraping and sentiment analysis.


# Sentimap - Social Media Sentiment Analysis

Sentimap is a Django-based application designed for analyzing sentiment in social media data. The project employs web scraping and machine learning to generate insights, including sentiment distributions, word clouds, and more. The results are visualized with interactive charts for an intuitive user experience.

## Features

- **Real-time Sentiment Analysis**: Analyze sentiments from social media data.
- **Data Visualization**: Includes bar plots, word clouds, and sentiment pyramids.
- **Custom Web Scraper**: Built-in scraper for gathering data from various platforms.
- **Django Backend**: Robust backend for managing user inputs and visualizations.
- **Media Gallery**: Stores generated plots and images for user reference.

## Project Structure

### Key Directories and Files

- **SENTIMAP**: Django project settings and configurations.
  - `settings.py`: Configures the project.
  - `urls.py`: Routes URLs to the appropriate views.
  - `wsgi.py`: Entry point for deploying the application.

- **sentimentanalysis**: Core app for sentiment analysis.
  - `models.py`: Defines database models.
  - `views.py`: Contains logic for processing user requests and rendering templates.
  - `scrape.py`: Custom scraper for gathering social media data.
  - `templates/`: HTML templates for the frontend.
    - `sentianalysis.html`: Main interface for analysis.
    - `newreport.html`: Displays detailed sentiment reports.
    - `report.html`: Summary view for sentiment insights.

- **media**: Stores generated plots (e.g., word clouds, bar plots).

- **Datasets**: Contains example data files (`new.csv`, `run.csv`, etc.) for testing.

## Prerequisites

Before running the application, ensure you have the following installed:

- Python 3.8+
- Django 4.2
- Google Chrome (required for `chromedriver.exe`)
- Dependencies listed in `requirements.txt` (install with `pip install -r requirements.txt`).

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/Sentimap.git
   cd Sentimap
   ```

2. **Set up a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate   # For Linux/Mac
   venv\Scripts\activate      # For Windows
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Apply Migrations**:
   ```bash
   python manage.py migrate
   ```

5. **Run the Server**:
   ```bash
   python manage.py runserver
   ```
   
## Usage

1. Navigate to the main page.
2. Enter the URL or text data for analysis.
3. View sentiment visualizations, including:
   - Word clouds
   - Sentiment pyramids
   - Distribution charts


## Technical Stack

- **Backend**: Django, Python
- **Frontend**: HTML, CSS (via Django templates)
- **Web Scraper**: Selenium
- **Visualization**: Matplotlib, Seaborn

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.


## Author

**Pyla Sandeep**  
LinkedIn: [(https://linkedin.com/in/pyla-sandeep-757489243)](#)  
GitHub: [(https://github.com/pylasandeep52)](#)
