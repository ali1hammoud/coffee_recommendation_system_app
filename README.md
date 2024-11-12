# Coffee Recommender

Welcome to the **Coffee Recommender** application! ☕️✨

Discover the perfect coffee tailored to your unique taste preferences using advanced machine learning algorithms. Whether you enjoy a bold dark roast or a light, fruity brew, our system provides personalized recommendations to enhance your coffee experience.

---

## Table of Contents

- [About](#about)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Training the Models](#training-the-models)
  - [Running the Application](#running-the-application)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## About

The **Coffee Recommender** is a web-based application built with Flask that leverages machine learning algorithms to provide personalized coffee recommendations based on user preferences. Users can specify their desired coffee attributes, such as color, roast level, flavor profile, acidity, body, caffeine content, and origin, to receive tailored suggestions.

---

## Features

- **Model Training**: Train machine learning models (Random Forest and K-Nearest Neighbors) directly through the web interface.
- **Personalized Recommendations**: Input your coffee preferences and receive customized coffee suggestions.
- **Multiple Algorithms**: Choose between Random Forest and KNN for generating recommendations.
- **Modern UI**: Enjoy a sleek and user-friendly interface styled with Bootstrap and custom CSS.
- **Persistent Models**: Models, encoders, and scalers are saved for future use without the need for retraining.

---

## Technologies Used

- **Backend**:
  - [Flask](https://flask.palletsprojects.com/) - Web framework
  - [scikit-learn](https://scikit-learn.org/) - Machine learning algorithms
  - [Pandas](https://pandas.pydata.org/) - Data manipulation
  - [Joblib](https://joblib.readthedocs.io/) - Model serialization

- **Frontend**:
  - [Bootstrap 5](https://getbootstrap.com/) - CSS framework
  - [HTML5 & CSS3](https://developer.mozilla.org/en-US/docs/Web/Guide/CSS)
  - [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript) (for Bootstrap functionalities)

---

## Getting Started

Follow these instructions to set up and run the Coffee Recommender application on your local machine.

### Prerequisites

- **Python 3.9 or higher**: Ensure you have Python installed. You can download it from [here](https://www.python.org/downloads/).
- **Git**: To clone the repository. Download from [here](https://git-scm.com/downloads).

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/coffee-recommender.git
   cd coffee-recommender
   ```

2. **Create a Virtual Environment**

   It's recommended to use a virtual environment to manage dependencies.

   ```bash
   python3 -m venv venv
   ```

3. **Activate the Virtual Environment**

   - **On Windows:**

     ```bash
     venv\Scripts\activate
     ```

   - **On macOS/Linux:**

     ```bash
     source venv/bin/activate
     ```

4. **Prepare the Dataset**

   Ensure the `coffee_data.csv` file is present in the `data/` directory. If not, create it using the sample data provided in the project.

### Training the Models

1. **Start the Flask Application**

   ```bash
   python app.py
   ```

2. **Navigate to the Training Page**

   Open your web browser and go to [http://127.0.0.1:5000/train](http://127.0.0.1:5000/train).

3. **Select an Algorithm and Train**

   - Choose either **K-Nearest Neighbors (KNN)** or **Random Forest** from the dropdown menu.
   - Click on **"Train and Save Model"**.
   - A success message will appear upon successful training.

   *Repeat the process for both algorithms to have both models trained and saved.*

### Running the Application

1. **Start the Flask Application**

   If not already running, start the app:

   ```bash
   python app.py
   ```

2. **Access the Application**

   Open your web browser and navigate to [http://127.0.0.1:5000/](http://127.0.0.1:5000/) to access the Coffee Recommender.

---

## Project Structure

```
coffee_recommender/
├── app.py
├── models/
│   ├── knn_model.joblib
│   ├── rf_model.joblib
│   ├── encoder.joblib
│   └── scaler.joblib
├── static/
│   ├── css/
│   │   └── styles.css
│   └── images/
│       └── coffee_background.webp
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── train.html
│   └── recommend.html
├── data/
│   └── coffee_data.csv
└── README.md
```

- **app.py**: Main Flask application containing routes and logic.
- **models/**: Directory to store trained models, encoders, and scalers.
- **static/**: Contains static files like CSS, JavaScript, and images.
  - **css/styles.css**: Custom styles for the application.
  - **images/**: Store images used in the application (e.g., background images).
- **templates/**: HTML templates for rendering pages.
  - **base.html**: Base template with common layout elements.
  - **index.html**: Landing page.
  - **train.html**: Model training page.
  - **recommend.html**: Recommendation form and results page.
- **data/**: Contains the dataset (`coffee_data.csv`) used for training models.
- **requirements.txt**: Lists all Python dependencies.
- **README.md**: Project documentation.

---

## Usage

1. **Train the Models**

   - Navigate to [http://127.0.0.1:5000/train](http://127.0.0.1:5000/train).
   - Select an algorithm (**KNN** or **Random Forest**).
   - Click **"Train and Save Model"**.

2. **Get Coffee Recommendations**

   - Navigate to [http://127.0.0.1:5000/recommend](http://127.0.0.1:5000/recommend).
   - Fill out the form with your coffee preferences:
     - **Color**: Light, Medium, Dark.
     - **Roast Level**: Light Roast, Medium Roast, Dark Roast.
     - **Flavor Profile**: Select from predefined options.
     - **Acidity**: High, Medium, Low.
     - **Body**: Light, Medium, Full.
     - **Caffeine Content (mg)**: Enter a numerical value (e.g., 150).
     - **Preferred Origin**: Select from predefined origins.
     - **Recommendation Algorithm**: Choose **KNN** or **Random Forest**.
   - Click **"Get Recommendation"** to view suggested coffees.

---

## License

Distributed under the MIT License. See `LICENSE` for more information.

---

## Acknowledgements

- [Flask Documentation](https://flask.palletsprojects.com/)
- [Bootstrap Documentation](https://getbootstrap.com/docs/5.3/getting-started/introduction/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Joblib Documentation](https://joblib.readthedocs.io/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Font Awesome](https://fontawesome.com/)
- [Google Fonts](https://fonts.google.com/)
- [Unsplash](https://unsplash.com/) for free images

---

**Happy Brewing! ☕️😊**

---