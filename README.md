# 🦆DuckRunner🏃‍♂️

DuckRunner is a Python-based Streamlit application designed to parse, analyze,
and visualize running metrics extracted from `.fit` files (Garmin, Strava, and
similar sources).

The project provides an interactive experience for exploring pace, heart rate, elevation, and lap data—with visualizations inspired by Strava’s performance charts.

You can access the [DuckerRunner](https://duckrunneranalysis.streamlit.app/) at Streamlit or run it in from your machine using Docker (for more information, go to Installation).

## 🧑‍💻 Installation

To get started with the DuckerRunnerAnalisis, follow these steps:

1. Ensure you have [Docker](https://www.docker.com/) installed and at least 3GB of free space.

2. Clone the repository from GitHub:
   ```bash
   git clone https://github.com/gregomelo/DuckRunnerAnalysis.git
   ```

3. Navigate into the project directory and build the Docker image:
    ```bash
    cd DuckRunnerAnalysis
    docker build -t duckrunner-image .
    ```

4. AOnce the build is complete, run the container:
    ```
    bash
    docker run -d --name duckrunner-container -p 8501:8501 duckrunner-image
    ```

5. Open your browser and go to http://localhost:8501/.

## 🚀 How to use it

1. **Get Your Data**: Download a `.fit` file from your running device. If you use Strava, you can download your original data by following [this guide](https://support.strava.com/hc/en-us/articles/216918437-Exporting-your-Data-and-Bulk-Export#h_01GDP2BZGF4KACBW90R6VZW16T).<br><br>
*Don't have a file?* You can use one of the sample training files in the data folder to test the app.

2. Open DuckRunner: Go to the [DuckRunner app on Streamlit](https://duckrunneranalysis.streamlit.app/).<br><br>
**Note**: The Streamlit app may go to sleep due to inactivity. If this happens, just click "Yes, get this app back up!" and wait a few moments.

4. **Configure Your Zones**: Set up your personal running training zones, which can be calculated from a [3km test](https://www.instagram.com/p/DF3LmsYPprr/?utm_source=ig_web_copy_link&igsh=MzRlODBiNWFlZA==). If you don't know your zones yet, you can use the default ones to see how it works.

3. **Voilà!** Explore your running data in detail and gain insights to improve your performance.

## 🏃‍♀️‍➡️Support🧑‍🦽‍➡️

If you like this project, please consider supporting [Correndo por Eles](https://www.instagram.com/correndoporeles/). a social project that helps include people with disabilities in street running events.

For questions or contributions, you can contact me on [LinkedIn](https://www.linkedin.com/in/gregomelo).


## 📈 Release Notes

### 2025-11-02

  - Introduced noise filtering for speed data

  - Integrated heart rate (BPM) and altitude (m) information into the main plot

  - Added X-axis range selection for interactive chart exploration

## 🔮 Future Features

- Improve tables

- New plots for lap and split analysis

- Enhanced interactivity with plots
