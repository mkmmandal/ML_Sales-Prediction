# ML_Sales-Prediction
This is an end-to-end Machine Learning Project from Defining problem to building an app working on the ML model.

In this project we have to predict the sales of various grocery items. Data is coming from Kaggle Ongoing Competition Dataset.(https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data)

After EDA, Data visualization, I deployed various Regression Models and then chose the best model based on RMSE. After that some parameters hypertuning for better results. Then I create an app that will be predicting the sales amount based on this model.

Description of files in this repository:

1.SalesPrediction_Notebook.ipynb -Main Jupyter notebook for this project. All processes of data loading,cleaning are done in this. The difference is model is deployed on 10% of sample data.

2.app.py -Application file for this project. App is build using Flask and basic html,css.

3.templates/index.html -HTML file for app. (I used inline-css for little styling, you can add static/styles.css file as well for styling)

4.Prophet_Notebook.ipynb -Sales prediction using Prophet library.

5.GPUModel_Notebook.ipynb -Google colab notebook where I deployed chosen model on the complete dataset using GPU processing provided by colab and cudf libraries by nvidia.
