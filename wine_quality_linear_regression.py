import pandas as pd
import seaborn as sea
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

pd.set_option("display.expand_frame_repr", False)

if __name__ == "__main__":

    # caricamento dataset
    df = pd.read_csv("winequality.csv")

    print(df.shape)
    print(df.info())
    print(df.describe())
    print(df.head())

    # istogramma dell'output
    df[["quality"]].hist()

    # graficare matrice di correlazione
    plt.figure()
    corr_matrix = df.corr()
    sea.heatmap(corr_matrix, annot=True)  # annot(ation) riporta anche i valori nelle varie caselle

    plt.show()

    # scelgo le feature e il target
    target = "quality"
    feature = ["alcohol", "sulphates", "citric acid", "volatile acidity"]

    x = df[feature].values
    y = df[target].values.reshape(-1, 1)

    # divido in training set e test set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # addestro il modello
    regressor = LogisticRegression()
    regressor.fit(x_train, y_train)

    # predizioni sul test set
    predictions = regressor.predict(x_test)
    print("MSE: ", metrics.mean_squared_error(y_test, predictions))
    print("MAE: ", metrics.mean_absolute_error(y_test, predictions))

    # confronto visivamente le predizioni con le etichette sul test set
    df_test_set = pd.DataFrame({"Labels": y_test.flatten(),
                                "Predictions": predictions.flatten()})
    print(df_test_set.head(20))

    # visualizzo la matrice di confusione
    conf_matrix = metrics.confusion_matrix(y_test, predictions)
    metrics.ConfusionMatrixDisplay(conf_matrix).plot()
    plt.show()
