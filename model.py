import matplotlib.pyplot as plt
import pandas as pd
from data import getStockData, convertData
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

df = convertData(getStockData("TSLA"))
print(df)

plt.plot(df['Date'], df['Close'])
#plt.show()

classifiers = ["Close", "Open", "High", "Low", "Volume", "RSI", "EMFA", "EMFM", "EMFS"]

split = int(len(df) * 0.95)

X_train = df.iloc[:split]
X_test = df.iloc[split:]

model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)
model.fit(X_train[classifiers], X_train["Target"])

preds = model.predict(X_test[classifiers])
preds = pd.Series(preds, index=X_test.index)

precision_score(X_test["Target"], preds)

res_table = pd.concat([X_test["Target"], preds], axis=1)

res_table.plot()

def predict(train, test, classifiers, model):
    model.fit(train[classifiers], train["Target"])
    preds = model.predict(test[classifiers])
    preds = pd.Series(preds, index=test.index, name="Predictions")
    res_table = pd.concat([test["Target"], preds], axis=1)
    return res_table

def backtest(data, model, classifiers, start=2500, step=250):
    all_predictions = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, classifiers, model)
        all_predictions.append(predictions)
    
    return pd.concat(all_predictions)

predictions = backtest(df, model, classifiers)

print(predictions["Predictions"].value_counts())

print(precision_score(predictions["Target"], predictions["Predictions"]))