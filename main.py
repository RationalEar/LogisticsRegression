# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def main(name):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sb
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report

    ad_data = pd.read_csv('advertising.csv')
    print(ad_data.columns)
    sb.histplot(ad_data, x='Age', bins=30)
    sb.jointplot(x='Age', y='Area Income', data=ad_data)
    sb.jointplot(x='Age', y='Daily Time Spent on Site', data=ad_data, kind='kde')
    sb.jointplot(x='Daily Time Spent on Site', y='Daily Internet Usage', data=ad_data)
    sb.pairplot(data=ad_data, hue='Clicked on Ad')
    plt.show()
    X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage']]
    y = ad_data['Clicked on Ad']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
    lojmodel = LogisticRegression()
    lojmodel.fit(X_train, y_train)
    predictions = lojmodel.predict(X_test)
    print(classification_report(y_test, predictions))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
