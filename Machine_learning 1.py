#Jon Smith
#CWID: 886383009


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import linear_model 
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold



def main(): 
    df = pd.read_csv("Data1.csv")
    df.insert(loc=0, column = 'intercept', value = 1)
    pd.set_option('display.max_columns', 80)
    pd.set_option('display.max_rows', 80)
    x = df.drop(['Idx'], axis = 1).values
    y = df['Idx'].values
    x = np.array(x)
    y = np.array(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)
    print("Choose a number: ")
    print("1. Least Square")
    print("2. Gradient Descent")
    print("3. Lasso, Ridge, and Elastic Net")
    print("4. Feature Scaling")
    print("5. K Folds")
    print("6. Exit")
    print("")
    choice = int(input())
    while choice != 6:
        if choice == 1:
            y_pred_train, y_pred_test = leastSquare(x, y, x_train, y_train, x_test, y_test)

            print("See Training graph? 1 for yes")
            choice = int(input())
            if choice == 1:
                plotGraphs(y_train, y_pred_train)

            print("See Testing graph? 1 for yes")
            choice = int(input())
            if choice == 1:
                plotGraphs(y_test, y_pred_test)

        elif choice == 2:
            print("Number of Iterations: ")
            iter = int(input())
            cost, theta = gradientDescent(x_train, y_train, 0.000007, iter)
            print("Final Cost: ", cost)
            print("Theta values generated by gradient descent: ", theta)

            plt.plot(cost)
            plt.xlabel("iter (number of iterations)")
            plt.ylabel("cost or loss")
            plt.show()

            y_pred_train = np.matmul(x_train, theta)
            y_pred_test = np.matmul(x_test, theta)

            print("Training dataset coefficient of determination: ", r2_score(y_train, y_pred_train))
            print("Testing dataset coefficient of determination: ", r2_score(y_test, y_pred_test))
            print("RMSE (train): ", mean_squared_error(y_train, y_pred_train))
            print("RMSE (test): ", mean_squared_error(y_test, y_pred_test))


            print("See Training graph? 1 for yes")
            choice = int(input())
            if choice == 1:
                plotGraphs(y_train, y_pred_train)

            print("See Testing graph? 1 for yes")
            choice = int(input())
            if choice == 1:
                plotGraphs(y_test, y_pred_test)
        elif choice == 3:
            LREN(x_train, y_train, x_test, y_test)
        elif choice == 4:
            featScale(x, y, df)
        elif choice == 5:
            kFolds(df)
        else:
            return
        x = df.drop(['Idx'], axis = 1).values
        y = df['Idx'].values
        x = np.array(x)
        y = np.array(y)
        print("Choose a number (only numbers): ")
        print("1. Least Square")
        print("2. Gradient Descent")
        print("3. Lasso, Ridge, and Elastic Net")
        print("4. Feature Scaling")
        print("5. K Folds")
        print("6. Exit")
        choice = int(input())

# # Least Square

# In[48]:

def leastSquare(x, y, x_train, y_train, x_test, y_test):
    w_hat = np.matmul(np.matmul(np.linalg.inv(np.matmul(x.transpose(), x)), x.transpose()), y)
    y_pred_train = np.matmul(x_train, w_hat)
    y_pred_test = np.matmul(x_test, w_hat)

    print("Training dataset coefficient of determination: ", r2_score(y_train, y_pred_train))
    print("Testing dataset coefficient of determination: ", r2_score(y_test, y_pred_test))
    print("RMSE (train): ", mean_squared_error(y_train, y_pred_train))
    print("RMSE (test): ", mean_squared_error(y_test, y_pred_test))

    return y_pred_train, y_pred_test

# # Gradient Descent
def gradientDescent(x, y, lrate, iter_val):
    m = x.shape[0]
    n = x.shape[1]
    theta = np.ones(n)
    h = np.dot(x, theta)
    
    cost = np.ones(iter_val)
    for i in range(0, iter_val):
        theta[0] = theta[0] - (lrate / m) * sum(h - y)
        for j in range(1, n):
            theta[j] = theta[j] - (lrate / m) * sum((h - y) * x[:, j])
        h = np.dot(x, theta)
        cost[i] = 1 / (2 * m) * sum(np.square(h - y))
    return cost, theta

def plotGraphs(train, predicted):
    plt.figure(figsize = (15,10))
    plt.scatter(train, predicted)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted')
    plt.show()

# # Feature Scaling (Min-Max Scale)

def featScale(x, y, df):
    x = df.drop(['Idx'], axis = 1)
    y = df['Idx']
    x['T'] = (x['T'] - min(x['T'])) / (max(x['T']) - min(x['T']))
    x['P'] = (x['P'] - min(x['P'])) / (max(x['P']) - min(x['P']))
    x['TC'] = (x['TC'] - min(x['TC'])) / (max(x['TC']) - min(x['TC']))
    x['SV'] = (x['SV'] - min(x['SV'])) / (max(x['SV']) - min(x['SV']))
    y = (y - min(y)) / (max(y) - min(y))
    x = np.array(x)
    y = np.array(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)
    print("Use feature scaled values: ")
    print("1. Least Square")
    print("2. Gradient Descent")
    print("3. Lasso, Ridge, and Elastic Net")
    print("4. K Folds")
    print("5. Exit")

    inner_choice = int(input())
    while not inner_choice ==  6:
        if inner_choice == 1:
            y_pred_train, y_pred_test = leastSquare(x, y, x_train, y_train, x_test, y_test)

            print("See Training graph? 1 for yes")
            choice = int(input())
            if choice == 1:
                plotGraphs(y_train, y_pred_train)

            print("See Testing graph? 1 for yes")
            choice = int(input())
            if choice == 1:
                plotGraphs(y_test, y_pred_test)

        elif inner_choice == 2:
            print("Number of Iterations: ")
            iter = int(input())
            cost, theta = gradientDescent(x_train, y_train, 0.000007, iter)
            print("Final Cost: ", cost)
            print("Theta values generated by gradient descent: ", theta)

            plt.plot(cost)
            plt.xlabel("iter (number of iterations)")
            plt.ylabel("cost or loss")
            plt.show()

            y_pred_train = np.matmul(theta, x_train)
            y_pred_test = np.matmul(x_test, theta)

            print("Training dataset coefficient of determination: ", r2_score(y_train, y_pred_train))
            print("Testing dataset coefficient of determination: ", r2_score(y_test, y_pred_test))
            print("RMSE (train): ", mean_squared_error(y_train, y_pred_train))
            print("RMSE (test): ", mean_squared_error(y_test, y_pred_test))

            print("See Training graph? 1 for yes")
            choice = int(input())
            if choice == 1:
                plotGraphs(y_train, y_pred_train)

            print("See Testing graph? 1 for yes")
            choice = int(input())
            if choice == 1:
                plotGraphs(y_test, y_pred_test)
        elif inner_choice == 3:
            LREN(x_train, y_train, x_test, y_test)
        elif inner_choice == 4:
            kFolds(df)
        print("Use feature scaled values: ")
        print("1. Least Square")
        print("2. Gradient Descent")
        print("3. Lasso, Ridge, and Elastic Net")
        print("4. K Folds")
        print("5. Exit")
        inner_choice = int(input())

def LREN(x_train, y_train, x_test, y_test):
    print("Choose which normilization methiod to use: ")
    print("1. LASSO")
    print("2. Ridge")
    print("3. Elastic Net")
    print("4. Exit")
    LREN_choice = int(input())

    while LREN_choice != 4:
        # # Lasso
        if LREN_choice == 1:
            lasso = Lasso(alpha = 0.000007, max_iter = 2500, tol = 0.0001)
            lasso.fit(x_train, y_train)

            y_pred = lasso.predict(x_train)
            y_pred_test = lasso.predict(x_test)

    # # Ridge
        elif LREN_choice == 2:
            ridge_reg = Ridge(alpha = 0.000007, max_iter = 2500, tol = 0.0001)
            ridge_reg.fit(x_train, y_train)

            y_pred = ridge_reg.predict(x_train)
            y_pred_test = ridge_reg.predict(x_test)

# # Elastic Net
        elif LREN_choice == 3:
            Elastic_reg = ElasticNet(alpha = 0.000007, max_iter = 2500)
            Elastic_reg.fit(x_train, y_train)

            y_pred = Elastic_reg.predict(x_train)
            print("Coef: ", Elastic_reg.coef_)
            print("Intercept: ", Elastic_reg.intercept_)
            y_pred_test = Elastic_reg.predict(x_test)

        print("Coefficient of determination (training): ", r2_score(y_train, y_pred))
        print("Coefficient of determination (testing): ", r2_score(y_test, y_pred_test))
        print("RMSE (train): ", mean_squared_error(y_train, y_pred))
        print("RMSE (test): ", mean_squared_error(y_test, y_pred_test))
        print("See Training graph? 1 for yes")
        choice = int(input())
        if choice == 1:
            plotGraphs(y_train, y_pred)

        print("See Training graph? 1 for yes")
        choice = int(input())
        if choice == 1:
            plotGraphs(y_test, y_pred_test)


        print("Choose which normilization methiod to use: ")
        print("1. LASSO")
        print("2. Ridge")
        print("3. Elastic Net")
        print("4. Exit")
        LREN_choice = int(input())

# # K Cross
def kFolds(df):
    x = df.iloc[:,:-1]
    y = df.iloc[:, -1]
    print("Which kFolds algorithm to test? ")
    print("1. Lasso")
    print("2. Ridge")
    print("3. Elastic Net")
    print("4. Exit")
    k_choice = int(input())
    k = 10
    kf = KFold(n_splits = k, random_state = None)
    while k_choice != 4:
        acc_score = []

# ### Lasso
        if k_choice == 1:
            for train_index, test_index in kf.split(x):
                x_train, x_test = x.iloc[train_index, :], x.iloc[test_index, :]
                y_train, y_test = y[train_index], y[test_index]
        
                lasso = linear_model.Lasso(alpha = 0.000007, max_iter = 2500, tol = 0.0001)
                lasso_pred = lasso.fit(x_train, y_train)
                pred_values = lasso_pred.predict(x_test)
        
                acc = r2_score(pred_values, y_test)
                acc_score.append(acc)
            avg_acc_score = sum(acc_score) / k
            print('accuracy of each fold - {}'.format(acc_score))
            print('Avg accuracy : {}'.format(avg_acc_score))


# ### Ridge 
        elif k_choice == 2:
            for train_index, test_index in kf.split(x):
                x_train, x_test = x.iloc[train_index, :], x.iloc[test_index, :]
                y_train, y_test = y[train_index], y[test_index]
                
                ridge_reg = Ridge(alpha = 0.000007, max_iter = 2500, tol = 0.0001)
                ridge_reg.fit(x_train, y_train)
                pred_values = ridge_reg.predict(x_test)
                
                acc = r2_score(pred_values, y_test)
                acc_score.append(acc)

            avg_acc_score = sum(acc_score)/ k

            print('accuracy of each fold - {}'.format(acc_score))
            print('Avg accuracy : {}'.format(avg_acc_score))


# ### Elastic Net  
        elif k_choice == 3:
            for train_index, test_index in kf.split(x):
                x_train, x_test = x.iloc[train_index, :], x.iloc[test_index, :]
                y_train, y_test = y[train_index], y[test_index]
                
                Elastic_reg = ElasticNet(alpha = 0.000007, max_iter = 2500, tol = 0.0001)
                Elastic_reg.fit(x_train, y_train)
                pred_values_elastic = Elastic_reg.predict(x_test)
                
                acc = r2_score(pred_values_elastic, y_test)
                acc_score.append(acc)

            avg_acc_score = sum(acc_score)/ k

            print('accuracy of each fold - {}'.format(acc_score))
            print('Avg accuracy : {}'.format(avg_acc_score))
        else:
            break
        
        print("Which kFolds algorithm to test? ")
        print("1. Lasso")
        print("2. Ridge")
        print("3. Elastic Net")
        print("4. Exit")
        k_choice = int(input())

main()