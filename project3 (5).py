    #!/usr/bin/env python
    # coding: utf-8

    # In[2]:


    #This is to take in options,stocks data


    # In[3]:


    import pandas as pd
    #Load your csv file
    df = pd.read_csv("Formatted_Options_Data.csv")
    df.dropna(inplace=True)
    #checking first few rows
    print(df.head())


    # In[ ]:





    # In[ ]:





    # In[ ]:





    # In[4]:


    #This is to calc theoretical price using BS model
    import numpy as np
    from scipy.stats import norm
    def black_scholes_call(S, K, T, r, sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    #This is to find best sigma, ie backtrack to find best implied vol
    #Usign newton rafson method
    def implied_vol(S, K, T, r, market_price, tol=1e-6, max_iter=100):
        sigma = 0.2  # Initial guess (can be adjusted)
        for i in range(max_iter):
            price = black_scholes_call(S, K, T, r, sigma)
            vega = (S * norm.pdf((np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))) * np.sqrt(T))
            
            diff = market_price - price
            if abs(diff) < tol:  # Stop if difference is small
                return sigma
            
            sigma += diff / vega  # Newton-Raphson update
            
        return sigma  # Return last computed sigma if max iterations reached



    # In[5]:


    #This is to calc greeks(we need only delta and gamma in our case)
    def delta(S, K, T, r, sigma,option_type):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        if option_type == "call":
            return norm.cdf(d1)
        elif option_type == "put":
            return norm.cdf(d1) - 1
        else:
            raise ValueError("Invalid option type. Please use'call' or 'put'.")
    def gamma(S, K, T, r, sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        N_prime_d1 = norm.pdf(d1)
        return N_prime_d1/(S*sigma*np.sqrt(T))
    #Extra values of greeks
    def other_greeks(S, K, T, r, sigma, option):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Vega (per 1% volatility change)
        vega = (S * norm.pdf(d1) * np.sqrt(T)) / 100  

        # Theta (per day)
        first_term = - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
        if option == "call":
            theta = first_term - r * K * np.exp(-r * T) * norm.cdf(d2)
        else:  # put option
            theta = first_term + r * K * np.exp(-r * T) * norm.cdf(-d2)
        theta /= 365  # Convert to per day

        # Rho (per 1% interest rate change)
        if option == "call":
            rho = (K * T * np.exp(-r * T) * norm.cdf(d2)) / 100
        else:  # put option
            rho = (-K * T * np.exp(-r * T) * norm.cdf(-d2)) / 100  

        return vega, theta, rho
        #all have T-t but we take t = current time to be 0


    # In[6]:


    #To add implied vol columns
    df["Implied Vol"] = df.apply(lambda row: implied_vol(row["Stock Price"], row["Strike Price"], 
                                                        row["Time to Expiration"], row["Risk-Free Rate"], 
                                                        row["Market Price"]), axis=1)
    print(df.head())


    # In[7]:


    import matplotlib.pyplot as plt
    import seaborn as sns
    #plot a histogram to see
    df.hist(bins=30, figsize=(12, 8))
    plt.suptitle("Feature Distributions")
    plt.show()
    #plot a stock price dist to see stock price vs implied vol
    plt.figure(figsize=(8,5))
    sns.scatterplot(x=df["Stock Price"],y=df["Implied Vol"]) #Make a scatter plot of impleidvol vs stocks price
    plt.xlabel("StockPrice")
    plt.ylabel("ImpliedVol")
    plt.show()

    # Compute Moneyness
    df["Moneyness"] = df["Stock Price"] / df["Strike Price"]

    # Scatter plot of Implied Volatility vs Moneyness , tells if intresic extrisnsic
    plt.scatter(df["Moneyness"], df["Implied Vol"])
    plt.xlabel("Moneyness")
    plt.ylabel("Implied Volatility")
    plt.title("Implied Volatility vs Moneyness")
    plt.show()


    # In[ ]:


    # #Compute Historical Volatility

    # Take a rolling standard deviation of log returns over a window (e.g., 30 days).
    # Multiply by √252 to annualize.
    # Calculate log returns
    df["Log Returns"] = np.log(df["Stock Price"] / df["Stock Price"].shift(1)) #basically divide stock price at time t / at time t-1
    # Compute rolling standard deviation (Historical Volatility)
    window_size = 30  # Adjust based on available data
    df["Hist Vol"] = df["Log Returns"].rolling(window=window_size).std() * np.sqrt(252)
    # Drop NaN values from rolling calculation
    df.dropna(inplace=True)
    # Scatter plot: Historical Volatility vs. Implied Volatility
    plt.figure(figsize=(8,6))
    plt.scatter(df["Hist Vol"], df["Implied Vol"], alpha=0.7)
    plt.xlabel("Historical Volatility")
    plt.ylabel("Implied Volatility")
    plt.title("Historical Volatility vs Implied Volatility")
    plt.grid(True)
    plt.show()


    # In[ ]:


    df["Option Type"] = df["Option Type"].str.lower()
    #convertion to small case for aour lambda func 
    df.head(5)


    # In[ ]:





    # In[ ]:


    df["Delta"] = df.apply(lambda row: delta(row["Stock Price"], row["Strike Price"], 
                                            row["Time to Expiration"], row["Risk-Free Rate"], 
                                            row["Implied Vol"], row["Option Type"]), axis=1)

    df["Gamma"] = df.apply(lambda row: gamma(row["Stock Price"], row["Strike Price"], 
                                            row["Time to Expiration"], row["Risk-Free Rate"], 
                                            row["Implied Vol"]), axis=1)
    df["Log Return"] = np.log(df["Stock Price"] / df["Stock Price"].shift(1))
    #Now we hv added our impo columns


    # In[ ]:


    #To make a heatmap
    df_numeric = df.select_dtypes(include=[np.number])  # Keep only numeric columns
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_numeric.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.show()
    #Risk free rate = nothing as each value is constant


    # In[ ]:


    #WE WILL  START ML FROM HERE IE TRAINING AND TESTING


    # In[ ]:


    #LETS SPLIT INTO TRAIN , VAL , TEST SET = 70 10 20 SPLIT
    from sklearn.model_selection import train_test_split
    df = df.drop(columns=['Risk-Free Rate'])
    #Selecting features and targets
    X = df[["Stock Price", "Strike Price", "Time to Expiration", "Hist Vol"]]
    Y = df["Implied Vol"]

    # Spliting 70 % 30%
    X_train, X_temp, Y_train , Y_temp = train_test_split(X,Y,test_size=0.3,random_state=42)
    # Splitting 30 into 10 20 
    X_val , X_test, Y_val, Y_test= train_test_split(X_temp,Y_temp,test_size=2/3,random_state=42)
    # Check sizes
    print(f"Train size: {len(X_train)}, Validation size: {len(X_val)}, Test size: {len(X_test)}")


    # In[ ]:


    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error

    # Initialize the Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train,Y_train)
    Y_val_pred=rf_model.predict(X_val)
    mse_val=mean_squared_error(Y_val,Y_val_pred)
    print("Validation set MSE is:" + str(mse_val))
    #Pred of test set
    Y_test_pred=rf_model.predict(X_test)
    mse_test=mean_squared_error(Y_test,Y_test_pred)
    print("Test set MSE is:" + str(mse_test))


    # In[ ]:


    #To see plot we can do 
    importances = rf_model.feature_importances_
    feature_names = X_train.columns
    plt.figure(figsize=(8,5))
    sns.barplot(x=importances,y=feature_names)
    plt.xlabel("Features importance")
    plt.ylabel("Features")
    plt.title("Random Forest Features Importance")
    plt.show()

    #We see that time to expirn is the most impo with 0.3 percent corr with iv


    # In[ ]:


    #Now we will do hyperparameter tuning
    # We will use gridsearchcv
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestRegressor
    # Define the hyperparameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],  # Number of trees
        'max_depth': [5, 10, None],  # Depth of each tree
        'min_samples_split': [2, 5, 10]  # Minimum samples to split a node
    }

    rf = RandomForestRegressor(random_state=42)
    #Set up gridsearch for 3 fold cross validn
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    #Fit GridSearchCV to training data
    grid_search.fit(X_train, Y_train)
    #Printing whcih is the best parameter
    print("Best hyperparameters:", grid_search.best_params_)
    # Get the best model from grid search
    best_rf = grid_search.best_estimator_
    # Evaluate on validation set
    val_predictions = best_rf.predict(X_val)
    val_mse = mean_squared_error(Y_val, val_predictions)
    print("Validation MSE with best hyperparameters:", val_mse)
    # Evaluate on test set
    test_predictions = best_rf.predict(X_test)
    test_mse = mean_squared_error(Y_test, test_predictions)
    print("Test MSE with best hyperparameters:", test_mse)


    # In[ ]:


    df


    # In[ ]:


    plt.scatter(Y_test, test_predictions, alpha=0.7)
    plt.xlabel("Market IV (Y_test)")
    plt.ylabel("Predicted IV (test_predictions)")
    plt.title("Predicted IV vs. Market IV")
    plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], color="red", linestyle="--")  # Ideal line
    plt.show()


    # In[ ]:


    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    mse = mean_squared_error(Y_test, test_predictions)
    mae = mean_absolute_error(Y_test, test_predictions)
    r2 = r2_score(Y_test, test_predictions)

    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print(f"Mean Absolute Error (MAE): {mae:.6f}")
    print(f"R² Score: {r2:.6f}")


    # In[ ]:


    from sklearn.metrics import r2_score
    import numpy as np

    Y_mean = np.full_like(Y_test, np.mean(Y_train))  # Predict mean of y_train for all test samples
    baseline_r2 = r2_score(Y_test, Y_mean)
    print(f"Baseline R² (mean prediction): {baseline_r2}")


    # In[ ]:


    import matplotlib.pyplot as plt

    Y_pred = best_rf.predict(X_test)  # Ensure this is correctly implemented
    plt.scatter(Y_test, Y_pred)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Predicted vs Actual Values")
    plt.show()


    # In[ ]:


    dfnew = pd.DataFrame(X_train, columns=feature_names)  # Replace feature_names with actual names
    dfnew['Target'] = Y_train
    print(dfnew.corr()['Target'].sort_values(ascending=False))


    # In[ ]:


    print(df['Implied Vol'].describe())


    # In[ ]:


    print(df.corr())


    # In[ ]:


    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.boxplot(data=df)
    plt.xticks(rotation=90)
    plt.show()


    # In[ ]:


    correlation = df.select_dtypes(include=['number']).corr()['Implied Vol'].dropna().sort_values(ascending=False)
    print(correlation)


    # In[ ]:


    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.pairplot(df[['Implied Vol', 'Market Price', 'Hist Vol', 'Gamma']])
    plt.show()


    # In[ ]:




