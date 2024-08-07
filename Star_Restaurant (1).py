#!/usr/bin/env python
# coding: utf-8

# # Star Restaurant

# In[1]:


# Importing data libraries
import pandas as pd
import numpy as np 
import os

# To display number rows and columns
pd.set_option('display.max_rows',20)
pd.set_option('display.max_columns',None)

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.simplefilter("ignore")


# In[2]:


Restrant_data=pd.read_excel("C:/ALEKYA/casptone/1683266696_dataset (1)/Dataset/data.xlsx")
Country_data=pd.read_excel("C:/ALEKYA/casptone/1683266696_dataset (1)/Dataset/Country-Code.xlsx")


# In[3]:


Country_data.head(5)


# In[4]:


Restrant_data.head(5)


# In[5]:


Restrant_data.shape


# In[6]:


Country_data.shape


# In[7]:


# Merge the tables based on the 'Country Code' column
Restrant_data1 = pd.merge(Restrant_data, Country_data, on='Country Code', how='left')

# Display the merged table
Restrant_data1.head(5)


# In[8]:


# 1. Perform preliminary data inspection and report the findings as the structure of the data, missing 
# values, duplicates, etc.
Restrant_data1.shape


# In[9]:


print ("Shape of data: {}" . format (Restrant_data1.shape))
print ("Number of rows: {}" . format (Restrant_data1.shape [0]))
print ("Number of columns: {}" . format (Restrant_data1.shape [1]))


# In[10]:


Restrant_data1.dtypes


# In[11]:


Restrant_data1.info()


# In[12]:


Restrant_data1.nunique()


# In[13]:


Restrant_data1.isnull().sum()


# In[14]:


# 2. Based on the findings from the previous questions, identify duplicates and remove them
Restrant_data1.duplicated().sum()


# In[15]:


Restrant_data1.dropna(inplace=True)


# In[16]:


Restrant_data1.isnull().sum()


# In[17]:


# 3 Explore the geographical distribution of the restaurants and identify the cities with the 
# maximum and minimum number of restaurants
# Count the number of restaurants in each city
restaurant_counts = Restrant_data1['City'].value_counts()

# Identify the city with the maximum number of restaurants
city_with_max_restaurants = restaurant_counts.idxmax()
max_restaurant_count = restaurant_counts.max()

# Identify the city with the minimum number of restaurants
city_with_min_restaurants = restaurant_counts.idxmin()
min_restaurant_count = restaurant_counts.min()


# Print the results
print(f"City with the maximum number of restaurants: {city_with_max_restaurants}, Count: {max_restaurant_count}")
print(f"City with the minimum number of restaurants: {city_with_min_restaurants}, Count: {min_restaurant_count}")


# In[18]:


# Counting the number of restaurants in each city
city_restaurant_counts = Restrant_data1['City'].value_counts()

# Finding the city with the minimum number of restaurants
min_city = city_restaurant_counts.idxmin()
min_count = city_restaurant_counts.min()

# Finding the city with the maximum number of restaurants
max_city = city_restaurant_counts.idxmax()
max_count = city_restaurant_counts.max()

# Plotting the geographical distribution of restaurants
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Longitude', y='Latitude', data=Restrant_data1, hue='City', palette='viridis', legend=False)
# Annotating the city with the minimum number of restaurants
plt.annotate(f'Min: {min_city} ({min_count} restaurants)', 
             xy=(Restrant_data1.loc[Restrant_data1['City'] == min_city, 'Longitude'].mean(), 
                 Restrant_data1.loc[Restrant_data1['City'] == min_city, 'Latitude'].mean()),
             xytext=(10, 10), textcoords='offset points', fontsize=10,
             arrowprops=dict(arrowstyle='->', color='blue'))
# Annotating the city with the maximum number of restaurants
plt.annotate(f'Max: {max_city} ({max_count} restaurants)', 
             xy=(Restrant_data1.loc[Restrant_data1['City'] == max_city, 'Longitude'].mean(), 
                 Restrant_data1.loc[Restrant_data1['City'] == max_city, 'Latitude'].mean()),
             xytext=(10, -20), textcoords='offset points', fontsize=10,
             arrowprops=dict(arrowstyle='->', color='red'))

plt.title('Geographical Distribution of Restaurants')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()


# In[19]:


# Grouping the data by restaurant franchise name and counting the number of unique cities
franchise_presence = Restrant_data1.groupby('Restaurant Name')['City'].nunique()

# Finding the franchise with the maximum national presence
most_national_franchise = franchise_presence.idxmax()
max_presence = franchise_presence.max()

print(f"The franchise with the most national presence is '{most_national_franchise}' with presence in {max_presence} cities.")


# In[20]:


#  5. Find out the ratio between restaurants that allow table booking vs. those that do not allow table 
# booking
# Calculate the counts of restaurants with and without table booking
booking_counts = Restrant_data1['Has Table booking'].value_counts()

# Calculate the ratio
booking_ratio = booking_counts / booking_counts.sum()

# Create a bar plot
booking_ratio.plot(kind='bar', color=['blue', 'orange'])
plt.xlabel('Table Booking')
plt.ylabel('Ratio')
plt.title('Ratio of Restaurants with and without Table Booking')
plt.xticks([1, 0], ['Yes', 'No'], rotation=0)

# Annotate bars with percentages
for i, ratio in enumerate(booking_ratio):
    plt.text(i, ratio + 0.01, f"{ratio:.2f}", ha='center', va='bottom')
plt.show()


# In[21]:


# Calculate the counts of restaurants with and without online delivery
delivery_counts = Restrant_data1['Has Online delivery'].value_counts()

# Calculate the percentage
delivery_percentage = delivery_counts / delivery_counts.sum() * 100

# Create a bar plot
delivery_percentage.plot(kind='bar', color=['green', 'red'])
plt.xlabel('Online Delivery')
plt.ylabel('Percentage')
plt.title('Percentage of Restaurants Providing Online Delivery')
plt.xticks([1, 0], ['Yes', 'No'], rotation=0)

# Annotate bars with percentages
for i, percentage in enumerate(delivery_percentage):
    plt.text(i, percentage + 0.5, f"{percentage:.2f}%", ha='center', va='bottom')

plt.show()


# In[22]:


#7. Calculate the difference in number of votes for the restaurants that deliver and the restaurants 
# that do not deliver
# Calculate the total number of votes for restaurants that deliver
delivery_yes_votes = Restrant_data1[Restrant_data1['Has Online delivery'] == 'Yes']['Votes'].sum()

# Calculate the total number of votes for restaurants that do not deliver
delivery_no_votes = Restrant_data1[Restrant_data1['Has Online delivery'] == 'No']['Votes'].sum()

# Calculate the difference in the number of votes
vote_difference = delivery_yes_votes - delivery_no_votes
print("Difference in number of votes between restaurants that deliver and those that do not deliver:", vote_difference)


# In[23]:


#1. What are the top 10 cuisines served across cities?

# Group the data by cuisine and city, count occurrences, and reset index
top_cuisines = Restrant_data1.groupby(['Cuisines', 'City']).size().reset_index(name='Count')

# Sort the data by count in descending order
top_cuisines = top_cuisines.sort_values(by='Count', ascending=False)

# Get the top 10 cuisines
top_10_cuisines = top_cuisines.head(10)

# Plot the top 10 cuisines
plt.figure(figsize=(10, 6))
plt.barh(top_10_cuisines['Cuisines'], top_10_cuisines['Count'], color='skyblue')
plt.xlabel('Number of Restaurants')
plt.ylabel('Cuisine')
plt.title('Top 10 Cuisines Served Across Cities')
plt.gca().invert_yaxis()  # Invert y-axis to display the cuisine with the highest count at the top
plt.show()


# In[24]:


# 2. What is the maximum and minimum number of cuisines that a restaurant serves? Also, which is 
# the most served cuisine across the restaurant for each city?
# Calculate the number of cuisines served by each restaurant
Restrant_data1['Num_Cuisines'] = Restrant_data1['Cuisines'].str.split(',').apply(lambda x: len(x))

# Maximum number of cuisines served
max_cuisines = Restrant_data1['Num_Cuisines'].max()

# Minimum number of cuisines served
min_cuisines = Restrant_data1['Num_Cuisines'].min()

print("Maximum number of cuisines served by a restaurant:", max_cuisines)
print("Minimum number of cuisines served by a restaurant:", min_cuisines)


# In[25]:


# Group the data by city and cuisine, count occurrences, and reset index
top_cuisine_per_city = Restrant_data1.groupby(['City', 'Cuisines']).size().reset_index(name='Count')

# Get the index of the most served cuisine for each city
idx = top_cuisine_per_city.groupby(['City'])['Count'].transform(max) == top_cuisine_per_city['Count']

# Filter the dataframe to get the most served cuisine for each city
most_served_cuisine_per_city = top_cuisine_per_city[idx]

print("Most served cuisine across restaurants for each city:")
print(most_served_cuisine_per_city[['City', 'Cuisines']])


# In[26]:


# What is the distribution cost across the restaurants?
# Plot a boxplot of the cost distribution
plt.boxplot(Restrant_data1['Average Cost for two'], vert=False)
plt.title('Distribution of Cost Across Restaurants')
plt.xlabel('Average Cost for Two')
plt.yticks([])
plt.show()


# In[27]:


# How are ratings distributed among the various factors?
# Plot a box plot of ratings for each factor
plt.figure(figsize=(10, 6))
sns.boxplot(x='Rating text', y='Aggregate rating', data=Restrant_data1)
plt.title('Distribution of Ratings Among Various Factors ')
plt.xlabel('Rating Text')
plt.ylabel('Aggregate Rating')
plt.xticks(rotation=45)
plt.show()


# In[28]:


Restrant_data1.corr()


# In[29]:


fig = plt.figure(figsize=(30, 30))
corr_map = sns.heatmap(Restrant_data1.corr(),
                      annot=True,
                      fmt='.2f',
                      cmap='coolwarm',
                      linewidth=2,
                      linecolor='green')


# In[30]:


sns.pairplot(Restrant_data1)


# ## Perform Logistic Regression

# In[31]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[32]:


Restrant_data1.columns


# In[33]:


# Define features and target variable
features = ['Average Cost for two', 'Has Table booking', 'Price range', 'Aggregate rating', 'Votes', 'Num_Cuisines']
X = Restrant_data1[features]
y = Restrant_data1['Has Online delivery']  # Assuming this is the target variable


# In[34]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform one-hot encoding for categorical variables
X_train_encoded = pd.get_dummies(X_train, columns=['Has Table booking'], drop_first=True)
X_test_encoded = pd.get_dummies(X_test, columns=['Has Table booking'], drop_first=True)


# In[35]:


# Initialize logistic regression model
log_reg_model = LogisticRegression()


# In[36]:


# Fit the model to the training data
log_reg_model.fit(X_train_encoded, y_train)


# In[37]:


# Predictions on the testing data
y_pred = log_reg_model.predict(X_test_encoded)


# In[38]:


from sklearn.metrics import accuracy_score
# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[39]:


from sklearn.metrics import confusion_matrix,classification_report
confusion_matrix(y_test,y_pred)


# In[40]:


print(classification_report(y_test,y_pred))


# In[ ]:




