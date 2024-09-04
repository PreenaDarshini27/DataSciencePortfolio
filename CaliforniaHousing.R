# Loading required libraries
library(ggplot2)
library(dplyr)
library(fastDummies)
library(caret)
library(MASS)
library(car)
library(corrplot)
library(lmtest)
library(leaflet)
# Reading data from csv file and storing in a dataframe
df <- read.csv("housing.csv")
# Basic Data Exploration
str(df)
summary(df)
dim(df)
# Omitting missing values if any
df <- na.omit(df)
# Histograms for numerical variables
hist(df$median_house_value, main="Distribution of Median House Value", xlab="Median House Value")
hist(df$longitude, main="Distribution of Longitude", xlab="Longitude")
hist(df$latitude, main="Distribution of Latitude", xlab="Latitude")
hist(df$housing_median_age, main="Distribution of housing_median_age", xlab="Housing median age")
hist(df$total_rooms, main="Distribution of total_rooms", xlab="total rooms")
hist(df$population, main="Distribution of population", xlab="population")
hist(df$households, main="Distribution of households", xlab="households")
hist(df$median_income, main="Distribution of median_income", xlab="median_income")
# Converting categorical varriables into factor type and creating dummy variables
print(unique(df[["ocean_proximity"]]))
df$ocean_proximity <- as.factor(df$ocean_proximity)
df <- dummy_cols(df, select_columns = "ocean_proximity", remove_first_dummy = TRUE)

# Applying transformations to variables with skewness and histograms
df$median_house_value_log <- log(df$median_house_value+1)
hist(df$median_house_value_log, main="Log Transformed Distribution of Median House Value", xlab="Log of Median House Value")
df$total_rooms_log <- log(df$total_rooms + 1)
hist(df$total_rooms_log, main = "Log Transformed Distribution of Total Rooms",
     xlab = "Log of Total Rooms", breaks = 50)
df$population_log <- log(df$population + 1)
hist(df$population_log, main = "Log Transformed Distribution of Population",
     xlab = "Log of Population", breaks = 50)
df$households_log <- log(df$households + 1)
hist(df$households_log, main = "Log Transformed Distribution of Households",
     xlab = "Log of Households", breaks = 50)
df$median_income_log <- log(df$median_income + 1)
hist(df$median_income_log, main = "Log Transformed Distribution of Median Income",
     xlab = "Log of Median Income", breaks = 50)

# Plotting a map for price distribution
df$Value_Category <- ifelse(df$median_house_value > median(df$median_house_value, na.rm = TRUE), "High", "Low")
map <- leaflet(df) %>%
  addTiles() %>%
  addCircleMarkers(lng = ~longitude, lat = ~latitude, radius = 5, color = "white",
                   fillColor = ~ifelse(Value_Category == "High", "red", "green"),
                   fillOpacity = 0.8, popup = ~paste("Median House Value: $", median_house_value)) %>%
  addLegend(colors = c("green", "red"), labels = c("Low Median House Value", "High Median House Value"),
            title = "Value Category", opacity = 0.8)
print(map)
# Identifying outliers using IQR Method
find_outliers_indices <- function(data) {
  Q1 <- quantile(data, 0.25)
  Q3 <- quantile(data, 0.75)
  IQR <- Q3 - Q1
  lower_bound <- Q1 - 1.5 * IQR
  upper_bound <- Q3 + 1.5 * IQR
  outliers_indices <- which(data < lower_bound | data > upper_bound)
  return(outliers_indices)
}

outliers_indices <- find_outliers_indices(df$median_house_value_log)

# Deleting rows with outliers
df <- df[-outliers_indices, ]

# Checking the new dimensions of the dataframe
dim(df)
summary(df)
str(df)

numerical_vars <- c("longitude", "latitude", "housing_median_age", "total_rooms_log", 
                    "population_log", "households_log", "median_income_log", "median_house_value_log", "total_bedrooms")

numerical_df <- df[numerical_vars]

# Calculating the correlation matrix
cor_matrix <- cor(numerical_df)
print(cor_matrix)
# Creating a correlation heatmap
corrplot(cor_matrix, method = "color")
# Setting seed for reproducibility and splitting data into train and test sets
set.seed(22238590) 
index <- createDataPartition(df$median_house_value_log, p = 0.8, list = FALSE)
train <- df[index, ]
test <- df[-index, ]

# Multiple linear regression model
mlr_formula <- as.formula("median_house_value_log ~ longitude + latitude  + housing_median_age +  median_income_log + 
                            ocean_proximity_INLAND + ocean_proximity_ISLAND + `ocean_proximity_NEAR BAY` + 
                            `ocean_proximity_NEAR OCEAN`")

# Fitting the model on train data
mlr_model <- lm(mlr_formula, data = train)

# Summary of the model
summary(mlr_model)

# Predictions on the train set
train$predicted <- predict(mlr_model, newdata = train)
# Evaluating the model on train data
train$residuals <- train$predicted - train$median_house_value_log


# Evaluation Metrics on Train Data
train_rmse <- sqrt(mean(train$residuals^2))
train_r_squared <- cor(train$predicted, train$median_house_value_log)^2

# Printing the evaluation metrics
cat("Train RMSE:", train_rmse, "\n")
cat("Train R-squared:", train_r_squared, "\n")

# P-P Plot for Normality of Residuals
residuals_model <- residuals(mlr_model)
qqnorm(residuals_model, main = "P-P Plot for Model Residuals")
qqline(residuals_model, col = "red")

# Cook's Distance
cooks_dist <- cooks.distance(mlr_model)
plot(cooks_dist, main = "Cook's Distance", type = "h")
abline(h = 4/length(cooks_dist), col = "red")

# Predictions on the test data
test$predicted <- predict(mlr_model, newdata = test)
# Evaluating the model on test data
test$residuals <- test$predicted - test$median_house_value_log
# Evaluation Metrics on test data
test_rmse <- sqrt(mean(test$residuals^2))
test_r_squared <- cor(test$predicted, test$median_house_value_log)^2

cat("Test RMSE:", test_rmse, "\n")
cat("Test R-squared:", test_r_squared, "\n")

# Histogram of Residuals for Train Data
hist(train$residuals, main = "Histogram of Residuals (Train Data)",
     xlab = "Residuals", ylab = "Frequency")

# Histogram of Residuals for Test Data
hist(test$residuals, main = "Histogram of Residuals (Test Data)",
     xlab = "Residuals", ylab = "Frequency")
# Plotting actual vs. predicted values for the train data
plot(train$median_house_value_log, train$predicted, main = "Actual vs. Predicted (Train Data)",
     xlab = "Actual Median House Value (log)",
     ylab = "Predicted Median House Value (log)")
abline(0, 1, col = "red")  # Add a 45-degree line for reference

# Plotting actual vs. predicted values for the test data
plot(test$median_house_value_log, test$predicted, main = "Actual vs. Predicted (Test Data)",
     xlab = "Actual Median House Value (log)",
     ylab = "Predicted Median House Value (log)")
abline(0, 1, col = "red")  # Add a 45-degree line for reference

