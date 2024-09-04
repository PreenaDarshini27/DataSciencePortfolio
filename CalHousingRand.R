# Loading required libraries
library(ggplot2)
library(dplyr)
library(fastDummies)
library(caret)
library(MASS)
library(car)
library(corrplot)
library(lmtest)
library(randomForest)
library(doParallel)
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
summary(df)
dim(df)
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
df$Value_Category <- ifelse(df$median_house_value > median(df$median_house_value, na.rm = TRUE), "High", "Low")
# Plotting a map for price distribution
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


# Setting seed for reproducibility
set.seed(22238590)

# Splitting data into train and test sets
index <- createDataPartition(df$median_house_value_log, p = 0.8, list = FALSE)
train_data <- df[index, ]
test_data <- df[-index, ]

outcome_var <- "median_house_value_log"

predictor_vars <- c("longitude", "latitude", "housing_median_age", "total_rooms_log", 
                    "population_log", "households_log", "median_income_log", 
                    "ocean_proximity_INLAND", "ocean_proximity_ISLAND", 
                    "ocean_proximity_NEAR BAY", "ocean_proximity_NEAR OCEAN")

# Training the Random Forest model
rf_model <- randomForest(train_data[, predictor_vars], 
                         train_data[, outcome_var],
                         ntree = 100,           # Number of trees in the forest
                         mtry = sqrt(length(predictor_vars)),  # Number of variables randomly sampled as candidates at each split
                         importance = TRUE,      # Calculate variable importance
                         nodesize = 5,           # Minimum size of terminal nodes
                         na.action = na.omit)    # Handling missing values

# Printing summary of the Random Forest model
print(rf_model)

# Making predictions on the test data
test_data$predicted <- predict(rf_model, newdata = test_data[, predictor_vars])

# Calculating RMSE on test data
rmse_test <- sqrt(mean((test_data$predicted - test_data[, outcome_var])^2))
print(paste("RMSE on Test Set:", rmse_test))

# Calculating R-squared on test data
r_squared_test <- cor(test_data$predicted, test_data[, outcome_var])^2
print(paste("R-squared on Test Set:", r_squared_test))

# Residuals Plot
residuals <- test_data$median_house_value_log - test_data$predicted
ggplot(test_data, aes(x = predicted, y = residuals)) +
  geom_point(alpha = 0.5) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  labs(title = "Residuals vs Predicted", x = "Predicted Values", y = "Residuals")

# Variable Importance Plot
importance_df <- as.data.frame(importance(rf_model))
importance_df$Variable <- rownames(importance_df)
ggplot(importance_df, aes(x = reorder(Variable, IncNodePurity), y = IncNodePurity)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Variable Importance", x = "Variables", y = "Importance")

# Prediction vs Actual Plot
ggplot(test_data, aes(x = median_house_value_log, y = predicted)) +
  geom_point(alpha = 0.5) +
  geom_abline(color = "red") +
  labs(title = "Actual vs Predicted", x = "Actual Values", y = "Predicted Values")

