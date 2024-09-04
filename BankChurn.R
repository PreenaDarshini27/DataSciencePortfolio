# Loading required libraries
library(Amelia)
library(ggplot2)
library(corrplot)
library(caret)
library(pscl)
library(pROC)

# Reading data from csv file and storing in a dataframe
df <- read.csv('Bank Customer Churn.csv')

# Basic Data Exploration
dim(df)
nrow(df)
ncol(df)
colnames(df)
head(df, 20)
str(df)
summary(df)
sapply(df, function(x) sum(is.na(x)))

# Missing Data Visualization
missmap(df, main = "Missing Map", col = c("yellow", "black"))

# ---Exploratory Data Analysis(EDA)---
# Numerical Variables
ggplot(df, aes(x = customer_id)) + 
  geom_histogram(color = "black", fill = "lightblue", bins = 30) +
  theme_minimal() +
  labs(title = "Distribution of Customer ID")

ggplot(df, aes(x = credit_score)) + 
  geom_histogram(color = "black", fill = "lightblue", bins = 30) +
  theme_minimal() +
  labs(title = "Distribution of Credit Score")

ggplot(df, aes(x = age)) + 
  geom_histogram(color = "black", fill = "lightblue", bins = 30) +
  theme_minimal() +
  labs(title = "Distribution of Age")

ggplot(df, aes(x = tenure)) + 
  geom_histogram(color = "black", fill = "lightblue", bins = 30) +
  theme_minimal() +
  labs(title = "Distribution of Tenure")

ggplot(df, aes(x = balance)) + 
  geom_histogram(color = "black", fill = "lightblue", bins = 30) +
  theme_minimal() +
  labs(title = "Distribution of Balance")

ggplot(df, aes(x = products_number)) + 
  geom_histogram(color = "black", fill = "lightblue", bins = 30) +
  theme_minimal() +
  labs(title = "Distribution of Products Number")

ggplot(df, aes(x = estimated_salary)) + 
  geom_histogram(color = "black", fill = "lightblue", bins = 30) +
  theme_minimal() +
  labs(title = "Distribution of Estimated Salary")

# Categorical Variables Transformation
df$credit_card <- factor(df$credit_card, levels = c(0, 1))
df$active_member <- factor(df$active_member, levels = c(0, 1))
df$country <- as.factor(df$country)
df$gender <- as.factor(df$gender)
df$churn <- as.factor(df$churn)

# Bar Plots for Categorical Variables
ggplot(df, aes(x = country, fill = country)) +
  geom_bar() +
  labs(title = "Bar Graph of Country Distribution", x = "Country", y = "Count") +
  theme_classic()

ggplot(df, aes(x = gender, fill = gender)) +
  geom_bar() +
  labs(title = "Bar Graph of Gender Distribution", x = "Gender", y = "Count") +
  theme_classic()

ggplot(df, aes(x = churn, fill = churn)) +
  geom_bar() +
  labs(title = "Bar Graph of Churn Distribution", x = "Churn", y = "Count") +
  theme_classic()

ggplot(df, aes(x = credit_card, fill = credit_card)) +
  geom_bar() +
  labs(title = "Bar Graph of Credit Card Distribution", x = "Credit Card", y = "Count") +
  theme_classic()

ggplot(df, aes(x = active_member, fill = active_member)) +
  geom_bar() +
  labs(title = "Bar Graph of Active Member Distribution", x = "Active Member", y = "Count") +
  theme_classic()

# Bivariate Analysis
# customer_id vs churn
ggplot(df, aes(x = customer_id, y = churn)) + 
  geom_point(alpha = 0.5) +
  theme_minimal() +
  labs(title = "Customer ID vs Churn")

# credit_score vs churn
ggplot(df, aes(x = credit_score, y = churn)) + 
  geom_point(alpha = 0.5) +
  theme_minimal() +
  labs(title = "Credit Score vs Churn")

# age vs churn
ggplot(df, aes(x = age, y = churn)) + 
  geom_point(alpha = 0.5) +
  theme_minimal() +
  labs(title = "Age vs Churn")

# tenure vs churn
ggplot(df, aes(x = tenure, y = churn)) + 
  geom_point(alpha = 0.5) +
  theme_minimal() +
  labs(title = "Tenure vs Churn")

# balance vs churn
ggplot(df, aes(x = balance, y = churn)) + 
  geom_point(alpha = 0.5) +
  theme_minimal() +
  labs(title = "Balance vs Churn")

# products_number vs churn
ggplot(df, aes(x = products_number, y = churn)) + 
  geom_point(alpha = 0.5) +
  theme_minimal() +
  labs(title = "Products Number vs Churn")

# estimated_salary vs churn
ggplot(df, aes(x = estimated_salary, y = churn)) + 
  geom_point(alpha = 0.5) +
  theme_minimal() +
  labs(title = "Estimated Salary vs Churn")


# Outlier Handling and Feature Engineering
# Function to calculate and print outliers based on IQR method
print_outliers <- function(data, variable) {
  Q1 <- quantile(data[[variable]], 0.25)
  Q3 <- quantile(data[[variable]], 0.75)
  IQR <- Q3 - Q1
  lower_bound <- Q1 - 1.5 * IQR
  upper_bound <- Q3 + 1.5 * IQR
  outliers <- data[[variable]][data[[variable]] < lower_bound | data[[variable]] > upper_bound]
  cat(paste("Outliers for", variable, ":\n"))
  print(outliers)
}

# Applying the function to each numerical variable
numerical_vars <- c("credit_score", "age", "tenure", "balance", "products_number", "estimated_salary")
for (var in numerical_vars) {
  print_outliers(df, var)
}

# Binning for Credit Score and Age
credit_score_bins <- c(-Inf, 350, 550, 700, 850, Inf)
df$credit_score_binned <- cut(df$credit_score, breaks = credit_score_bins, labels = c("Very Low", "Low", "Medium", "High", "Very High"))

age_bins <- c(-Inf, 35, 50, 65, Inf)
df$age_binned <- cut(df$age, breaks = age_bins, labels = c("Young", "Middle-Aged", "Senior", "Elder"))

# Log Transformation to normalise data
df$log_products_number <- log(df$products_number + 1) # Adding 1 to avoid log(0)



# Splitting data into Train-Test sets
set.seed(22238590)
train_index <- createDataPartition(df$churn, p = 0.7, list = FALSE)
train_data <- df[train_index, ]
test_data <- df[-train_index, ]

# Logistic Regression Model
log_model <- glm(churn ~ credit_score_binned + age_binned + log_products_number + credit_card + active_member + country + gender + balance,
                 data = train_data, family = "binomial")

# Model Performance Evaluation
# Train Set
train_pred <- predict(log_model, type = "response", newdata = train_data)
train_roc <- roc(train_data$churn, train_pred)
train_auc <- as.numeric(auc(train_roc))
print(paste("Train AUC:", train_auc))

# Test Set
test_pred <- predict(log_model, newdata = test_data, type = "response")
test_roc <- roc(test_data$churn, test_pred)
test_auc <- as.numeric(auc(test_roc))
print(paste("Test AUC:", test_auc))

# Confusion Matrix
conf_matrix <- confusionMatrix(table(round(test_pred), test_data$churn))
print(conf_matrix)

# ROC Curve
plot(train_roc, col = "blue", main = "ROC Curve")
lines(test_roc, col = "red")

