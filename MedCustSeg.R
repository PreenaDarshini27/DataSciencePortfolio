# Loading required libraries
library(Amelia)
library(ggplot2)
library(corrplot)
library(caret)
library(dplyr)
library(cluster)
library(factoextra)

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

# Converting non-binary categorical variables to factor type
df$country <- as.factor(df$country)
df$gender <- as.factor(df$gender)

# Creating dummy variables for categorical variables that are not binary
dummy_vars <- model.matrix(~ country + gender - 1, data=df)
df <- cbind(df, dummy_vars)

# Dropping the original categorical columns that are not binary
df <- df[, !colnames(df) %in% c("country", "gender")]
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


# Checking if binary categorical variables are numeric
df$credit_card <- as.numeric(df$credit_card)
df$active_member <- as.numeric(df$active_member)
df$products_number <- as.numeric(df$products_number)
str(df)

# Selecting the numerical features required for scaling
numerical_features <- df[, c(
  "credit_score", "age", "tenure", "balance", "products_number",
  "credit_card", "active_member", "estimated_salary"
)]

# Scaling the numerical features
df_scaled <- scale(numerical_features)

# Determining the optimal number of clusters
set.seed(22238590)  # setting for reproducibility


# Determining the optimal number of clusters using the Elbow Method
elbow <- fviz_nbclust(df_scaled, pam, method = "wss", k.max = 5)

# Plotting the Elbow Method
print(elbow)


# Based on the Elbow Method, we choose k=3
optimal_k <- 3  

# Running K-medoids clustering with the chosen k=3
kmedoids_result <- pam(df_scaled, optimal_k)

# Adding cluster assignments
df$cluster <- kmedoids_result$clustering
cluster_counts <- table(df$cluster)
barplot(cluster_counts, main = "Cluster Distribution", xlab = "Cluster", ylab = "Count", col = rainbow(optimal_k))
# Comparing means across clusters for numerical features
cluster_means <- aggregate(
  df[, c("credit_score", "age", "balance", "estimated_salary")],
  by = list(df$cluster),
  FUN = mean
)
print(cluster_means)

# Scatterplot of Cluster centers 
cluster_centers <- kmedoids_result$medoids
colnames(cluster_centers) <- colnames(df_scaled)
cluster_centers_df <- as.data.frame(cluster_centers)

# Plotting the cluster centers
fviz_cluster(kmedoids_result, geom = "point", ellipse = FALSE)
silhouette_score <- silhouette(kmedoids_result$cluster, dist(df_scaled))
fviz_silhouette(silhouette_score)
