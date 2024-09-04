# Loading necessary libraries
library(ggplot2)
library(dplyr)
library(Amelia)
library(cluster)
library(factoextra)
library(clValid)
library(stats)
# Reading in the data and storing it in a dataframe
df <- read.csv('filterReviews1.csv')

# Basic Exploratory Data Analysis
dim(df)
colnames(df)
str(df)
summary(df)
head(df, 20)

# Visualization for Missing Values
missmap(df, main = "Missing Map", col = c("purple", "orange"), legend = FALSE)

# Data Preprocessing and cleaning
df_clean <- df %>%
  mutate(HelpfulnessRatio = ifelse(HelpfulnessDenominator > 0, 
                                   HelpfulnessNumerator / HelpfulnessDenominator, 
                                   0)) %>%
  mutate(HelpfulnessNumeratorLog = log1p(HelpfulnessNumerator),
         HelpfulnessDenominatorLog = log1p(HelpfulnessDenominator))


# Normalizing the log-transformed data
df_normalized <- as.data.frame(scale(df_clean[, c("HelpfulnessNumeratorLog", "HelpfulnessDenominatorLog")]))

# Visualizations
ggplot(df, aes(x = Score)) +
  geom_histogram(binwidth = 1, fill = "blue", color = "black") +
  ggtitle("Distribution of Scores")

ggplot(df, aes(x = HelpfulnessNumerator)) + 
  geom_histogram(binwidth = 1, fill = "blue") +
  ggtitle("Distribution of Helpfulness Numerator")

ggplot(df, aes(x = HelpfulnessDenominator)) + 
  geom_histogram(binwidth = 1, fill = "green") +
  ggtitle("Distribution of Helpfulness Denominator")

# Histograms for log-transformed variables
ggplot(df_clean, aes(x = HelpfulnessNumeratorLog)) + 
  geom_histogram(binwidth = 0.2, fill = "blue", color = "black") +
  ggtitle("Histogram of Log-transformed Helpfulness Numerator")

ggplot(df_clean, aes(x = HelpfulnessDenominatorLog)) + 
  geom_histogram(binwidth = 0.2, fill = "green", color = "black") +
  ggtitle("Histogram of Log-transformed Helpfulness Denominator")

# Boxplots to check for outliers in the log-transformed variables
ggplot(df_clean, aes(y = HelpfulnessNumeratorLog)) + 
  geom_boxplot(fill = "blue") +
  ggtitle("Boxplot for Log-transformed Helpfulness Numerator")

ggplot(df_clean, aes(y = HelpfulnessDenominatorLog)) + 
  geom_boxplot(fill = "green") +
  ggtitle("Boxplot for Log-transformed Helpfulness Denominator")

# Identifying outliers in HelpfulnessNumeratorLog
Q1_Numerator <- quantile(df_clean$HelpfulnessNumeratorLog, 0.25)
Q3_Numerator <- quantile(df_clean$HelpfulnessNumeratorLog, 0.75)
IQR_Numerator <- Q3_Numerator - Q1_Numerator
outliers_Numerator <- df_clean %>%
  filter(HelpfulnessNumeratorLog < (Q1_Numerator - 1.5 * IQR_Numerator) |
           HelpfulnessNumeratorLog > (Q3_Numerator + 1.5 * IQR_Numerator))

# Identifying outliers in HelpfulnessDenominatorLog
Q1_Denominator <- quantile(df_clean$HelpfulnessDenominatorLog, 0.25)
Q3_Denominator <- quantile(df_clean$HelpfulnessDenominatorLog, 0.75)
IQR_Denominator <- Q3_Denominator - Q1_Denominator
outliers_Denominator <- df_clean %>%
  filter(HelpfulnessDenominatorLog < (Q1_Denominator - 1.5 * IQR_Denominator) |
           HelpfulnessDenominatorLog > (Q3_Denominator + 1.5 * IQR_Denominator))

# Printing the outliers
print(outliers_Numerator)
print(outliers_Denominator) # Valid information so the outliers will be considered

# Determining the optimal number of clusters using the Elbow method
set.seed(22238590) # Set a seed for reproducibility
fviz_nbclust(df_normalized, kmeans, method = "wss") +
  theme_minimal() +
  ggtitle("Elbow Method for Determining Optimal k")

# Assuming 3 is the optimal number of clusters based on the Elbow method
optimal_clusters <- 3
final_kmeans <- kmeans(df_normalized, centers = optimal_clusters, nstart = 25)
# Silhouette plot to evaluate clustering
silhouette_score <- silhouette(final_kmeans$cluster, dist(df_normalized))
plot(silhouette_score)
# Silhouette Score
avg_silhouette_width <- mean(silhouette_score[, "sil_width"])
print(paste("Average Silhouette Width:", avg_silhouette_width))
# Assigning clusters
df_clean$cluster <- as.factor(final_kmeans$cluster)

# Cluster Interpretation
# Calculating summary statistics for each cluster
cluster_summary <- df_clean %>%
  group_by(cluster) %>%
  summarise(across(c(HelpfulnessNumerator, HelpfulnessDenominator), list(mean = mean, median = median)))
print(cluster_summary)

# Adding the cluster assignments to the dataframe
df_normalized$cluster <- as.factor(final_kmeans$cluster)

# Plotting the K-means clusters of reviews
ggplot(df_normalized, aes(x = HelpfulnessNumeratorLog, y = HelpfulnessDenominatorLog, color = cluster)) +
  geom_point(alpha = 0.5) +
  theme_minimal() +
  ggtitle("K-means Clustering of Reviews (Normalized Data)")

