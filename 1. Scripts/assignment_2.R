library(tidyverse)
library(recommenderlab)
library(recosystem)

options(scipen = 999)
set.seed(12345)

rm(list = ls())

# Loading necessary functions
source("1. Scripts/evaluation_metrics.R")
source("1. Scripts/ubcf_cv_eval.R")
source("1. Scripts/libmf_cv_eval.R")

# Reading the Movies dataset
movies <- read_csv("0. Data/movies.csv")
ratings_train <- read_csv("0. Data/ratings_train.csv")

#--------------------------------------------------------------#
####           Task 0 - Understanding the dataset           ####
#--------------------------------------------------------------#
# Checking for any missing values
sum(is.na(movies))          # 0
sum(is.na(ratings_train))   # 0

# Descriptive statistics for the movies dataset
n_distinct(movies$movieId) # 9742 unique movie IDs

n_distinct(ratings_train$userId) # 600 unique users
n_distinct(ratings_train$movieId) # 9680 unique movies rated) (Not all movies have ratings!)

range(ratings_train$rating) # Ratings range from 0.5 to 5.0

# Checking number of user reviews per movie
# Ideally there should be only one review per user per movie.
user_reviews_per_movie <- ratings_train %>%
  group_by(userId, movieId) %>%
  summarise(n_reviews = n(), .groups = "drop") %>%
  ungroup()
all(user_reviews_per_movie$n_reviews == 1) # TRUE, each user has only one review per movie

# Checking the average rating for each movie
movie_average_rating <- ratings_train %>%
  group_by(movieId) %>%
  summarise(n_reviews = n(),
            average_rating = mean(rating)) %>%
  ungroup() %>%
  left_join(movies, by = "movieId")

# Histogram of average movie rating
avg_rating_plot <- ggplot(movie_average_rating, aes(x = average_rating)) +
  geom_histogram(binwidth = 0.5,
                 boundary = 0.5,
                 closed = "left",
                 fill = "lightblue",
                 color = "black") +
  scale_x_continuous(limits = c(0.5, 5), breaks = seq(0.5, 5, by = 0.5)) +
  labs(title = "Distribution of Average Movie Ratings",
       x = "Average Rating",
       y = "Number of Movies") +
  theme_minimal()

avg_rating_plot

ggsave(filename = "2. Figures/average_movie_ratings.png",
       plot = avg_rating_plot,
       width = 6,
       height = 6,
       dpi = 300)

round(mean(movie_average_rating$average_rating, na.rm = TRUE), 2) # 3.26 average rating across all movies with a rating
round(sd(movie_average_rating$average_rating, na.rm = TRUE), 2) # 0.87 standard deviation of average ratings across movies


#--------------------------------------------------#
####     Task 1 - Model tuning & Evaluation     ####
#--------------------------------------------------#
# Specifying global training parameters
train_fraction <- 0.8
good_rating_threshold <- 3 # > 3 ratings are considered good
top_k <- 10
K_folds <- 5


# Splitting the movie ratings into folds
ratings_train_cv <- ratings_train %>%
  mutate(userId = as.character(userId),
         movieId = as.character(movieId)) %>%
  group_by(userId) %>%
  mutate(cv_fold = sample(rep(1:K_folds, length.out = n()))) %>%
  ungroup()

#----------------------------------------------------------------------#
#       Specifying model parameters for the model parameter sweep      #
#----------------------------------------------------------------------#
# Specifying parameters for the UBCF (User-Based Collaborative Filtering) model
ubcf_param_values <- list(
  method = c("cosine", "pearson", "jaccard"),
  nn = c(10, 30, 50, 100),
  normalize = c("center", "Z-score")
)

# Constructing the parameter combinations to be evaluated
ubcf_param_grid <- crossing(
  method = ubcf_param_values$method,
  nn = ubcf_param_values$nn,
  normalize = ubcf_param_values$normalize,
)

# Specifying parameters for the LIBMF (Matrix Factorization) model
libmf_param_values <- list(
  dim = c(10, 25, 50, 75, 100, 200),
  niter = c(30, 50, 80),
  costp_l2 = c(0.005, 0.01, 0.05, 0.1),
  costq_l2 = c(0.005, 0.01, 0.05, 0.1),
  lrate =  c(0.01, 0.03, 0.05, 0.1)
)

# Constructing the parameter combinations to be evaluated
libmf_param_grid <- crossing(
  dim = libmf_param_values$dim,
  niter = libmf_param_values$niter,
  costp_l2 = libmf_param_values$costp_l2,
  costq_l2 = libmf_param_values$costq_l2,
  lrate = libmf_param_values$lrate
)


#-------------------------------------------------#
#      Running parameter sweep on the models      #
#-------------------------------------------------#

# Note: Uncommenting the following section takes a long time to run
#-------------------------------------------------------------------------------------#
# message(sprintf("UBCF configurations: %d", nrow(ubcf_param_grid)))
# ubcf_results <- run_ubcf_cv(
#   ratings_train = ratings_train_cv,
#   K_folds = K_folds,
#   ubcf_param_grid = ubcf_param_grid,
#   top_k = top_k,
#   good_rating_threshold = good_rating_threshold
# )
# 
# message(sprintf("LIBMF configurations: %d", nrow(libmf_param_grid)))
# libmf_results <- run_libmf_cv(
#   ratings_train = ratings_train_cv,
#   K_folds = K_folds,
#   libmf_param_grid = libmf_param_grid,
#   top_k = top_k,
#   good_rating_threshold = good_rating_threshold
# )


#----------------------------------#
#      Final combined results      #
#----------------------------------#
# # Display average metrics per model configuration.
# fold_level_results <- bind_rows(ubcf_results, libmf_results)
# 
# # Aggregating average results across folds per model configuration
# model_tuning_results <- fold_level_results %>%
#   group_by(model, params) %>%
#   summarise(rmse = mean(rmse, na.rm = TRUE),
#             precision_at_k = mean(precision_at_k, na.rm = TRUE),
#             recall_at_k = mean(recall_at_k, na.rm = TRUE),
#             .groups = "drop") %>%
#   unnest_wider(params) %>%
#   arrange(model, rmse)
#-------------------------------------------------------------------------------------#

# save(model_tuning_results, file = "0. Data/tuning_parameters.Rdata")
# write_csv(model_tuning_results, "0. Data/tuning_parameters.csv", na = "")

load("0. Data/tuning_parameters.Rdata")
View(model_tuning_results)

# Best model configurations based on Precision/Recall@K #
#-------------------------------------------#
# model | method | nn | normalize | rmse      | precision_at_k | recall_at_k
# UBCF  | cosine | 10 | Z-score   | 1.2056635 | 0.0227333333   | 0.0688606322

# model | dim | niter | costp_l2 | costq_l2 | lrate | rmse      | precision_at_k | recall_at_k
# LIBMF | 10  | 30    | 0.100    | 0.005    | 0.01  | 0.9889373 | 0.09660000     | 0.05933533
#-------------------------------------------#


#--------------------------------------------------------------#
####     Task 2 - Making Predictions for Specific Users     ####
#--------------------------------------------------------------#
# Generating predictions for the "ratings_test.csv" users using the LIBMF model with the following configuration:
# dim | niter | costp_l2 | costq_l2 | lrate 
# 10  | 30    | 0.100    | 0.005    | 0.01  

# Loading the users who will receive recommendations
ratings_test <- read_csv("0. Data/ratings_test.csv") %>%
  mutate(userId = as.character(userId))

# Ensure consistent types for joins and matrices
ratings_train_mf <- ratings_train %>%
  mutate(userId = as.character(userId),
         movieId = as.character(movieId)) %>%
  select(userId, movieId, rating)

# Creating a user and item map with the IDs in a sequence which map to the original IDs
user_levels <- sort(unique(ratings_train_mf$userId))
item_levels <- sort(unique(ratings_train_mf$movieId))
user_map <- setNames(seq_along(user_levels), user_levels)
item_map <- setNames(seq_along(item_levels), item_levels)

# Creating the model's instance
libmf_model <- Reco()

# Storing the training data in memory
train_data_mf <- data_memory(user_index = unname(user_map[ratings_train_mf$userId]),
                             item_index = unname(item_map[ratings_train_mf$movieId]),
                             rating = ratings_train_mf$rating,
                             index1 = TRUE)

# Training the LIBMF model with the current parameters
libmf_model$train(train_data_mf,
                  opts = list(dim = 10,
                              niter = 30,
                              costp_l2 = 0.1,
                              costq_l2 = 0.005,
                              lrate = 0.01,
                              verbose = FALSE)
                  )

# Keeping known users who will receive recommendations
known_test_users <- ratings_test %>%
  filter(userId %in% names(user_map)) %>%
  pull(userId) %>%
  unique()

# Identifying users who do not exist in the training data (cold start users)
cold_start_users <- setdiff(unique(ratings_test$userId), known_test_users)
all_items <- sort(unique(ratings_train_mf$movieId))

# Keeping the known user-item combinations from the training data
seen_long <- ratings_train_mf %>%
  select(userId, movieId)

# Defining all possible user-item combinations for the recommendations
known_candidate_pairs <- expand_grid(userId = known_test_users,
                                     movieId = all_items) %>%
  anti_join(seen_long, by = c("userId", "movieId"))

if (nrow(known_candidate_pairs) > 0) {
  # Storing the prediction data into memory
  pred_request <- data_memory(user_index = unname(user_map[known_candidate_pairs$userId]),
                              item_index = unname(item_map[known_candidate_pairs$movieId]),
                              index1 = TRUE)
  
  # Using the model to score the items for each user
  known_candidate_pairs$score <- libmf_model$predict(pred_request, out_memory())
} else {
  known_candidate_pairs$score <- numeric(0)
}

# Keeping the top 10 model recommendations for each user
known_top10 <- known_candidate_pairs %>%
  group_by(userId) %>%
  arrange(desc(score), .by_group = TRUE) %>%
  slice_head(n = 10) %>%
  mutate(rec_rank = row_number()) %>%
  ungroup() %>%
  select(userId, rec_rank, movieId)


# Determining the most popular movies based on Bayesian weighted rating
# The idea is that movies with a lot of positive reviews should be recommended to cold start users
# WR = (v / (v + m)) * R + (m / (v + m)) * C
# v: number of reviews for movie, 
# R: movie mean rating, 
# C: global mean rating
# m: minimum reviews prior (set as the 90th percentile of review counts to prioritize movies with a lot of reviews)
movie_stats <- ratings_train %>%
  group_by(movieId) %>%
  summarise(
    n_reviews = n(),
    average_rating = mean(rating),
    .groups = "drop"
  )

C_global <- mean(ratings_train$rating, na.rm = TRUE)
m_prior <- as.numeric(quantile(movie_stats$n_reviews, probs = 0.9, na.rm = TRUE))

# Calculating the Bayesian score and keeping the top 10 movies
cold_fallback_ids <- movie_stats %>%
  mutate(bayesian_score = (n_reviews / (n_reviews + m_prior)) * average_rating + (m_prior / (n_reviews + m_prior)) * C_global) %>%
  arrange(desc(bayesian_score), desc(n_reviews), desc(average_rating), movieId) %>%
  slice_head(n = 10) %>%
  pull(movieId) %>%
  as.character()

# Creating a table with the recommended movies for the cold start users
cold_top10 <- if (length(cold_start_users) > 0) {
  expand_grid(userId = cold_start_users, rec_rank = 1:10) %>%
    mutate(movieId = cold_fallback_ids[rec_rank])
} else {
  tibble(userId = character(), rec_rank = integer(), movieId = character())
}

# Combining existing and cold start users into a single table
all_top10 <- bind_rows(known_top10, cold_top10) %>%
  mutate(recommendation_col = paste0("recommendation", rec_rank)) %>%
  select(userId, recommendation_col, movieId)

# Converting the recommendations table to wide for the final table
ratings_test_filled <- ratings_test %>%
  select(userId) %>%
  left_join(all_top10 %>% pivot_wider(names_from = recommendation_col, values_from = movieId), by = "userId") %>%
  mutate(across(starts_with("recommendation"), as.integer))

# Exporting the recommendations
write_csv(ratings_test_filled, "0. Data/ratings_test.csv", na = "")

