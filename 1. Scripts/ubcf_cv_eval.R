# Evaluates the UBCF model parameters using K-fold cross-validation
run_ubcf_cv <- function(ratings_train, 
                        K_folds, 
                        ubcf_param_grid, 
                        top_k, 
                        good_rating_threshold) {
  
  #### CHECKING THE FUNCTION ####
  #--------------------------------------------------------------------------#
  #--------------------------------------------------------------------------#
  #--------------------------------------------------------------------------#
  # ratings_train <- read_csv("0. Data/ratings_train.csv")
  # 
  # ratings_train <- ratings_train %>%
  #   mutate(userId = as.character(userId),
  #          movieId = as.character(movieId)) %>%
  #   group_by(userId) %>%
  #   mutate(cv_fold = sample(rep(1:K_folds, length.out = n()))) %>%
  #   ungroup()
  # 
  # K_folds <- 5
  # top_k <- 10
  # good_rating_threshold <- 3 
  # ubcf_param_values <- list(
  #   method = c("cosine"),
  #   nn = c(10),
  #   normalize = c("center")
  # )
  # 
  # method = c("cosine")
  # nn = c(10)
  # normalize = c("center")
  # 
  # # Constructing the parameter combinations to be evaluated
  # ubcf_param_grid <- crossing(
  #   method = ubcf_param_values$method,
  #   nn = ubcf_param_values$nn,
  #   normalize = ubcf_param_values$normalize,
  # )
  # fold <- 1
  
  #--------------------------------------------------------------------------#
  #--------------------------------------------------------------------------#
  #--------------------------------------------------------------------------#
  
  
  # Ensure consistent types for joins and matrices
  ratings_train <- ratings_train %>%
    mutate(userId = as.character(userId),
           movieId = as.character(movieId))

  # Preparing the final results table
  ubcf_cv_results <- vector("list", length = K_folds)

  # Looping over each fold
  for (fold in seq_len(K_folds)) {
    message(sprintf("[UBCF] Starting fold %d/%d", fold, K_folds))

    # Splitting the ratings into train/test splits based on the assigned fold number
    ratings_train_tr <- ratings_train %>% filter(cv_fold != !!fold) %>% select(-cv_fold)
    ratings_train_te <- ratings_train %>% filter(cv_fold == !!fold) %>% select(-cv_fold)

    # Constructing the user - item rating matrix for the UBCF model training.
    ratings_wide_tr <- ratings_train_tr %>%
      select(userId, movieId, rating) %>%
      pivot_wider(names_from = movieId, values_from = rating)

    rating_matrix_tr <- ratings_wide_tr %>%
      column_to_rownames("userId") %>%
      as.matrix() %>%
      `mode<-`("numeric") %>%
      as("realRatingMatrix")

    # Predicting movies for the users that appear in this fold's held-out set.
    test_users <- unique(ratings_train_te$userId)
    common_users <- intersect(test_users, rownames(rating_matrix_tr))
    if (length(common_users) == 0) next
    rating_matrix_given <- rating_matrix_tr[common_users, ]

    # Performing the parameter sweep
    fold_res <- pmap_dfr(ubcf_param_grid, function(method, nn, normalize) {
        message(sprintf("[UBCF] fold=%d method=%s nn=%s normalize=%s",
                        fold, method, nn, normalize))

        # Constructing the model with the current parameters
        model <- Recommender(data = rating_matrix_tr,
                             method = "UBCF",
                             parameter = list(method = method,
                                              nn = nn,
                                              normalize = normalize))

        # Predicting the ratings each user would give to all movies based on their training data ratings
        pred_ratings <- predict(model, rating_matrix_given, type = "ratings")
        pred_mat <- as(pred_ratings, "matrix")
        pred_long <- as_tibble(pred_mat, rownames = "userId") %>%
          pivot_longer(cols = -userId,
                       names_to = "movieId",
                       values_to = "score_pred") %>%
          filter(!is.na(score_pred))

        # Preparing the true ratings to be joined with the predicted ratings
        true_long <- ratings_train_te %>%
          transmute(userId = as.character(userId),
                    movieId = as.character(movieId),
                    rating_true = rating) %>%
          filter(userId %in% common_users)

        # Joining the true ratings with the predicted ratings
        eval_long <- true_long %>%
          inner_join(pred_long, by = c("userId", "movieId"))

        # Removing the training data movies from the movie candidates
        seen_long <- ratings_train_tr %>%
          transmute(userId = as.character(userId),
                    movieId = as.character(movieId)) %>%
          filter(userId %in% common_users)

        candidate_pred <- pred_long %>%
          anti_join(seen_long, by = c("userId", "movieId"))

        # Computing RMSE on held-out ratings and Precision/Recall@k on candidate items
        metric_res <- compute_eval_metrics(eval_long = eval_long,
                                           candidate_pred = candidate_pred,
                                           common_users = common_users,
                                           top_k = top_k,
                                           good_rating_threshold = good_rating_threshold)

        # Constructing the results entry with the evaluation results
        tibble(model = "UBCF",
               params = list(
                 list(method = method,
                      nn = nn,
                      normalize = normalize)),
               fold = fold,
               rmse = metric_res$rmse,
               precision_at_k = metric_res$precision_at_k,
               recall_at_k = metric_res$recall_at_k)
      }
    )
    # Adding the results to the appropriate fold
    ubcf_cv_results[[fold]] <- fold_res
  }
  # Joining the final results
  bind_rows(ubcf_cv_results)
}

