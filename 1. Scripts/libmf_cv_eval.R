run_libmf_cv <- function(ratings_train,
                         K_folds,
                         libmf_param_grid,
                         top_k,
                         good_rating_threshold) {
  
  #### CHECKING THE FUNCTION ####
  #--------------------------------------------------------------------------#
  #--------------------------------------------------------------------------#
  #--------------------------------------------------------------------------#
  # ratings_train <- read_csv("0. Data/ratings_train.csv")
  # 
  # K_folds <- 5
  # 
  # ratings_train <- ratings_train %>%
  #   mutate(userId = as.character(userId),
  #          movieId = as.character(movieId)) %>%
  #   group_by(userId) %>%
  #   mutate(cv_fold = sample(rep(1:K_folds, length.out = n()))) %>%
  #   ungroup()
  # 
  # 
  # top_k <- 10
  # good_rating_threshold <- 3
  # 
  # libmf_param_values <- list(dim = c(10),
  #                            niter = c(30),
  #                            costp_l2 = c(0.01),
  #                            costq_l2 = c(0.005),
  #                            lrate =  c(0.01))
  # 
  # # Constructing the parameter combinations to be evaluated
  # libmf_param_grid <- crossing(
  #   dim = libmf_param_values$dim,
  #   niter = libmf_param_values$niter,
  #   costp_l2 = libmf_param_values$costp_l2,
  #   costq_l2 = libmf_param_values$costq_l2,
  #   lrate = libmf_param_values$lrate
  # )
  # 
  # dim = c(10)
  # niter = c(30)
  # costp_l2 = c(0.01)
  # costq_l2 = c(0.005)
  # lrate =  c(0.01)
  # 
  # fold <- 1
  
  #--------------------------------------------------------------------------#
  #--------------------------------------------------------------------------#
  #--------------------------------------------------------------------------#
  
  # Ensure consistent types for joins and matrices
  ratings_train <- ratings_train %>%
    mutate(userId = as.character(userId),
           movieId = as.character(movieId))

  # Creating a user and item map with the IDs in a sequence which map to the original IDs
  user_levels <- sort(unique(ratings_train$userId))
  item_levels <- sort(unique(ratings_train$movieId))
  user_map <- setNames(seq_along(user_levels), user_levels)
  item_map <- setNames(seq_along(item_levels), item_levels)

  # Preparing the final results table
  libmf_cv_results <- vector("list", length = K_folds)

  # Looping over each fold
  for (fold in seq_len(K_folds)) {
    message(sprintf("[LIBMF] Starting fold %d/%d", fold, K_folds))

    # Splitting the ratings into train/test splits based on the assigned fold number
    ratings_train_tr <- ratings_train %>% filter(cv_fold != !!fold) %>% select(-cv_fold)
    ratings_train_te <- ratings_train %>% filter(cv_fold == !!fold) %>% select(-cv_fold)
    
    # Predicting movies for the users that appear in this fold's held-out set.
    test_users <- unique(ratings_train_te$userId)
    common_users <- intersect(test_users, unique(ratings_train_tr$userId))
    if (length(common_users) == 0) next
    all_items <- sort(unique(ratings_train$movieId))

    # Performing the parameter sweep
    fold_res <- pmap_dfr(libmf_param_grid, function(dim, niter, costp_l2, costq_l2, lrate) {
        message(sprintf("[LIBMF] fold=%d dim=%s niter=%s costp_l2=%s costq_l2=%s lrate=%s",
                        fold, dim, niter, costp_l2, costq_l2, lrate))

        # Creating the model's instance
        libmf_model <- Reco()

        # Storing the training data in memory
        train_data_reco <- data_memory(user_index = unname(user_map[ratings_train_tr$userId]),
                                       item_index = unname(item_map[ratings_train_tr$movieId]),
                                       rating = ratings_train_tr$rating,
                                       index1 = TRUE)
        
        # Training the LIBMF model with the current parameters
        libmf_model$train(train_data_reco, 
                          opts = list(dim = dim,
                                      niter = niter,
                                      costp_l2 = costp_l2,
                                      costq_l2 = costq_l2,
                                      lrate = lrate,
                                      verbose = FALSE))

        # Removing the training data movies from the movie candidates
        seen_long <- ratings_train_tr %>%
          transmute(userId = as.character(userId),
                    movieId = as.character(movieId)) %>%
          filter(userId %in% common_users)

        # Defining all possible user-item combinations for the recommendations
        candidate_pairs <- expand_grid(userId = common_users,
                                       movieId = all_items) %>%
          anti_join(seen_long, by = c("userId", "movieId"))

        # Storing the prediction data into memory
        pred_request <- data_memory(user_index = unname(user_map[candidate_pairs$userId]),
                                    item_index = unname(item_map[candidate_pairs$movieId]),
                                    index1 = TRUE)

        # Using the model to score the items for each user
        candidate_pairs$score_pred <- libmf_model$predict(pred_request, out_memory())

        # Preparing the true ratings to be joined with the predicted ratings
        true_long <- ratings_train_te %>%
          transmute(userId = as.character(userId),
                    movieId = as.character(movieId),
                    rating_true = rating) %>%
          filter(userId %in% common_users)

        # Joining the true ratings with the predicted ratings
        eval_long <- true_long %>%
          inner_join(candidate_pairs, by = c("userId", "movieId"))

        # Computing the RMSE, Precision/Recall@k of the remaining candidates
        metric_res <- compute_eval_metrics(eval_long = eval_long,
                                           candidate_pred = candidate_pairs,
                                           common_users = common_users,
                                           top_k = top_k,
                                           good_rating_threshold = good_rating_threshold)

        # Constructing the results entry with the evaluation results
        tibble(model = "LIBMF",
               params = list(
                 list(dim = dim,
                      niter = niter,
                      costp_l2 = costp_l2,
                      costq_l2 = costq_l2,
                      lrate = lrate)),
               fold = fold,
               rmse = metric_res$rmse,
               precision_at_k = metric_res$precision_at_k,
               recall_at_k = metric_res$recall_at_k)
      }
    )
    # Adding the results to the appropriate fold
    libmf_cv_results[[fold]] <- fold_res
  }
  # Joining the final results
  bind_rows(libmf_cv_results)
}

