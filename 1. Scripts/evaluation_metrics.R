compute_eval_metrics <- function(eval_long,
                                 candidate_pred,
                                 common_users,
                                 top_k,
                                 good_rating_threshold) {
  
  # Calculating RMSE of the model ratings against the actual ratings of the test set
  if (nrow(eval_long) == 0) {
    rmse_val <- NA_real_
  } else {
    rmse_val <- sqrt(mean((eval_long$rating_true - eval_long$score_pred)^2, na.rm = TRUE))
  }

  # Getting the top K items with the highest predicted rating
  top_k_pred <- candidate_pred %>%
    group_by(userId) %>%
    arrange(desc(score_pred), .by_group = TRUE) %>%
    slice_head(n = top_k) %>%
    ungroup()

  # Ground-truth relevant items in the evaluation set (rating above threshold)
  relevant_true <- eval_long %>%
    filter(rating_true > good_rating_threshold) %>%
    select(userId, movieId)
  
  # Identifying how many of the top K recommended items were correct recommendations
  hits <- top_k_pred %>%
    semi_join(relevant_true, by = c("userId", "movieId")) %>%
    count(userId, name = "hits")

  # Calculating the number of relevant items per user
  n_relevant <- relevant_true %>%
    count(userId, name = "n_relevant")

  # Calculating Precision/Recall@K
  user_eval <- tibble(userId = common_users) %>%
    left_join(hits, by = "userId") %>%
    left_join(n_relevant, by = "userId") %>%
    mutate(hits = replace_na(hits, 0L),
           n_relevant = replace_na(n_relevant, 0L),
           precision_k = hits / top_k,
           recall_k = if_else(n_relevant > 0, hits / n_relevant, NA_real_))

  precision_val <- if (all(is.na(user_eval$precision_k))) NA_real_ else mean(user_eval$precision_k, na.rm = TRUE)
  recall_val <- if (all(is.na(user_eval$recall_k))) NA_real_ else mean(user_eval$recall_k, na.rm = TRUE)

  # Creating a table with the RMSE and Prediction/Recall@K results
  tibble(rmse = rmse_val,
         precision_at_k = precision_val,
         recall_at_k = recall_val)
}

