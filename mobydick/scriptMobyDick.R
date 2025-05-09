library(tidyverse)
library(tokenizers)
library(ggplot2)

read_and_clean_text <- function(file_path) {
  lines <- readLines(file_path, warn = FALSE)
  text <- paste(lines[lines != ""], collapse = " ")
  text <- tolower(text)
  return(text)
}

tokenize_text <- function(text) {
  tokenize_words(text, lowercase = TRUE, strip_punct = FALSE)[[1]]
}

build_ngrams <- function(tokens, n = 2) {
  ngram_list <- map_chr(seq_len(length(tokens) - n + 1), function(i) {
    paste(tokens[i:(i + n - 1)], collapse = " ")
  })
  tibble(ngram = ngram_list) %>%
    count(ngram, sort = TRUE)
}

plot_ngrams <- function(ngram_df, top_n = 15, title = "Top n-grams") {
  ngram_df %>%
    slice_max(n, n = top_n) %>%
    mutate(ngram = reorder(ngram, n)) %>%
    ggplot(aes(x = ngram, y = n)) +
    geom_col(fill = "steelblue") +
    coord_flip() +
    labs(x = NULL, y = "Frequenza", title = title)
}

generate_sentence <- function(ngram_df, n = 2, max_words = 20) {
  ngram_split <- str_split(ngram_df$ngram, " ")
  if (n == 2) {
    bigrams <- tibble(w1 = sapply(ngram_split, `[`, 1),
                      w2 = sapply(ngram_split, `[`, 2),
                      freq = ngram_df$n)
    current <- sample(bigrams$w1, 1)
    sentence <- current
    for (i in 1:(max_words - 1)) {
      next_words <- bigrams %>% filter(w1 == current)
      if (nrow(next_words) == 0) break
      next_word <- sample(next_words$w2, 1, prob = next_words$freq)
      sentence <- paste(sentence, next_word, sep = " ")
      current <- next_word
    }
  } else {
    trigrams <- tibble(w1 = sapply(ngram_split, `[`, 1),
                       w2 = sapply(ngram_split, `[`, 2),
                       w3 = sapply(ngram_split, `[`, 3),
                       freq = ngram_df$n)
    current <- sample_n(trigrams, 1)
    sentence <- paste(current$w1, current$w2, current$w3)
    for (i in 1:(max_words - 3)) {
      next_candidates <- trigrams %>% filter(w1 == current$w2, w2 == current$w3)
      if (nrow(next_candidates) == 0) break
      current <- next_candidates %>% slice_sample(n = 1, weight_by = freq)
      sentence <- paste(sentence, current$w3, sep = " ")
    }
  }
  return(sentence)
}

file_path <- "moby-dick.txt"
text <- read_and_clean_text(file_path)
tokens <- tokenize_text(text)

bigrams_df <- build_ngrams(tokens, n = 2)
plot_ngrams(bigrams_df, title = "Top 15 Bigrammi")

trigrams_df <- build_ngrams(tokens, n = 3)
plot_ngrams(trigrams_df, title = "Top 15 Trigrammi")

cat("Frase (bigrammi):", generate_sentence(bigrams_df, n = 2), "\n\n")
cat("Frase (trigrammi):", generate_sentence(trigrams_df, n = 3), "\n")
