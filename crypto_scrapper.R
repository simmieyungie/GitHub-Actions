library(ralger)
library(dplyr)
library(magrittr)

#scrape table from page
data <- ralger::table_scrap("https://goldprice.org/cryptocurrency-price") %>%
  select(-8) %>% #drop column
  mutate(DateTime = Sys.time()) #add time of scraping

#Write data to file, new data scraped after automation will be appended
write.table(data, "data/data.csv",
            sep = ",", col.names = !file.exists("data/data.csv"),
            append = T, row.names = F)

