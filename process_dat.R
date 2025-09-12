library(dplyr)
library(MASS)
library(ks)
library(ggplot2)
library(RColorBrewer)
library(ggnewscale)
library(cowplot)
library(HDInterval)
library(zoo)
library(stringr)
library(tidyr)
setwd("C:/Users/samil/Documents/penalty_function")
df = read.csv("multiple_iterations.csv")
df_clean <- df %>%
  filter(!if_any(where(is.numeric), ~ is.infinite(.) | is.nan(.)))


df_long <- df %>%
  pivot_longer(
    cols = c(col1, col2, col3),   # the columns you want to gather
    names_to = "variable",        # new column name for former column names
    values_to = "value"           # new column name for values
  )

# Step 1: Convert from wide to long format
df_long <- df_clean %>%
  pivot_longer(cols = starts_with("iter_"), names_to = "iteration", values_to = "value")


plot = ggplot(data=df_clean)+
  geom_line(aes(x=x,y=iter_0,color='No Update'),alpha=1,linewidth=2)+
  geom_line(aes(x=x,y=iter_1,color='First Update') ,alpha=1,linewidth=2)+
geom_line(aes(x=x,y=iter_2,color='Second Update'),alpha=1,linewidth=2)+
  scale_color_manual(values = c("No Update"="#66C2A5",'First Update'="#FC8D62",'Second Update'="#8DA0CB"))+
  theme_minimal(base_size = 14) +
  theme(
    panel.background = element_rect(fill = "snow", color = NA),
    panel.grid.major = element_line(color = "grey80"),
    panel.grid.minor = element_blank(),
    axis.title.x = element_text(face = "italic"),
    axis.title.y = element_text(face = "italic"),
    plot.title = element_text(face = "bold")
  ) +
  scale_x_continuous(breaks = seq(-2, 2, by = 0.4)) +
  scale_y_continuous(breaks = seq(-2, 2, by = 0.2)) +
  labs(
    title = "Univariate Exponential Penalty Application",
    x = "x",
    y = "y",
    color="Update Iterate"
  )
ggsave('univariate_update.png',plot,width=15 ,height =12 )
