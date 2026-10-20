library(ggplot2)

set.seed(42)

# Parameters
n       <- 120
mu_pop  <- 100
VA      <- 36
VE      <- 36
VP      <- VA + VE
h2      <- VA / VP        # 0.5
sigma_e <- 4

m_vals  <- c(0, 0.2, 0.4, 0.6, 0.8, 1.0)

# Simulate one dataset per m
sim_data <- function(m) {
  G_mum <- rnorm(n, mu_pop, sqrt(VA))
  E_mum <- rnorm(n, 0,      sqrt(VE))
  Z_mum <- G_mum + E_mum
  G_dad <- rnorm(n, mu_pop, sqrt(VA))
  G_off <- 0.5 * (G_mum + G_dad) + rnorm(n, 0, sqrt(VA / 2))
  Z_off <- G_off + m * E_mum + rnorm(n, 0, sigma_e)
  r     <- round(cor(Z_mum, Z_off), 3)
  data.frame(Z_mum = Z_mum, Z_off = Z_off, m = m, r = r)
}

df_all <- do.call(rbind, lapply(m_vals, sim_data))

df_all$m_label <- factor(
  sprintf("m = %.1f", df_all$m),
  levels = sprintf("m = %.1f", m_vals)
)

# Per-panel slope + r annotations
ann <- do.call(rbind, lapply(m_vals, function(m) {
  sub      <- df_all[df_all$m == m, ]
  lm_fit   <- lm(Z_off ~ Z_mum, data = sub)
  data.frame(
    m_label   = factor(sprintf("m = %.1f", m),
                       levels = sprintf("m = %.1f", m_vals)),
    slope_obs = round(coef(lm_fit)[2], 3),
    r         = round(unique(sub$r), 3)
  )
}))

# Colour ramp grey -> teal
pal <- c("#AAAAAA","#7EB8C9","#4EAAB5","#2E9E8E","#1D8C6B","#0F6E4E")

p <- ggplot(df_all, aes(x = Z_mum, y = Z_off)) +
  
  geom_point(aes(colour = m_label), alpha = 0.35, size = 1.4, shape = 16) +
  
  geom_smooth(method = "lm", se = TRUE, colour = "black",
              linewidth = 0.8, fill = "grey85", alpha = 0.5) +
  
  # 1:1 perfect resemblance reference
  geom_abline(slope = 1, intercept = 0,
              linetype = "dashed", colour = "#CC3333", linewidth = 0.55) +
  
  # Slope + r text inside each panel
  geom_text(data = ann,
            aes(x = -Inf, y = Inf,
                label = paste0("slope = ", slope_obs, "\nr = ", r)),
            hjust = -0.12, vjust = 1.3,
            size = 3.4, colour = "black", inherit.aes = FALSE) +
  
  facet_wrap(~ m_label, ncol = 3) +
  
  scale_colour_manual(
    values = setNames(pal, sprintf("m = %.1f", m_vals))
  ) +
  
  labs(
    title    = "Maternal effects and parent-offspring resemblance",
    subtitle = expression(Z[off] == G[off] + italic(m) %.% E[mother] + epsilon),
    x        = "Mother phenotype (Z_mother)",
    y        = "Offspring phenotype (Z_off)",
    caption  = "Red dashed = perfect resemblance (slope 1)  |  Black = OLS fit  |  h2 = 0.5,  n = 120 per panel",
    colour   = NULL
  ) +
  
  theme_classic(base_size = 12) +
  theme(
    plot.title       = element_text(face = "bold", size = 15),
    plot.subtitle    = element_text(size = 11, colour = "#444444",
                                    margin = margin(b = 8)),
    plot.caption     = element_text(size = 8.5, colour = "#777777"),
    strip.text       = element_text(face = "bold", size = 13),
    strip.background = element_rect(fill = "#F4F4F4", colour = NA),
    legend.position  = "none",
    panel.grid.major = element_line(colour = "#EEEEEE"),
    axis.title       = element_text(size = 10.5),
    plot.margin      = margin(12, 16, 12, 12)
  )

ggsave("/home/usriniva/Downloads/maternal_effects.png",
       plot = p, width = 13, height = 9, dpi = 150)
cat("Done.\n")