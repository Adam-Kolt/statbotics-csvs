team_data = read.csv("/Users/adamkoltuniuk/Documents/GitHub/statbotics-csvs/v3/team_events.csv")

team_data$norm_epa=as.factor(team_data$norm_epa)
sapply(team_data, class)

summary(team_data)


recent_teams = team_data[team_data$year >= 2016,]

hist(recent_teams$year)
