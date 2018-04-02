landslides['date_parsed'] = pd.to_datetime(landslides['date'], format = "%m/%d/%y")
day_of_month_landslides = landslides['date'].dt.day
