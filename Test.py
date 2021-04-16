import pandas as pd
start_date = '3/12/20'
end_date = '3/16/20'
file = pd.read_csv("US_data.csv")
US_data = pd.DataFrame(file, columns = ['Date', 'Open Price','High Price', 'Low Price', 'Last Price', 'VWAP', 'Volume'])
start = pd.to_datetime(start_date)
end = pd.to_datetime(end_date)
US_data.Date = pd.to_datetime(US_data.Date)
US_data = US_data[US_data.Date >= start_date]
US_data = US_data[US_data.Date <= end_date]

print(US_data)