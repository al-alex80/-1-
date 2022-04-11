import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression

def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if str(col_type)[:5] == "float":
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min > np.finfo("f2").min and c_max < np.finfo("f2").max:
                df[col] = df[col].astype(np.float16)
            elif c_min > np.finfo("f4").min and c_max < np.finfo("f4").max:
                df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col].astype(np.float64)
        elif str(col_type)[:3] == "int":
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min > np.iinfo("i1").min and c_max < np.iinfo("i1").max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo("i2").min and c_max < np.iinfo("i2").max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo("i4").min and c_max < np.iinfo("i4").max:
                df[col] = df[col].astype(np.int32)
            elif c_min > np.iinfo("i8").min and c_max < np.iinfo("i8").max:
                df[col] = df[col].astype(np.int64)
        elif col == "timestamp":
            df[col] = pd.to_datetime(df[col])
        elif str(col_type)[:8] != "datetime":
            df[col] = df[col].astype("category")
    end_mem = df.memory_usage().sum() / 1024**2
    print('Потребление памяти меньше на', round(start_mem - end_mem, 2), 'Мб (минус', round(100 * (start_mem - end_mem) / start_mem, 1), '%)')
    return df

buildings = pd.read_csv("http://video.ittensive.com/machine-learning/ashrae/building_metadata.csv.gz")
weather = pd.read_csv("http://video.ittensive.com/machine-learning/ashrae/weather_train.csv.gz")
weather = weather[weather["site_id"] == 0]
energy = pd.read_csv("http://video.ittensive.com/machine-learning/ashrae/train.0.csv.gz")
energy = energy[energy["building_id"]<50]
energy = pd.merge(left=energy, right=buildings, how="left",
                   left_on="building_id", right_on="building_id")

def weather_interpolate_diff (weather):
    interpolate_columns = ["air_temperature", "dew_temperature",
                       "cloud_coverage", "wind_speed",
                       "sea_level_pressure"]
    for col in interpolate_columns:
        weather[col] = weather[col].interpolate(limit_direction='both',
                            kind='cubic')
    weather["air_temperature_diff1"] = weather["air_temperature"].diff()
    weather.at[0, "air_temperature_diff1"] = weather.at[1, "air_temperature_diff1"]
    weather["air_temperature_diff2"] = weather["air_temperature_diff1"].diff()
    weather.at[0, "air_temperature_diff2"] = weather.at[1, "air_temperature_diff2"]
    return weather

weather = weather_interpolate_diff(weather)

energy = energy.set_index(["timestamp", "site_id"])
weather = weather.set_index(["timestamp", "site_id"])
energy = pd.merge(left=energy, right=weather, how="left",
                  left_index=True, right_index=True)
energy.reset_index(inplace=True)
energy = energy.drop(columns=["meter", "year_built",
                              "square_feet", "floor_count"], axis=1)
del weather
energy = reduce_mem_usage(energy)

def add_dates_log (df):
    df["hour"] = df["timestamp"].dt.hour.astype("int8")
    df["weekday"] = df["timestamp"].dt.weekday.astype("int8")
    df["week"] = df["timestamp"].dt.week.astype("int8")
    df["month"] = df["timestamp"].dt.month.astype("int8")
    df["date"] = pd.to_datetime(df["timestamp"].dt.date)
    for weekday in range(0,7):
        df['is_wday' + str(weekday)] = df['weekday'].isin([weekday]).astype("int8")
    for week in range(1,54):
        df['is_w' + str(week)] = df['week'].isin([week]).astype("int8")
    for month in range(1,13):
        df['is_m' + str(month)] = df['month'].isin([month]).astype("int8")
    dates_range = pd.date_range(start='2015-12-31', end='2018-06-01')
    us_holidays = calendar().holidays(start=dates_range.min(),
                                  end=dates_range.max())
    df['is_holiday'] = df['date'].isin(us_holidays).astype("int8")
    if "meter_reading" in df.columns:
        df["meter_reading_log"] = np.log(df["meter_reading"] + 1)
    return df

energy = add_dates_log(energy)

hours = range(0, 24)
buildings = range(0, energy["building_id"].max() + 1)

def calculate_model_coeffs (columns):
    energy_train_lr = pd.DataFrame(energy, columns=columns)
    coeffs = [[]]*len(buildings)
    for building in buildings:
        coeffs[building] = [[]]*len(hours)
        energy_train_b = energy_train_lr[energy_train_lr["building_id"]==building]
        for hour in hours:
            energy_train_bh = energy_train_b[energy_train_b["hour"]==hour]
            y = energy_train_bh["meter_reading_log"]
            x = energy_train_bh.drop(labels=["meter_reading_log",
                "hour", "building_id"], axis=1)
            model = LinearRegression(fit_intercept=False).fit(x, y)
            coeffs[building][hour] = model.coef_
            coeffs[building][hour] = np.append(coeffs[building][hour], model.intercept_)
    return coeffs

lr_columns_weather = ["meter_reading_log", "hour", "building_id",
             "air_temperature", "dew_temperature",
             "sea_level_pressure", "wind_speed", "cloud_coverage",
             "air_temperature_diff1", "air_temperature_diff2"]
energy_lr_w = calculate_model_coeffs(lr_columns_weather)

lr_columns_days = ["meter_reading_log", "hour", "building_id",
                   "is_holiday"]
for wday in range(0,7):
    lr_columns_days.append("is_wday" + str(wday))
energy_lr_d = calculate_model_coeffs(lr_columns_days)

lr_columns_weeks = ["meter_reading_log", "hour", "building_id"]
for week in range(1,54):
    lr_columns_weeks.append("is_w" + str(week))
energy_lr_ww = calculate_model_coeffs(lr_columns_weeks)

lr_columns_monthes = ["meter_reading_log", "hour", "building_id"]
for month in range(1,13):
    lr_columns_monthes.append("is_m" + str(month))
energy_lr_m = calculate_model_coeffs(lr_columns_monthes)

del energy

buildings = pd.read_csv("http://video.ittensive.com/machine-learning/ashrae/building_metadata.csv.gz")
weather = pd.read_csv("http://video.ittensive.com/machine-learning/ashrae/weather_test.csv.gz")
weather = weather[weather["site_id"] == 0]
results = pd.read_csv("http://video.ittensive.com/machine-learning/ashrae/test.csv.gz")
results = results[(results["building_id"] < 50) & (results["meter"] == 0)]
results = pd.merge(left=results, right=buildings, how="left",
                   left_on="building_id", right_on="building_id")
del buildings

weather = weather_interpolate_diff(weather)

results = results.set_index(["timestamp", "site_id"])
weather = weather.set_index(["timestamp", "site_id"])
results = pd.merge(left=results, right=weather, how="left",
                  left_index=True, right_index=True)
results.reset_index(inplace=True)
results = results.drop(columns=["meter", "site_id", "year_built",
                              "square_feet", "floor_count"], axis=1)
del weather
results = reduce_mem_usage(results)

results = add_dates_log(results)

def calculate_model (x, model, columns):
    return (np.sum([x[col] * model[i] for i,col in enumerate(columns[3:])])
            + model[len(columns)-3])

def calculate_ensemble (x):
    lr = -1
    lr_w = calculate_model(x, 
            energy_lr_w[x.building_id][x.hour], lr_columns_weather)
    lr_d = calculate_model(x, 
            energy_lr_d[x.building_id][x.hour], lr_columns_days)
    lr_ww = calculate_model(x, 
            energy_lr_ww[x.building_id][x.hour], lr_columns_weeks)
    lr_m = calculate_model(x, 
            energy_lr_m[x.building_id][x.hour], lr_columns_monthes)
    lr = np.exp((lr_w*3 + lr_d*3 + lr_ww + lr_m)/8)
    if lr < 0 or lr != lr or lr*lr == lr:
        lr = 0
    x["meter_reading"] = lr
    return x

results = results.apply(calculate_ensemble, axis=1, result_type="expand")

results_ready = pd.DataFrame(results, columns=["row_id", "meter_reading"])
print (results_ready.info())
print (len(results_ready[results_ready["meter_reading"]>0]))

results = pd.read_csv("http://video.ittensive.com/machine-learning/ashrae/test.csv.gz",
                     usecols=["row_id"])
results = pd.merge(left=results, right=results_ready, how="left",
                  left_on="row_id", right_on="row_id")
results.fillna(value=0, inplace=True)
print (results.info())

results.to_csv("submission.csv",index=False)

del results
del results_ready
