import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

if __name__ == "__main__":
    df = pd.read_csv('../datasets/SFC_Trainset.csv')

    df = df.drop('Resolution', axis=1)

    assert df.Dates.isnull().any() == False
    assert df.Dates.str.match('\d\d\d\d-\d\d-\d\d \d\d:\d\d:\d\d').all() == True

    df['Date'] = pd.to_datetime(df.Dates)
    df = df.drop('Dates', axis=1)

    df['IsDay'] = 0
    df.loc[(df.Date.dt.hour > 6) & (df.Date.dt.hour < 20), 'IsDay'] = 1

    days_to_int_dic = {
        'Monday': 1,
        'Tuesday': 2,
        'Wednesday': 3,
        'Thursday': 4,
        'Friday': 5,
        'Saturday': 6,
        'Sunday': 7,
    }
    df['DayOfWeek'] = df['DayOfWeek'].map(days_to_int_dic)
    df.DayOfWeek.unique()

    df['Hour'] = df.Date.dt.hour
    df['Month'] = df.Date.dt.month
    df['Year'] = df.Date.dt.year
    df['Year'] = df['Year'] - 2000  # The Algorithm doesn't know the difference. It's just easier to work like that

    df['HourCos'] = np.cos((df['Hour'] * 2 * np.pi) / 24)
    df['DayOfWeekCos'] = np.cos((df['DayOfWeek'] * 2 * np.pi) / 7)
    df['MonthCos'] = np.cos((df['Month'] * 2 * np.pi) / 12)

    df = pd.get_dummies(df, columns=['PdDistrict'])

    cat_le = LabelEncoder()
    df['CategoryInt'] = pd.Series(cat_le.fit_transform(df.Category))

    df['InIntersection'] = 1
    df.loc[df.Address.str.contains('Block'), 'InIntersection'] = 0

    cols = ['X', 'Y', 'IsDay', 'DayOfWeek', 'Month', 'Hour', 'Year', 'InIntersection',
            'PdDistrict_BAYVIEW', 'PdDistrict_CENTRAL', 'PdDistrict_INGLESIDE',
            'PdDistrict_MISSION', 'PdDistrict_NORTHERN', 'PdDistrict_PARK',
            'PdDistrict_RICHMOND', 'PdDistrict_SOUTHERN', 'PdDistrict_TARAVAL',
            'PdDistrict_TENDERLOIN', 'CategoryInt']

    df_edited = df[cols]

    df_edited.to_csv('../datasets/San_Francisco_Crime.csv', index=False, header=None)

    print("OK")
