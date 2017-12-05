from riskpy.modeling import binning
from riskpy.graphs import graphs
import pandas as pd

data=pd.read_csv('C:\data\cs-training.csv')
bs=binning.BinningSettings('NumberOfTime30-59DaysPastDueNotWorse',False)
b=binning.Binner()
b.fit(data,'SeriousDlqin2yrs',binning_settings=[bs])


binning.to_file(b,'saving_test.123')

b2=binning.read_file('saving_test.123')


#woe_data=b2.transform(data)
b2.to_sql()

print(b2.to_sql())