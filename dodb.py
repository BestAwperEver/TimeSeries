#import myelastic, mycouch, pars
#import myorient

#import couchdb, pars
import pyorient, datetime, json
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import matplotlib.pyplot as mplt
import matplotlib
#matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries: pd.DataFrame, plotgraph = False):

    #Determing rolling statistics
    #rolmean = pd.rolling_mean(timeseries, window=12)
	rolmean = timeseries.rolling(window=12,center=False).mean()
	rolstd = timeseries.rolling(window=12,center=False).std()
	#rolstd = pd.rolling_std(timeseries, window=12)

	if plotgraph:
		#Plot rolling statistics:
		orig = plt.plot(timeseries, color='blue',label='Original')
		mean = plt.plot(rolmean, color='red', label='Rolling Mean')
		std = plt.plot(rolstd, color='black', label = 'Rolling Std')
		plt.legend(loc='best')
		plt.title('Rolling Mean & Standard Deviation')
		plt.show(block=False)

	#Perform Dickey-Fuller test:
	print('Results of Dickey-Fuller Test:')
	dftest = adfuller(timeseries)
	dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
	for key,value in dftest[4].items():
		dfoutput['Critical Value (%s)'%key] = value
	print(dfoutput)

#client = pyorient.OrientDB('217.197.0.70', 2428)
#session_id = client.connect('root', 'yayva59')
#session_id = client.db_open('tsdb', 'root', 'yayva59')

client = pyorient.OrientDB('localhost', 2424)
session_id = client.connect('root', 'Trololo1')
#client.db_drop('quebec')
#client.db_create('tsdb', pyorient.DB_TYPE_GRAPH, pyorient.STORAGE_TYPE_PLOCAL)
client.db_open('tsdb', 'root', 'Trololo1')


#client.command("create class year extends V")
#client.command("create class month extends V")
#client.command("create class day extends V")
#client.command("create class data extends V")

#client.command("create property year.value string")
#client.command("create property year.month linkmap month")
#client.command("create property month.day linkmap day")
#client.command("create property day.data linkset data")
#client.command("create property data.date date")

#data = pars.readjson('mint.json')
#data = json.load(open('number-of-daily-births-in-quebec.json', 'r', encoding='utf8'))

#count = 0
#for d in data:
#	dt = datetime.datetime.strptime(d['date'], "%Y-%m-%d")
#	dy, dm, dd = dt.year, dt.month, dt.day
#	res = client.command("insert into data set value="+str(d['data'])+", date='"+str(d['date'])+"' return @rid")
#	res = client.command("insert into day(data) values(["+str(res[0])[1:]+"]) return @rid")
#	res = client.command("insert into month(day) values({'"+str(dd)+"': "+str(res[0])[1:]+"}) return @rid")
#	res = client.command("insert into year(value, month) values('"+str(dy)+"', {'"+str(dm)+"': "+str(res[0])[1:]+"}) return @rid")
#	count += 1
#	print(str(count)+' из '+str(len(data)))


#code = (
#	"var l = new Date(lint);"
#	"var r = new Date(rint);"
#	"var res = [];var n;"
#	"if (l>r) return;"
#	"while (+l!=+r){"
#	"var lg = l.getFullYear();"
#	"var lm = l.getMonth()+1;"
#	"var ld = l.getDate();"
#	'var sq = "select expand(month["+lm+"].day["+ld+"].data) from year where value=\\\'"+lg+"\\\'";'
#	"var re = db.query(sq);"
#	"if (re.length != 0){n = Number(JSON.parse(re[0].getRecord().toJSON()).value);}"
#	"else{n = res[res.length-1];}"
#	"l.setDate(l.getDate()+1);"
#	"res.push(n);}"
#	"return res;")
#client.command("create function getFromInt '"+code+"' parameters [lint, rint] idempotent true language JavaScript")

#res = client.query("SELECT EXPAND(month[8].day[5].mint) FROM Year WHERE value='1988'")[0].oRecordData['data']
#res = client.query("select getFromInt('1983-02-04', '1983-03-08')")[0].gfi()
query_res = client.query("select * from mint where date > '1980-01-01' and date < '1990-01-01' order by date limit 10000")
#query_res = client.query("select * from uah where date > '1999-01-01' and date < '2005-01-01' order by date limit 10000")
res = [(pd.Timestamp( query_res[i].oRecordData['date'] ), query_res[i].oRecordData['value']) for i in range(len(query_res))]

#data = pd.DataFrame(res, columns = ("Date", "Births"))
data = pd.Series([query_res[i].oRecordData['value'] for i in range(len(query_res))], [pd.to_datetime(query_res[i].oRecordData['date']) for i in range(len(query_res))])
#data.columns = ["Births"]
#print(data)
#print('\n')
print(data.head())
print('\n Data Types:')
print(data.dtypes)
print(data.index)

#asd = data['2001'] + data['2002']
#dsa = data['2001':'2002']
#asdf = data['2006']

#print(data['2005':'2006'])
#data.plot()

#test_stationarity(data['1982':'1988'])

#dates = matplotlib.dates.date2num([x[0] for x in res])

#plt.plot(data)
#plt.show()
#mplt.plot_date(dates, [x[1] for x in res])
#mplt.show()

ts_log = data
ts_log_diff = ts_log - ts_log.shift()
#mplt.plot(ts_log)
moving_avg = pd.rolling_mean(ts_log,12)
#mplt.plot(moving_avg, color='red')

ts_log_moving_avg_diff = ts_log - moving_avg

ts_log_moving_avg_diff.dropna(inplace=True)
test_stationarity(ts_log_moving_avg_diff)

#expwighted_avg = pd.ewma(ts_log, halflife=12)
#plt.plot(ts_log)
#plt.plot(expwighted_avg, color='red')

#ts_log_ewma_diff = ts_log - expwighted_avg
#test_stationarity(ts_log_ewma_diff)

#l = [[pd.Timestamp(query_res[i].oRecordData['date']) for i in range(len(query_res))], [query_res[i].oRecordData['value'] for i in range(len(query_res))]]
#l = list(map(list, zip(*l)))

#df = pd.DataFrame(l, columns = ['Date', 'Value'])
#df.reset_index(inplace=True)
#df['Date'] = pd.to_datetime(df['Date'])
#df = df.set_index('Date')

#df = pd.DataFrame(res, columns = ('Date', 'Values'))
#df.reset_index(inplace=True)
#df['Date'] = pd.to_datetime(df['Date'])
#df = df.set_index('Date')

df = data
#df.asfreq('D', method='pad')

#freq1 = pd.infer_freq(df.index)
#print(freq1)

decomposition = seasonal_decompose(df, freq = 365)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

#plt.subplot(411)
#plt.plot(ts_log, label='Original')
#plt.legend(loc='best')
#plt.subplot(412)
#plt.plot(trend, label='Trend')
#plt.legend(loc='best')
#plt.subplot(413)
#plt.plot(seasonal,label='Seasonality')
#plt.legend(loc='best')
#plt.subplot(414)
#plt.plot(residual, label='Residuals')
#plt.legend(loc='best')
#plt.tight_layout()

#ACF and PACF plots:
from statsmodels.tsa.stattools import acf, pacf

ts_acf = ts_log

ts_acf.dropna(inplace=True)
lag_acf = acf(ts_acf, nlags=30)
lag_pacf = pacf(ts_acf, nlags=20, method='ols')

##Plot ACF: 20
#plt.subplot(121) 
#plt.plot(lag_acf)
#plt.axhline(y=0,linestyle='--',color='gray')
#plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
#plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
#plt.title('Autocorrelation Function')

##Plot PACF: 6
#plt.subplot(122)
#plt.plot(lag_pacf)
#plt.axhline(y=0,linestyle='--',color='gray')
#plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
#plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
#plt.title('Partial Autocorrelation Function')
#plt.tight_layout()

from statsmodels.tsa.arima_model import ARIMA

ts_log.dropna()
model = ARIMA(ts_log, order=(3, 0, 6))  
results_ARIMA = model.fit(disp=-1)  
plt.plot(ts_log)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts_log)**2))

#predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
##print(predictions_ARIMA_diff.head())

#predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
##print(predictions_ARIMA_diff_cumsum.head())

#predictions_ARIMA_log = pd.Series(ts_log.ix[0], index=ts_log.index)
#predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)

#predictions_ARIMA = predictions_ARIMA_log
#plt.plot(data)
#plt.plot(predictions_ARIMA)
#plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-data)**2)/len(data)))

plt.show()
















#AO = pd.Series([data[i][1] for i in range(20)], index = [data[i][0] for i in range(20)])
#AO.plot()

#'''
#create class year extends V
#create class month extends V
#create class day extends V
#create class uah extends V

#create property year.value string
#create property year.month linkmap month
#create property month.day linkmap day
#create property day.uah linkmap uah



#CREATE VERTEX uah SET price=34.2684

#INSERT INTO Day(uah) VALUES ([#16:0])
#INSERT INTO Month(day) VALUES ({'4':#14:0})
#INSERT INTO Year(value,month) VALUES ('2012',{'3':#13:0})
#'''







#dbname = 'iot'

#couch = couchdb.Server()

#if dbname in couch:
#	db = couch[dbname]
#else:
#	db = couch.create(dbname)




#couch.delete('pytest')
#db = couch.create('rus_ukr') # newly created
#db = couch['mydb'] # existing



#geomas = pars.readjson('rus_ukr.json')

#for geo in geomas:
#	db.save(geo)

#client = myorient.connect("localhost", 2424, "root", "lev_gridnev")
#myorient.creategraph(client, "mypytest", "goroda.json", "dorogi.json")
#client.db_open("mypytest", "root", "lev_gridnev")

#put = myorient.searchput(client, "F", "G")
#print(put)


'''
es = myelastic.connect('user', 'bitnami', '192.168.0.106', '80')

#myelastic.creategeo(es, 'rus_ukr.json')

brz = "59.4030726, 56.8182876"
spb = "59.8740953, 29.8305951"


myelastic.searchgeo(es, "400km", spb, True)
#print()
#myelastic.sortgeo(es, brz, True)
#myelastic.testsearch(es, True)
'''