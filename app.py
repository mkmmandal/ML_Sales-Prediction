#This app is for sales prediction
#we predict the sales after user select date, store no. and item 
from flask import Flask ,render_template,request
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pickle
import pandas
import sklearn
app = Flask(__name__)

#This is our model after fitting train data
xgb_model=pickle.load(open('xgb_model_final.pkl','rb'))
#This is a sample of test data, combine this with the user input value and we will generate the predictions
salesDf=pickle.load(open('salesappDf.pkl','rb'))

#Using column transformer for onehotencoding
transformer=ColumnTransformer(transformers=[('tnf1'
                ,OneHotEncoder(sparse=False,drop='first')
                ,['item','store_nbr','City','category'])],remainder='passthrough')


@app.route("/",methods=['GET','POST'])
def home_page():
   prediction=-1
   cdate=''
   cstore=''
   citem=''
   #Get user input value for date, store no. and item type
   if request.method=='POST':
       cdate =request.form['Date']
       cstore=request.form['Store']
       citem=request.form['Item']
       #converting date from string to pandas datetime
       pdate=pandas.to_datetime(cdate)
       #Based on store no. fetch its city and category from salesappDf
       dfDetails=salesDf.iloc[salesDf[salesDf.store_nbr==cstore].index]
       indx=salesDf[salesDf.store_nbr==cstore].index
       #fetched city and category
       scity=dfDetails.City[indx[0]]
       scat=dfDetails.category[indx[0]]
       #make dataframe out of chosen value and concat with salesappDf
       chosenDf=pandas.DataFrame([[citem,cstore,pdate.day,pdate.month,pdate.year,pdate.dayofweek,scity,scat]],columns=salesDf.columns)
       finalDf=pandas.concat([salesDf,chosenDf])
       #Sorting the columns
       finalDf=finalDf[['item','store_nbr','year','month','day','dayofweek','City','category']]
       #Manually setting the oil price
       oil_price=50
       if pdate.year==2018:
         oil_price=55
       elif pdate.year>2018:
         oil_price=60
       else:
         oil_price=50
       oil_list=[]
       #making list for oil price setting value 0 for all other rows in dataframe bcz we need prediction value only for last one
       #which is our chosen df row
       for i in range(0,54):
         oil_list.append(0)
       oil_list.append(oil_price)
       #we have our final dataframe here with oil price 
       finalDf['oil_price']=oil_list
       #One hot encoding
       tFinalDf=transformer.fit_transform(finalDf)
       #predict the value
       predictions=xgb_model.predict(tFinalDf)
       #if prediction value is less then 0; show 0 to the user
       if predictions[-1]<0:
         prediction=0
       else:
         prediction=predictions[-1]
   return render_template('index.html',prediction=prediction,cdate=cdate,cstore=cstore,citem=citem)


if __name__=="__main__":
    app.run(debug=True)


