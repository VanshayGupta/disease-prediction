import numpy as np
from flask import Flask, request, render_template, jsonify
import pickle
import tabula
import pandas as pd
from firebase_admin import credentials, firestore, initialize_app, storage
from datetime import date
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Initialize Firestore DB
cred = credentials.Certificate('key.json')
default_app = initialize_app(cred, {'storageBucket': 'insuranceportal-e9729.appspot.com'})
db = firestore.client()
user_ref = db.collection('users')

bucket = storage.bucket(app=default_app)

@app.route('/', methods=['GET'])
def home():
    user_data=read()
    print(user_data)
    gender = user_data["personalData"]["personalData"]["gender"]
    if(gender=="M"):
        sex=1
    else:
        sex=0
    age=AgeCalc(user_data["personalData"]["personalData"]["birthDate"])
    personal_data = {'Age':[age], 'Gender':[sex]}
    age_df= pd.DataFrame({'Age':[age]}, index=['Value']) 
    df_personal_data = pd.DataFrame(personal_data, index=['Value']) 
    report_id = user_data["reportId"]
    test_results = createDf(report_id) 
    df_heart=test_results[['cp', 'trestbps', 'chol', 'fbs',	'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']]
    df_liver=test_results[['TB','DB','Alkphos','Sgpt', 'Sgot','TP',	'ALB', 'A/G Ratio']]
    df_diabetes=test_results[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction']]
    df_heart=pd.concat([df_personal_data,df_heart], axis=1)
    df_liver=pd.concat([df_personal_data,df_liver], axis=1)
    df_diabetes=pd.concat([df_diabetes,age_df], axis=1)
    heart_proba = HeartCalculation(df_heart)
    liver_proba = LiverCalculation(df_liver)
    diabetes_proba = DiabetesCalculation(df_diabetes)
    print(heart_proba, liver_proba, diabetes_proba)
    MedRiskFactor = CalcRisk(heart_proba, liver_proba, diabetes_proba)
    print(MedRiskFactor)
    if(MedRiskFactor<0.458):
        riskLevel="low 8"
    elif(MedRiskFactor<0.518):
        riskLevel="medium 10"
    elif(MedRiskFactor<0.596):
        riskLevel="high 15"
    elif(MedRiskFactor>0.596):
        riskLevel="critical ineligible"
    print(riskLevel)
    return jsonify(str(MedRiskFactor))

#reading data from Firebase
def read():
    try:
        user_id = request.args.get('id')    
        if user_id:
            user = user_ref.document(user_id)
            data=(user.get().to_dict())
            return data
        else:
            info = [doc.to_dict() for doc in user_ref.stream()]
            return jsonify(info), 200
    except Exception as e:
        return f"An Error Occured: {e}"

#calculating age from date of birth
def AgeCalc(birthdate):
    bday=birthdate.split('-')
    year=int(bday[0])
    month=int(bday[1])
    day=int(bday[2])
    today = date.today()
    age = today.year - year -((today.month, today.day) < (month, day))
    return age

#Medical Reports OCR using tabula-py
def createDf(report_id):
    blob = bucket.blob("Reports/"+report_id)
    blob.download_to_filename(filename= "medical_report.pdf")
    df_pdf = tabula.read_pdf("med_report.pdf", pages="all")
    df_list=[]
    for i in range(len(df_pdf)):
        df_list.append(df_pdf[i])
    large_df=(pd.concat(df_list))
    large_df=large_df.dropna()
    test_results=large_df[["Attribute", "Value"]]
    test_results=test_results.transpose()
    new_header = test_results.iloc[0]
    test_results = test_results[1:] 
    test_results.columns = new_header
    print(test_results) 
    return test_results

#calculation functions
def HeartCalculation(df_heart):
    df_heart.rename(columns = {'Age':'age', 'Gender':'sex'}, inplace = True)
    df_heart["cp"] = df_heart["cp"].apply(np.int64)
    df_heart["ca"] = df_heart["ca"].apply(np.int64)
    df_heart["thal"] = df_heart["thal"].apply(np.int64)
    df_heart["slope"] = df_heart["slope"].apply(np.int64)
    df_heart["restecg"] = df_heart["restecg"].apply(np.int64)
    df_heart["exang"] = df_heart["exang"].apply(np.int64)
    df_heart["fbs"] = df_heart["fbs"].apply(np.int64)
    df_heart["trestbps"] = df_heart["trestbps"].apply(np.int64)
    df_heart["chol"] = df_heart["chol"].apply(np.int64)
    df_heart["thalach"] = df_heart["thalach"].apply(np.int64)
    print(df_heart)
    df_= PreprocessData_Heart(df_heart)
    print(df_)
    result = ValuePredictor_Heart(df_.values)       
    return result[1]

def LiverCalculation(df_liver):
    df_liver["Alkphos"] = df_liver["Alkphos"].apply(np.int64)
    df_liver.rename(columns = {'A/G Ratio':'A/G',}, inplace = True)
    print(df_liver)
    df_ = PreprocessData_Liver(df_liver)
    print(df_)
    result = ValuePredictor_Liver(df_.values) 
    return result[1]

def DiabetesCalculation(df_diabetes):
    df_diabetes["Pregnancies"] = df_diabetes["Pregnancies"].apply(np.int64)
    df_diabetes["Glucose"] = df_diabetes["Glucose"].apply(np.int64)
    df_diabetes["BloodPressure"] = df_diabetes["BloodPressure"].apply(np.int64)
    df_diabetes["SkinThickness"] = df_diabetes["SkinThickness"].apply(np.int64)
    df_diabetes["Insulin"] = df_diabetes["Insulin"].apply(np.int64)
    print(df_diabetes)
    result = ValuePredictor_Diabetes(df_diabetes.values) 
    return result[1]

#preprocessing functions
def PreprocessData_Heart(df):
    a = pd.get_dummies(df['cp'], prefix = "cp")
    b = pd.get_dummies(df['thal'], prefix = "thal")
    c = pd.get_dummies(df['slope'], prefix = "slope")
    frames = [df, a, b, c]
    df = pd.concat(frames, axis = 1)
    df = df.drop(columns = ['cp', 'thal', 'slope'])
    if 'cp_0' in df.columns:
        df['cp_1']=0
        df['cp_2']=0
        df['cp_3']=0
    elif 'cp_1' in df.columns:
        df['cp_0']=0
        df['cp_2']=0
        df['cp_3']=0
    elif 'cp_2' in df.columns:
        df['cp_0']=0
        df['cp_1']=0
        df['cp_3']=0
    elif 'cp_3' in df.columns:
        df['cp_0']=0
        df['cp_1']=0
        df['cp_2']=0
    if 'thal_0' in df.columns:
        df['thal_1']=0
        df['thal_2']=0
        df['thal_3']=0
    elif 'thal_1' in df.columns:
        df['thal_0']=0
        df['thal_2']=0
        df['thal_3']=0
    elif 'thal_2' in df.columns:
        df['thal_0']=0
        df['thal_1']=0
        df['thal_3']=0
    elif 'thal_3' in df.columns:
        df['thal_0']=0
        df['thal_1']=0
        df['thal_2']=0
    if 'slope_0' in df.columns:
        df['slope_1']=0
        df['slope_2']=0
    elif 'slope_1' in df.columns:
        df['slope_0']=0
        df['slope_2']=0
    elif 'slope_2' in df.columns:
        df['slope_0']=0
        df['slope_1']=0
    columns = ['age','sex','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','ca', 'cp_0',	'cp_1',	'cp_2',	'cp_3',	'thal_0','thal_1','thal_2','thal_3','slope_0','slope_1','slope_2']
    df = df[columns]
    return df

def PreprocessData_Liver(liver):
    liver['IB'] = liver['TB'] - liver['DB']
    liver['sg_ratio'] = liver['Sgot']/liver['Sgpt']
    liver['Gender'] = liver['Gender'].astype(int)
    a = pd.get_dummies(liver['Gender'], prefix = "Gender")
    frames = [liver, a]
    liver = pd.concat(frames, axis = 1)
    liver = liver.drop(columns = ['Gender','TB', 'DB', 'Sgot', 'Sgpt'])
    if 'Gender_0' in liver.columns:
        liver['Gender_1']=0
    elif 'Gender_1' in liver.columns:
        liver['Gender_0']=0
    columns = ['Age','Alkphos','TP','ALB','A/G','IB','sg_ratio','Gender_0','Gender_1']
    liver = liver[columns]
    return liver

#prediction functions
def ValuePredictor_Heart(df):
    loaded_model = pickle.load(open("model.pkl", "rb"))
    result = loaded_model.predict_proba(df)
    return result[0]

def ValuePredictor_Liver(df):
    loaded_model = pickle.load(open("rclf_81_oversampling.sav", "rb"))
    result = loaded_model.predict_proba(df)
    return result[0]

def ValuePredictor_Diabetes(df):
    loaded_model = pickle.load(open("diabetes_model.pkl", "rb"))
    result = loaded_model.predict_proba(df)
    return result[0]
 
#risk calculation function 
def CalcRisk(heart_proba, liver_proba, diabetes_proba):
    heart_weight=21834
    liver_weight=10918
    diabetes_weight=1194
    risk_fac=((heart_proba*heart_weight) + (liver_proba*liver_weight) + (diabetes_proba*diabetes_weight))/(heart_weight + liver_weight + diabetes_weight)
    return risk_fac

if __name__ == "__main__":
    app.run(threaded=True, port=5000)