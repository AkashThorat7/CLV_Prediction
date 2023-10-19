import pickle
import json
import numpy as np 
from sklearn.preprocessing import FunctionTransformer, PowerTransformer
from scipy.stats import skew, boxcox, yeojohnson
import warnings
warnings.filterwarnings('ignore')

class CustomerLifetimeValue():
    

    def __init__(self, Response, Education, Gender, Income, Monthly_Premium_Auto, Months_Since_Last_Claim, Months_Since_Policy_Inception, Number_of_Open_Complaints,
                 Number_of_Policies, Renew_Offer_Type, Total_Claim_Amount, State, Coverage, EmploymentStatus,  Location_Code, Marital_Status,
                 Policy_Type, Policy, Sales_Channel, Vehicle_Class, Vehicle_Size):
        

        self.Response = Response
        self.Education = Education
        self.Gender = Gender
        self.Income = Income
        self.Monthly_Premium_Auto = Monthly_Premium_Auto
        self.Months_Since_Last_Claim = Months_Since_Last_Claim
        self.Months_Since_Policy_Inception = Months_Since_Policy_Inception
        self.Number_of_Open_Complaints = Number_of_Open_Complaints
        self.Number_of_Policies = Number_of_Policies
        self.Renew_Offer_Type = Renew_Offer_Type
        self.Total_Claim_Amount = Total_Claim_Amount
        self.State = 'State_'+State
        self.Coverage = 'Coverage_'+Coverage 
        self.EmploymentStatus =  'EmploymentStatus_'+EmploymentStatus 
        self.Location_Code = 'Location_Code_'+Location_Code 
        self.Marital_Status = 'Marital_Status_'+Marital_Status 
        self.Policy_Type = 'Policy_Type_'+Policy_Type 
        self.Policy = 'Policy_'+Policy 
        self.Sales_Channel = 'Sales_Channel_'+Sales_Channel     
        self.Vehicle_Class = 'Vehicle_Class_'+Vehicle_Class 
        self.Vehicle_Size = 'Vehicle_Size_'+Vehicle_Size 

    def load_model(self):

        with open('project_app/Random_Forest.pkl','rb') as f:

            self.model = pickle.load(f)


        with open('project_app/transform.pkl', 'rb') as f:
            self.transformation = pickle.load(f)

        with open('project_app/project_data.json','rb') as f:

            self.project_data = json.load(f)

    def get_predicted_clv(self):

        self.load_model()
        test_array = np.zeros(len(self.project_data['columns']))

        test_array[0] = self.project_data['Response'][self.Response]
        test_array[1] = self.project_data['Education'][self.Education]
        test_array[2] = self.project_data['Gender'][self.Gender]
        test_array[3] = self.Income
        test_array[4] = self.Monthly_Premium_Auto
        test_array[5] = self.Months_Since_Last_Claim
        test_array[6] = self.Months_Since_Policy_Inception
        test_array[7] = self.Number_of_Open_Complaints
        test_array[8] = self.Number_of_Policies
        test_array[9] = self.project_data['Renew_Offer_Type'][self.Renew_Offer_Type]
        test_array[10] = self.Total_Claim_Amount

        State_index = self.project_data['columns'].index(self.State)
        test_array[State_index] = 1

        Coverage_index = self.project_data['columns'].index(self.Coverage)
        test_array[Coverage_index] = 1
        
        EmploymentStatus_index = self.project_data['columns'].index(self.EmploymentStatus)
        test_array[EmploymentStatus_index] = 1

        Location_Code_index = self.project_data['columns'].index(self.Location_Code)
        test_array[Location_Code_index] = 1

        Marital_Status_index = self.project_data['columns'].index(self.Marital_Status)
        test_array[Marital_Status_index] = 1

        Policy_Type_index = self.project_data['columns'].index(self.Policy_Type)
        test_array[Policy_Type_index] = 1

        Policy_index = self.project_data['columns'].index(self.Policy)
        test_array[Policy_index] = 1

        Sales_Channel_index = self.project_data['columns'].index(self.Sales_Channel)
        test_array[Sales_Channel_index] = 1

        Vehicle_Class_index = self.project_data['columns'].index(self.Vehicle_Class)
        test_array[Vehicle_Class_index] = 1

        Vehicle_Size_index = self.project_data['columns'].index(self.Vehicle_Size)
        test_array[Vehicle_Size_index] = 1

        print('Test Array :', test_array)

        predicted_clv = self.model.predict([test_array])
        print('We Are In Precdiction Method')

        clv = self.transformation.inverse_transform(np.array(predicted_clv).reshape(-1, 1))[:,0][0]
        print(f"Customer life time value is  : Dollar. {clv}")
        return clv


    
if __name__ == '__main__':

    Response = 'Yes'
    Education = 'Doctor'
    Gender = 'F'
    Income = 56274
    Monthly_Premium_Auto = 93
    Months_Since_Last_Claim = 18
    Months_Since_Policy_Inception = 50
    Number_of_Open_Complaints = 0
    Number_of_Policies = 2
    Renew_Offer_Type = 'Offer2'
    Total_Claim_Amount = 511.2
    State = 'Washington'
    Coverage = 'Premium'
    EmploymentStatus = 'Employed'
    Location_Code = 'Suburban'
    Marital_Status = 'Married'
    Policy_Type = 'Special Auto'
    Policy = 'Special L2'
    Sales_Channel = 'Branch'
    Vehicle_Class = 'SUV'
    Vehicle_Size = 'Medsize'

    
    clv = CustomerLifetimeValue(Response, Education, Gender, Income, Monthly_Premium_Auto, Months_Since_Last_Claim, Months_Since_Policy_Inception, Number_of_Open_Complaints,
                 Number_of_Policies, Renew_Offer_Type, Total_Claim_Amount, State, Coverage, EmploymentStatus,  Location_Code, Marital_Status, Policy_Type, Policy, Sales_Channel, Vehicle_Class, Vehicle_Size)

    clv.get_predicted_clv()
