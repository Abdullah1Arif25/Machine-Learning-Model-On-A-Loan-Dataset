

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
#Function to check if the file exist and throws exception instead of crashing the code                         
def get_file(file_path):
        
        while True:
                try:
                        data_set = pd.read_csv(file_path)
                        
                        return data_set
                        break
                except FileNotFoundError:
                        print("file not found")

#Cleaning the Data to remove any "NA" values. 
def cleaned_data(cleaned_data):
        print("Cleaning your data...")
        data = cleaned_data.dropna()
        cleaned_data["earliest_cr_line"] = pd.to_datetime(cleaned_data["earliest_cr_line"])
        return cleaned_data

# Changing the Dtype of Attributes to avoid error.      
def correcting_Data(df_cleaned):        
        df_cleaned['grade_num'] = pd.factorize(df_cleaned['grade'])[0].copy() 
        df_cleaned['Lending_Reason_num'] = pd.factorize(df_cleaned['title'])[0].copy()
        df_cleaned['home_ownership_num'] = pd.factorize(df_cleaned['home_ownership'])[0].copy()
        df_cleaned['Time_period_num']  = df_cleaned['term'].str.extract(r'(\d+)').astype(float) 
        return df_cleaned


# Building a Training Model that will take cleaned data and train the model and also takes input for "Prediction".
def predict_grade(df_cleaned):

        features = [ 'loan_amnt',  'Time_period_num', 'int_rate',
               'installment', 'Lending_Reason_num', 'home_ownership_num',
               'annual_inc',  'pub_rec', 'revol_util', 'tot_coll_amt', 'tot_cur_bal', 'total_acc',
               'open_acc', 'total_rev_hi_lim']
        X = df_cleaned[features]
        Y = df_cleaned['grade_num']
        

        print("Training on your data...")
        Loan_model = RandomForestClassifier(random_state = 0)

        Loan_model.fit(X,Y)
        print("Done\nPlease Enter The Customer Specifications")
        try:
                
                user_input = []

                user_input.append(float(input("Enter the Loan_amount: \n")))
                user_input.append(float(input("Enter the Time period for loan as:\n(monthes = 36)\n(monthes = 60)) \n")))
                user_input.append(float(input("Enter the interest rate:\n (Withour percentage symbol) \n").strip('%')))
                user_input.append(float(input("Enter the installment paid so far: \n")))
                user_input.append(input("""Enter the Lending Reason as:
                Debt consolidation         = 0
                Home improvement           = 1
                Credit card refinancing    = 2
                Other                      = 3
                Medical expenses           = 4
                Business                   = 5
                Home buying                = 6
                Major purchase             = 7
                Moving and relocation      = 8
                Car financing              = 9
                Vacation                   = 10
                Green loan                 = 11\n"""))
                user_input.append(int(input("""Enter the Home Ownership status as :
                MORTGAGE   = 0
                RENT       = 1
                OWN        = 2
                ANY        = 3

                """)))
                user_input.append(float(input("Enter the annual income :  \n")))
                user_input.append(float(input("""Enter the public record as : 
                1.0   
                2.0   
                3.0   
                4.0   
                5.0   
                6.0   
                7.0   
                8.0   
                9.0   
                10.0  
                11.0
                12.0    
                44.0  
                28.0  
                """)))
                user_input.append(float(input("Enter the the amount of credit the borrower is using relative to all available revolving credit:  \n")))
                user_input.append(float(input("Enter the Total Amount Collected :  \n")))
                user_input.append(float(input("Enter the total current balance of all accounts:  \n")))
                user_input.append(float(input("Enter the total number of credit lines currently in the borrower's credit file:  \n")))
                user_input.append(input("Enter the number of open credit lines in the borrower's credit file :  \n"))
                user_input.append(float(input("Enter the total total credit limit on revolving accounts :  \n")))
                print("Thanks for the Input please wait")

                user_input_df = pd.DataFrame([user_input], columns=features)
                prediction_Y = Loan_model.predict(user_input_df)

                # As we can see that the prediction is in the form of array of the numerical figures. In order to understant what these predictions are, we must revert the change using factorize

                original_grades = pd.factorize(df_cleaned['grade'])[1]
                # We will map the "prediction_grades" to the "original_grades"
                prediction_grades = original_grades[prediction_Y]

                likely = ""
                if prediction_grades == 'A':
                        likely = "more"
                elif prediction_grades =='B':
                        likely = "almost"
                else:
                        likely = "less"
                print(f"The Customer {likely} likey to pay back with  {prediction_grades} and its 94% accurate")
        except Exception as e:
                print(e)

def main():
        file_url = input(f"Enter The File url in formate: ")
        data = get_file(file_url)
        clean_data = cleaned_data(data)
        dataframe_clean= correcting_Data(clean_data)
        predict_grade(dataframe_clean)


        


if __name__ == "__main__":
        main()
