from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
import models, schemas, crud
from database import engine, Base, get_db
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from typing import List
import matplotlib.pyplot as plt
import io
import base64
from collections import defaultdict
import re
import pandas as pd
from merchant_month import merchant
import json

import re
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4')


from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression


# Create all tables in the database
Base.metadata.create_all(bind=engine)

app = FastAPI()

origins = [
    "http://localhost:3000",  # React frontend address
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.post("/users/", response_model=schemas.User)
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    return crud.create_user(db=db, user=user)



def clean_merchant_name(merchant_name):
    if merchant_name is None or not isinstance(merchant_name, str):
        return ""
  # Return an empty string instead of None for empty or invalid inputs

    # Remove anything before and including the first '*'
    name = re.sub(r'^.*?\*', '', merchant_name)

    # Remove numbers (standalone and attached to words)
    name = re.sub(r'\d+', '', name)

    # Remove any remaining special characters
    name = re.sub(r'[^\w\s]', '', name)

    # Convert to lowercase
    name = name.lower()
    print(name)
  


    # #Lemmatize words to their root form
    # lemmatizer = WordNetLemmatizer()
    # name_words = [lemmatizer.lemmatize(word) for word in name_words]
    # #Remove common words
    # name_words = [word for word in name_words if word not in common_words]
    # #Remove extra spaces and join words
    # name = ' '.join(name_words).strip()

    return name if name else ""



@app.post("/transactions/applepayscv/", response_model=list[schemas.ApplePayRecord])
async def upload_apple_csv(apple_pay_csv: List[schemas.ApplePayRecord]):
    data = [record.model_dump(by_alias=True) for record in apple_pay_csv]

    df = pd.DataFrame(data)


    # Aggregate data by merchant
    # merchant_totals = defaultdict(float)
    # for record in data:
    #     amount = record['amount']
    #     if record['amount'] is not None:
    #         try:
    #             amount = float(amount)
    #             merchant_totals[record['merchant']] += amount
    #         except ValueError:
    #             continue


    # Extract data for plotting
    # merchants = list(merchant_totals.keys())
    # print(merchants)
    # amounts = list(merchant_totals.values())

    # # Sort merchants by total amount in descending order and select top 20
    # sorted_merchants = sorted(merchant_totals.items(), key=lambda x: x[1], reverse=True)[:20]
    # merchants, amounts = zip(*sorted_merchants) if sorted_merchants else ([], [])


    # # Check if there is data to plot
    # if not merchants or not amounts:
    #     return JSONResponse(content={"message": "No valid data to plot", "data": data, "plot": None})



    # Step 2: Data Preprocessing
    # Convert 'amount' to float
    df['amount'] = df['amount'].astype(float)

    # Convert 'transactionDate' to datetime
    df['transactionDate'] = pd.to_datetime(df['transactionDate'], format='%m/%d/%Y')

    # Handle missing values (if any)
    df.fillna('', inplace=True)

    # Step 3: Preprocess Merchant Names

    df['Cleaned Merchant'] = df['merchant'].apply(clean_merchant_name)
    merchant_names = df['Cleaned Merchant'].tolist()

    # Generate embeddings for merchant names using a pre-trained model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    merchant_embeddings = model.encode(merchant_names)

    # Cluster similar merchant names
    clustering_model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0.5,
        metric='euclidean',
        linkage='ward'
    )
    clustering_model.fit(merchant_embeddings)
    df['Merchant Cluster'] = clustering_model.labels_


    # Step 4: Categorize Transactions Using Machine Learning
    if 'category' in df.columns and not df['category'].isnull().any():
        # Prepare text data
        df['Text'] = df['Merchant Cluster'].astype(str)
        X = df['Text']
        y = df['category']

        # Encode labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )

        # Generate embeddings for text
        X_train_embeddings = model.encode(X_train.tolist())
        X_test_embeddings = model.encode(X_test.tolist())

        # Train a classifier
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train_embeddings, y_train)

        # Predict categories for the entire dataset
        df_embeddings = model.encode(df['Text'].tolist())
        df['Predicted Category'] = le.inverse_transform(clf.predict(df_embeddings))
    else:
        print("Category labels are missing. Skipping categorization step.")
        df['Predicted Category'] = 'Uncategorized'

    # Step 5: Extract Transaction Month
    df['Transaction Month'] = df['transactionDate'].dt.to_period('M')

    # Step 6: Aggregate Spending
    aggregated_spending = df.groupby(
        ['Merchant Cluster', 'Transaction Month']
    )['amount'].sum().reset_index()

    # Map Merchant Cluster to a representative Merchant Name
    cluster_to_merchant = df.groupby('Merchant Cluster')['Cleaned Merchant'].agg(
        lambda x: x.value_counts().index[0]
    ).reset_index()
    aggregated_spending = aggregated_spending.merge(
        cluster_to_merchant, on='Merchant Cluster'
    )

    # Step 7: Save the Aggregated Spending to a CSV File
    aggregated_spending.to_csv('aggregated_spending.csv', index=False)

    #filter out what we want to hold 
    filtered_aggregated_spending = aggregated_spending[['Cleaned Merchant', 'Transaction Month', 'amount']]
    
    filtered_aggregated_spending.rename(columns={'Cleaned Merchant': 'merchant', 'Transaction Month': 'month'}, inplace=True)

    if 'month' in filtered_aggregated_spending.columns:
        filtered_aggregated_spending['month'] = (
            filtered_aggregated_spending['month'].astype(str)
        )
 
  
    # set the data to the merchant_month class
    print(filtered_aggregated_spending)
    merchant_records =  merchant()
    merchant_records.set_merchant(filtered_aggregated_spending)
    print(merchant().get_merchant())
    return JSONResponse(content={"message": "Data processed successfully"})


@app.get("/users/", response_model=list[schemas.User])
def read_users(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
    users = crud.get_users(db, skip=skip, limit=limit)
    return users

@app.get("/transactions/merchant/")
async def get_merchant_data():
   
    data = merchant_month().get_Df()
    print(data)
    json_data = jsonable_encoder(data.to_dict(orient="records"))
    print(json_data)
    return JSONResponse(content=json_data)






#def plot_data(merchants, amounts):
    # Plot the data
    # plt.figure(figsize=(10, 5))
    # plt.bar(merchants, amounts)
    # plt.xlabel('merchant')
    # plt.ylabel('Total Amount')
    # plt.title('Top 20 Merchants by Total Amount of Purchases')
    # plt.xticks(rotation=45)
    # plt.tight_layout()

    # # Save the plot to a BytesIO object
    # buf = io.BytesIO()
    # plt.savefig(buf, format='png')
    # buf.seek(0)
    # plt.close()

    # # Encode the image to base64
    # img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
