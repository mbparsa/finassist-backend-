from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
import models, schemas, crud
from database import engine, Base, get_db
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List
import matplotlib.pyplot as plt
import io
import base64
from collections import defaultdict
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


@app.post("/transactions/applepayscv/", response_model=list[schemas.ApplePayRecord])
async def upload_apple_csv(apple_pay_csv: List[schemas.ApplePayRecord]):
    data = [record.model_dump(by_alias=True) for record in apple_pay_csv]


    # Aggregate data by merchant
    merchant_totals = defaultdict(float)
    for record in data:
        amount = record['amount']
        if record['amount'] is not None:
            try:
                amount = float(amount)
                merchant_totals[record['merchant']] += amount
            except ValueError:
                continue


    # Extract data for plotting
    merchants = list(merchant_totals.keys())
    amounts = list(merchant_totals.values())

    # Sort merchants by total amount in descending order and select top 20
    sorted_merchants = sorted(merchant_totals.items(), key=lambda x: x[1], reverse=True)[:20]
    merchants, amounts = zip(*sorted_merchants) if sorted_merchants else ([], [])


    # Check if there is data to plot
    if not merchants or not amounts:
        return JSONResponse(content={"message": "No valid data to plot", "data": data, "plot": None})

    # Plot the data
    plt.figure(figsize=(10, 5))
    plt.bar(merchants, amounts)
    plt.xlabel('merchant')
    plt.ylabel('Total Amount')
    plt.title('Top 20 Merchants by Total Amount of Purchases')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    # Encode the image to base64
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    return JSONResponse(content={"message": "Data processed successfully",  "plot": img_base64})

@app.get("/users/", response_model=list[schemas.User])
def read_users(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
    users = crud.get_users(db, skip=skip, limit=limit)
    return users