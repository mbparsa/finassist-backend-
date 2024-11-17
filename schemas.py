from pydantic import BaseModel, Field
from typing import List, Optional

class UserCreate(BaseModel):
    name: str
    email: str
    password: str

class User(BaseModel):
    id: int
    name: str
    email: str

    class Config:
        from_attributes = True

class ApplePayRecord(BaseModel):
    amount: Optional[str] = Field(None, alias="amount")
    category: Optional[str]
    merchant: Optional[str]
    transactionDate: Optional[str]
    type: Optional[str]

class ApplePayCSV(BaseModel):
    records: List[ApplePayRecord]

    class Config:
        allow_population_by_field_name = True