from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os


user_name = os.environ.get('DB_USERNAME')
password = os.environ.get('DB_PASSWORD')

print(user_name)

DATABASE_URL = f"postgresql://{user_name}:{password}!@localhost/finassist"
print(DATABASE_URL)

# Set up the database engine
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()