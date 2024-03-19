from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    id: str

@app.post("/process_string")
async def process_string(item: Item):
    # Xử lý logic ở đây (nếu cần)
    return {"message": "hello"}

@app.options("/process_string")
async def options_process_string():
    # Trả về phản hồi để cho phép yêu cầu OPTIONS
    return {"Allow": "POST"}
