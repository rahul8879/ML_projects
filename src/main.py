from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/items")
async def read_items():
    return "This is the items endpoint"

@app.post("/item_create")
async def create_item(item: dict):
    return {"message": "Item created", "item": item}
