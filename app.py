from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from product_rephraser import ProductRefiner

# ──────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────
load_dotenv()
app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Welcome to the Product Refiner API!"}


@app.post("/refine")
async def refine_product(request: Request, x_api_key: Optional[str] = Header(None)):
    if not x_api_key:
        raise HTTPException(
            status_code=401, detail="Missing OpenAI API Key in 'x-api-key' header"
        )

    raw = await request.body()
    text = raw.decode("utf-8", errors="ignore")

    # --- extract fields manually to handle HTML ---
    try:
        after_title = text.split('"title":', 1)[1].lstrip()
        if not after_title.startswith('"'):
            raise ValueError("No opening quote for title")
        after_title = after_title[1:]
        title, _ = after_title.split('",', 1)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to extract title: {e}")

    try:
        after_desc = text.split('"description":', 1)[1].lstrip()
        if not after_desc.startswith('"'):
            raise ValueError("No opening quote for description")
        after_desc = after_desc[1:]
        if after_desc.endswith('"}'):
            desc = after_desc[:-2]
        elif after_desc.endswith('" }'):
            desc = after_desc[:-3]
        else:
            desc = after_desc.rstrip()
            if desc.endswith('"'):
                desc = desc[:-1]
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to extract description: {e}"
        )

    try:
        after_id = text.split('"product_id":', 1)[1].lstrip()
        if not after_id.startswith('"'):
            raise ValueError("No opening quote for product_id")
        after_id = after_id[1:]
        product_id, _ = after_id.split('"', 1)
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to extract product_id: {e}"
        )

    # --- call refiner ---
    try:
        refiner = ProductRefiner(api_key=x_api_key)
        result = refiner.refine(product_id, title, desc)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
