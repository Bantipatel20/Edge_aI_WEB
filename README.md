# Edge AI Project

Vercel-compatible architecture with a separated frontend and ML backend.

## Stack
- Frontend: Next.js (Vercel)
- Backend: FastAPI + PyTorch (Render, Railway, or AWS)

## Project Structure
- `frontend/`: Next.js UI for image upload and result display
- `backend/`: FastAPI model inference API
- `docs/`: Optional architecture images and project docs

## Run Locally

### 1) Backend
1. Open terminal in `backend/`
2. Create and activate a Python environment
3. Install dependencies:
   - `pip install -r requirements.txt`
4. Put your model file at `backend/weights/best_densenet_9ch.pth`
5. Run API:
   - `uvicorn app:app --host 0.0.0.0 --port 10000`

### 2) Frontend
1. Open terminal in `frontend/`
2. Install dependencies:
   - `npm install`
3. Configure API URL in `.env.local`
4. Run app:
   - `npm run dev`

## Deploy

### Backend (Render)
Use one of these two methods:

#### Option A: Docker deploy (recommended)
1. Push this repo to GitHub.
2. In Render: New > Web Service > Build and deploy from a Git repository.
3. Select this repo.
4. Render auto-detects `render.yaml` from project root.
5. Deploy.

Expected backend URL format:
- `https://edge-ai-backend.onrender.com`

#### Option B: Manual Web Service settings
If you do not use `render.yaml`, use:
- Root directory: `backend`
- Runtime: `Python 3`
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn app:app --host 0.0.0.0 --port 10000`
- Environment variables (Render):
   - `MODEL_WEIGHTS_PATH=weights/best_densenet_9ch.pth`
   - Optional if file is not in repo: `MODEL_WEIGHTS_URL=https://.../best_densenet_9ch.pth`

### Frontend (Vercel)
1. Push repo to GitHub.
2. In Vercel: Add New > Project > Import the same repository.
3. Set Root Directory to `frontend`.
4. Add environment variable in Vercel project settings:
    - `NEXT_PUBLIC_API_URL=https://your-api.onrender.com`
5. Deploy.

Expected frontend URL format:
- `https://your-project.vercel.app`

### Connect Frontend to Backend
1. Deploy backend on Render first.
2. Copy backend URL.
3. Set `NEXT_PUBLIC_API_URL` in Vercel to that URL.
4. Redeploy frontend.

## Notes
- Model weights stay in backend only.
- Grad-CAM scaffold exists in `backend/gradcam.py` and can be integrated later.

### Model Weights Note
- Your backend expects: `backend/weights/best_densenet_9ch.pth`.
- If this file is larger than GitHub file-size limits, host it externally (S3, Hugging Face, or cloud storage) and set `MODEL_WEIGHTS_URL`.
