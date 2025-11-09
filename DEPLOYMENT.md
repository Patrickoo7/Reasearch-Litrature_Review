
# Deployment Guide

This guide explains how to deploy Research Reproducer to make it accessible online.

## Option 1: GitHub Pages (Frontend Only)

The static website is automatically deployed to GitHub Pages.

### Steps:

1. **Enable GitHub Pages**
   - Go to your repository settings
   - Navigate to Pages section
   - Set source to `main` branch and `/docs` folder
   - Save

2. **Access Your Site**
   - Your site will be available at: `https://patrickoo7.github.io/Reasearch-Litrature_Review/`
   - Note: This is frontend only (demo mode)

3. **Connect to Backend** (after deploying backend)
   - Edit `docs/app.js`
   - Update `API_URL` to your backend URL
   - Set `USE_DEMO_MODE` to `false`
   - Commit and push changes

## Option 2: Hugging Face Spaces (Full Stack - FREE!)

Deploy both frontend and backend to Hugging Face Spaces.

### Steps:

1. **Create Hugging Face Account**
   - Go to https://huggingface.co/
   - Sign up (free)

2. **Create New Space**
   - Click "New" → "Space"
   - Name: `research-reproducer`
   - SDK: Gradio
   - Visibility: Public (free)

3. **Clone and Push**
   ```bash
   # Clone your HF space
   git clone https://huggingface.co/spaces/YOUR_USERNAME/research-reproducer
   cd research-reproducer

   # Copy files from Research Reproducer
   cp -r /path/to/Reasearch-Litrature_Review/src .
   cp /path/to/Reasearch-Litrature_Review/app.py .
   cp /path/to/Reasearch-Litrature_Review/requirements.txt .
   cp /path/to/Reasearch-Litrature_Review/README_HF.md ./README.md

   # Commit and push
   git add .
   git commit -m "Initial deployment"
   git push
   ```

4. **Configure Secrets** (Optional but recommended)
   - Go to your Space settings
   - Add secret: `GITHUB_TOKEN` with your GitHub token
   - This increases API rate limits

5. **Access Your Space**
   - URL: `https://huggingface.co/spaces/YOUR_USERNAME/research-reproducer`
   - Share with anyone!

### Update Frontend to Use Your Backend

1. Edit `docs/app.js` in your GitHub repo:
   ```javascript
   const CONFIG = {
       API_URL: 'https://YOUR_USERNAME-research-reproducer.hf.space',
       USE_DEMO_MODE: false,
   };
   ```

2. Commit and push - GitHub Pages will auto-update

## Option 3: Render.com (Backend API)

Deploy as a web service on Render.

### Steps:

1. **Create Account** at https://render.com (free tier available)

2. **Create Web Service**
   - New → Web Service
   - Connect your GitHub repository
   - Name: `research-reproducer-api`
   - Environment: Python 3
   - Build Command: `pip install -e .`
   - Start Command: `research-reproduce web --port $PORT`

3. **Configure**
   - Add environment variable: `GITHUB_TOKEN`
   - Deploy

4. **Get URL**
   - Your API: `https://research-reproducer-api.onrender.com`
   - Update `docs/app.js` with this URL

## Option 4: Railway.app (Backend API)

Similar to Render, with nice UI.

### Steps:

1. **Create Account** at https://railway.app

2. **New Project**
   - Deploy from GitHub repo
   - Select your repository

3. **Configure**
   - Add start command: `research-reproduce web --port $PORT`
   - Add `GITHUB_TOKEN` variable

4. **Deploy**
   - Get your URL
   - Update frontend configuration

## Option 5: Local Deployment

Run locally for development or private use.

### Steps:

1. **Install Dependencies**
   ```bash
   pip install -e .
   ```

2. **Run Web Interface**
   ```bash
   research-reproduce web
   ```

3. **Access**
   - Open http://localhost:7860
   - For network access: `research-reproduce web --share`

## Recommended Setup

For best results, combine:

1. **Frontend**: GitHub Pages (free, fast CDN)
   - Website: `https://yourusername.github.io/Reasearch-Litrature_Review/`

2. **Backend**: Hugging Face Spaces (free, auto-scaling)
   - API: `https://yourusername-research-reproducer.hf.space`

This gives you a fully functional, free, public website!

## CORS Configuration

If you get CORS errors when connecting frontend to backend:

### For Gradio (Hugging Face Spaces)

Gradio handles CORS automatically, but ensure:
```python
# In web_interface.py
interface.launch(
    server_port=7860,
    enable_queue=True,
    share=False  # HF Spaces handles sharing
)
```

### For Custom Backend

Add CORS headers:
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourusername.github.io"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Environment Variables

Set these in your deployment platform:

- `GITHUB_TOKEN`: GitHub API token (optional, increases rate limits)
- `PORT`: Port to run on (usually set automatically)

## Monitoring

### Hugging Face Spaces
- View logs in the space's "Logs" tab
- Check usage in Analytics

### Render/Railway
- Built-in monitoring dashboards
- Set up alerts for errors

## Updating

### GitHub Pages
Just push to main branch - auto-updates

### Hugging Face Spaces
```bash
cd your-space-repo
git pull origin main  # Get latest changes
# Make updates
git push  # Deploys automatically
```

## Cost Comparison

| Platform | Frontend | Backend | Total | Notes |
|----------|----------|---------|-------|-------|
| GitHub Pages + HF Spaces | Free | Free | **$0** | Recommended |
| GitHub Pages + Render | Free | $0-7/mo | $0-7 | Free tier limited |
| GitHub Pages + Railway | Free | $0-5/mo | $0-5 | Free tier limited |
| Local Only | N/A | N/A | $0 | No public access |

## Security Notes

- Never commit `GITHUB_TOKEN` to git
- Use environment variables for secrets
- HF Spaces secrets are encrypted
- Set appropriate CORS origins
- Use rate limiting for public APIs

## Troubleshooting

### "Backend not responding"
- Check backend is deployed and running
- Verify API_URL in `docs/app.js`
- Check browser console for CORS errors
- Test backend directly: visit API_URL in browser

### "Rate limit exceeded"
- Add GitHub token to backend
- Cache is enabled by default
- Consider upgrading API limits

### "Space sleeping" (Hugging Face)
- Free tier spaces sleep after inactivity
- First request wakes it up (may take 30s)
- Upgrade to prevent sleeping

## Support

- Documentation: See README.md
- Issues: https://github.com/Patrickoo7/Reasearch-Litrature_Review/issues
- Examples: See QUICKSTART.md
