# 5-Minute Setup Guide

Get Research Reproducer online in just 5 minutes!

## Option 1: GitHub Pages (Frontend Only) - 2 Minutes

### Step 1: Enable GitHub Pages
1. Go to your repository on GitHub
2. Click **Settings**
3. Scroll to **Pages** section (left sidebar)
4. Under "Source":
   - Branch: `main`
   - Folder: `/docs`
5. Click **Save**

### Step 2: Wait for Deployment
- GitHub will build and deploy automatically (~1 minute)
- Check the Actions tab to see progress

### Step 3: Visit Your Site
- URL: `https://yourusername.github.io/Reasearch-Litrature_Review/`
- The site works in **demo mode** (simulated data)

**You're done!** 🎉

---

## Option 2: Hugging Face Spaces (Full Stack) - 5 Minutes

Get a fully working backend + frontend for FREE!

### Step 1: Create Hugging Face Account
1. Go to https://huggingface.co/
2. Click "Sign Up" (free, no credit card needed)
3. Verify your email

### Step 2: Create New Space
1. Click your profile → **New Space**
2. Fill in:
   - **Name**: `research-reproducer`
   - **License**: MIT
   - **SDK**: Gradio
   - **Visibility**: Public (free)
3. Click **Create Space**

### Step 3: Upload Files
You can either:

**A) Upload via Web Interface:**
1. Click **Files** tab in your space
2. Upload these files from your repo:
   - `app.py`
   - `requirements.txt`
   - `README_HF.md` (rename to `README.md`)
   - Copy entire `src/` folder

**B) Or Use Git:**
```bash
# Clone your HF space
git clone https://huggingface.co/spaces/YOUR_USERNAME/research-reproducer
cd research-reproducer

# Copy files
cp /path/to/repo/app.py .
cp /path/to/repo/requirements.txt .
cp /path/to/repo/README_HF.md README.md
cp -r /path/to/repo/src .

# Push
git add .
git commit -m "Initial deployment"
git push
```

### Step 4: Wait for Build
- HF Spaces will automatically build (~3-5 minutes)
- Watch the progress in the "Logs" section
- When done, you'll see "Running" status

### Step 5: Test Your Space
- URL: `https://huggingface.co/spaces/YOUR_USERNAME/research-reproducer`
- Try reproducing a paper!
- Share the link with anyone

**You're done!** 🚀

### Step 6 (Optional): Connect to GitHub Pages
Make your static website use the real backend:

1. Edit `docs/app.js` in your GitHub repo
2. Update line 5:
   ```javascript
   const CONFIG = {
       API_URL: 'https://YOUR_USERNAME-research-reproducer.hf.space',
       USE_DEMO_MODE: false,
   };
   ```
3. Commit and push
4. GitHub Pages will auto-update

Now you have:
- ✅ Beautiful static website on GitHub Pages
- ✅ Fully working backend on Hugging Face Spaces
- ✅ Both are FREE!

---

## Recommended: Add GitHub Token

Increase API rate limits (optional but recommended):

### On Hugging Face Spaces:
1. Go to your Space → **Settings**
2. Scroll to **Repository secrets**
3. Click **New secret**
4. Name: `GITHUB_TOKEN`
5. Value: Your GitHub token ([create one here](https://github.com/settings/tokens))
6. Click **Add secret**

### Get GitHub Token:
1. Go to https://github.com/settings/tokens
2. Click **Generate new token (classic)**
3. Give it a name: "Research Reproducer"
4. Select scopes: `public_repo` (read access)
5. Click **Generate token**
6. Copy the token (save it somewhere!)

---

## Troubleshooting

### GitHub Pages not working
- Check Settings → Pages shows green checkmark
- Wait 5 minutes after first setup
- Check Actions tab for build errors
- Make sure `/docs` folder exists with `index.html`

### Hugging Face Space not building
- Check logs for error messages
- Verify all files uploaded correctly
- Make sure `requirements.txt` is present
- Try restarting the space (Settings → Factory reboot)

### "Backend not responding" on website
- If using demo mode: it's supposed to show fake data
- If connected to HF Spaces: wait for space to wake up (first request takes ~30s)
- Check HF Space logs for errors
- Verify API_URL in `docs/app.js` is correct

---

## What You Get

### GitHub Pages Only
- ✅ Beautiful static website
- ✅ Demo mode (simulated data)
- ✅ Fast loading
- ✅ Free hosting
- ❌ Can't run real reproductions

### GitHub Pages + Hugging Face Spaces
- ✅ Beautiful static website
- ✅ Real paper reproductions
- ✅ Fully functional backend
- ✅ Share with anyone
- ✅ 100% FREE
- ✅ No credit card needed
- ✅ Auto-scaling
- ✅ HTTPS included

---

## Next Steps

After deploying:

1. **Test it**: Try reproducing `1706.03762` (Attention Is All You Need)
2. **Share it**: Send the link to colleagues
3. **Customize it**: Edit `docs/index.html` to change colors, text, etc.
4. **Star the repo**: Help others find it!
5. **Contribute**: Open PRs with improvements

---

## Cost Breakdown

| What | Cost | Notes |
|------|------|-------|
| GitHub Pages | **$0** | Unlimited bandwidth |
| Hugging Face Spaces | **$0** | Free tier, may sleep after inactivity |
| Domain name | $0-$10/year | Optional, use .github.io for free |
| **Total** | **$0** | 100% free! |

---

## Support

Need help?
- 📖 Read [DEPLOYMENT.md](../DEPLOYMENT.md) for detailed guide
- 🐛 Open an issue on GitHub
- 💬 Ask in HF Spaces community
- 📧 Contact repository maintainers

---

Happy deploying! 🚀✨
