// Research Reproducer - Frontend Application

// Configuration
const CONFIG = {
    // Backend API URL - update this when you deploy the backend
    API_URL: 'http://localhost:7860',  // Change to your deployed API URL
    USE_DEMO_MODE: true,  // Set to false when backend is deployed
};

// Tab switching
function switchTab(tabName) {
    // Update tab buttons
    const tabs = document.querySelectorAll('.tab');
    tabs.forEach(tab => tab.classList.remove('active'));
    event.target.closest('.tab').classList.add('active');

    // Update tab content
    const contents = document.querySelectorAll('.tab-content');
    contents.forEach(content => content.classList.remove('active'));
    document.getElementById(`${tabName}-tab`).classList.add('active');
}

// Reproduce form handler
document.getElementById('reproduce-form').addEventListener('submit', async (e) => {
    e.preventDefault();

    const paperSource = document.getElementById('paper-source').value;
    const sourceType = document.getElementById('source-type').value;

    if (CONFIG.USE_DEMO_MODE) {
        handleDemoMode('reproduce', paperSource, sourceType);
    } else {
        await handleReproduction(paperSource, sourceType);
    }
});

// Analyze form handler
document.getElementById('analyze-form').addEventListener('submit', async (e) => {
    e.preventDefault();

    const paperSource = document.getElementById('analyze-source').value;

    if (CONFIG.USE_DEMO_MODE) {
        handleDemoMode('analyze', paperSource);
    } else {
        await handleAnalysis(paperSource);
    }
});

// Handle reproduction
async function handleReproduction(paperSource, sourceType) {
    showLoading('reproduce');

    try {
        updateLoadingStatus('Connecting to backend...');

        const response = await fetch(`${CONFIG.API_URL}/api/reproduce`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                paper_source: paperSource,
                source_type: sourceType,
            }),
        });

        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }

        const data = await response.json();

        hideLoading('reproduce');
        displayResults('reproduce', data);
    } catch (error) {
        hideLoading('reproduce');
        displayError('reproduce', error.message);
    }
}

// Handle analysis
async function handleAnalysis(paperSource) {
    showLoading('analyze');

    try {
        const response = await fetch(`${CONFIG.API_URL}/api/analyze`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                paper_source: paperSource,
            }),
        });

        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }

        const data = await response.json();

        hideLoading('analyze');
        displayResults('analyze', data);
    } catch (error) {
        hideLoading('analyze');
        displayError('analyze', error.message);
    }
}

// Demo mode handler
function handleDemoMode(action, paperSource, sourceType = 'auto') {
    showLoading(action);

    // Simulate API call
    setTimeout(() => {
        hideLoading(action);

        if (action === 'reproduce') {
            const demoData = {
                success: true,
                paper: {
                    title: "Example Research Paper",
                    authors: ["John Doe", "Jane Smith"],
                    arxiv_id: paperSource.includes('arxiv') ? extractArxivId(paperSource) : paperSource,
                },
                repositories: [
                    {
                        url: "https://github.com/example/repo",
                        stars: 1234,
                        language: "Python",
                    }
                ],
                analysis: {
                    languages: ["Python"],
                    dependencies: { python: { packages: ["torch", "numpy", "pandas"] } },
                    complexity: "medium",
                    gpu_required: true,
                },
                execution: {
                    success: true,
                    execution_time: 123.45,
                },
            };
            displayResults('reproduce', demoData);
        } else {
            const demoData = {
                paper: {
                    title: "Example Research Paper",
                    authors: ["John Doe", "Jane Smith"],
                },
                repositories: [
                    {
                        url: "https://github.com/example/repo",
                        stars: 1234,
                    }
                ],
            };
            displayResults('analyze', demoData);
        }
    }, 2000);
}

// Extract arXiv ID from various formats
function extractArxivId(input) {
    const match = input.match(/(\d+\.\d+)/);
    return match ? match[1] : input;
}

// Show loading state
function showLoading(type) {
    const loadingEl = document.getElementById(type === 'reproduce' ? 'loading' : 'analyze-loading');
    loadingEl.classList.add('active');

    // Hide previous results
    const resultsEl = document.getElementById(type === 'reproduce' ? 'results' : 'analyze-results');
    resultsEl.classList.remove('active');
}

// Hide loading state
function hideLoading(type) {
    const loadingEl = document.getElementById(type === 'reproduce' ? 'loading' : 'analyze-loading');
    loadingEl.classList.remove('active');
}

// Update loading status message
function updateLoadingStatus(message) {
    const statusEl = document.getElementById('loading-status');
    if (statusEl) {
        statusEl.textContent = message;
    }
}

// Display results
function displayResults(type, data) {
    const resultsEl = document.getElementById(type === 'reproduce' ? 'results' : 'analyze-results');
    const contentEl = document.getElementById(type === 'reproduce' ? 'result-content' : 'analyze-content');

    let html = '';

    if (type === 'reproduce') {
        html = generateReproductionResults(data);
    } else {
        html = generateAnalysisResults(data);
    }

    contentEl.innerHTML = html;
    resultsEl.classList.add('active');

    // Scroll to results
    resultsEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Generate reproduction results HTML
function generateReproductionResults(data) {
    const isSuccess = data.success || data.execution?.success;
    const iconClass = isSuccess ? 'success' : 'error';
    const icon = isSuccess ? 'fa-check-circle' : 'fa-times-circle';

    let html = `
        <div class="result-card">
            <div class="result-header">
                <div class="result-icon ${iconClass}">
                    <i class="fas ${icon}"></i>
                </div>
                <div>
                    <div class="result-title">${isSuccess ? 'Reproduction Successful!' : 'Reproduction Failed'}</div>
                    <div class="result-content">${data.paper?.title || 'Unknown Paper'}</div>
                </div>
            </div>
    `;

    // Paper Info
    if (data.paper) {
        html += `
            <h3 style="margin-top: 1.5rem;"><i class="fas fa-file-alt"></i> Paper Information</h3>
            <div class="result-content">
                <strong>Title:</strong> ${data.paper.title || 'N/A'}<br>
                <strong>Authors:</strong> ${(data.paper.authors || []).slice(0, 3).join(', ')}${data.paper.authors?.length > 3 ? '...' : ''}<br>
                ${data.paper.arxiv_id ? `<strong>arXiv ID:</strong> ${data.paper.arxiv_id}<br>` : ''}
            </div>
        `;
    }

    // Repositories
    if (data.repositories && data.repositories.length > 0) {
        html += `
            <h3 style="margin-top: 1.5rem;"><i class="fab fa-github"></i> Repositories Found</h3>
            <div class="result-content">
        `;
        data.repositories.slice(0, 3).forEach(repo => {
            html += `
                <div style="margin-bottom: 0.5rem;">
                    <a href="${repo.url}" target="_blank" style="color: var(--primary); text-decoration: none;">
                        <i class="fab fa-github"></i> ${repo.url}
                    </a>
                    ${repo.stars ? `<span style="color: var(--text-muted);"> (⭐ ${repo.stars})</span>` : ''}
                    ${repo.language ? `<span style="color: var(--text-muted);"> - ${repo.language}</span>` : ''}
                </div>
            `;
        });
        html += `</div>`;
    }

    // Analysis
    if (data.analysis) {
        html += `
            <h3 style="margin-top: 1.5rem;"><i class="fas fa-code"></i> Code Analysis</h3>
            <div class="result-content">
                <strong>Languages:</strong> ${(data.analysis.languages || []).join(', ')}<br>
                <strong>Complexity:</strong> ${data.analysis.complexity || 'unknown'}<br>
                ${data.analysis.gpu_required ? '<strong>GPU:</strong> <span style="color: var(--warning);">Required</span><br>' : ''}
                ${data.analysis.dependencies?.python ? `<strong>Python Packages:</strong> ${data.analysis.dependencies.python.packages?.length || 0}<br>` : ''}
            </div>
        `;
    }

    // Execution Results
    if (data.execution) {
        html += `
            <h3 style="margin-top: 1.5rem;"><i class="fas fa-play-circle"></i> Execution Results</h3>
            <div class="result-content">
                <strong>Status:</strong> ${data.execution.success ? '<span style="color: var(--success);">Success</span>' : '<span style="color: var(--error);">Failed</span>'}<br>
                ${data.execution.execution_time ? `<strong>Time:</strong> ${data.execution.execution_time.toFixed(2)}s<br>` : ''}
                ${data.execution.exit_code !== undefined ? `<strong>Exit Code:</strong> ${data.execution.exit_code}<br>` : ''}
            </div>
        `;

        if (data.execution.errors && data.execution.errors.length > 0) {
            html += `
                <div class="alert alert-warning" style="margin-top: 1rem;">
                    <strong><i class="fas fa-exclamation-triangle"></i> Errors:</strong><br>
                    ${data.execution.errors.slice(0, 3).map(err => `• ${err}`).join('<br>')}
                </div>
            `;
        }
    }

    html += `</div>`;

    // Demo mode notice
    if (CONFIG.USE_DEMO_MODE) {
        html += `
            <div class="alert alert-info" style="margin-top: 1rem;">
                <strong><i class="fas fa-info-circle"></i> Demo Mode:</strong>
                This is simulated data. Deploy the backend API for real reproductions.
                <a href="https://github.com/Patrickoo7/Reasearch-Litrature_Review" target="_blank">See deployment guide →</a>
            </div>
        `;
    }

    return html;
}

// Generate analysis results HTML
function generateAnalysisResults(data) {
    let html = `
        <div class="result-card">
            <h3><i class="fas fa-file-alt"></i> ${data.paper?.title || 'Paper Analysis'}</h3>
    `;

    if (data.paper) {
        html += `
            <div class="result-content">
                <strong>Authors:</strong> ${(data.paper.authors || []).slice(0, 5).join(', ')}<br>
            </div>
        `;
    }

    if (data.repositories && data.repositories.length > 0) {
        html += `
            <h4 style="margin-top: 1.5rem;"><i class="fab fa-github"></i> Repositories Found: ${data.repositories.length}</h4>
            <div class="result-content">
        `;
        data.repositories.forEach((repo, idx) => {
            html += `
                <div style="margin-bottom: 1rem; padding: 0.75rem; background: white; border-radius: 8px;">
                    <a href="${repo.url}" target="_blank" style="color: var(--primary); font-weight: 500; text-decoration: none;">
                        ${idx + 1}. ${repo.url}
                    </a><br>
                    ${repo.stars ? `<span style="font-size: 0.875rem; color: var(--text-muted);">⭐ ${repo.stars} stars</span>` : ''}
                    ${repo.language ? `<span style="font-size: 0.875rem; color: var(--text-muted);"> • ${repo.language}</span>` : ''}
                </div>
            `;
        });
        html += `</div>`;
    } else {
        html += `
            <div class="alert alert-warning" style="margin-top: 1rem;">
                <strong><i class="fas fa-exclamation-triangle"></i> No repositories found</strong><br>
                This paper may not have publicly available code.
            </div>
        `;
    }

    html += `</div>`;

    if (CONFIG.USE_DEMO_MODE) {
        html += `
            <div class="alert alert-info" style="margin-top: 1rem;">
                <strong><i class="fas fa-info-circle"></i> Demo Mode:</strong>
                This is simulated data. Deploy the backend API for real analysis.
            </div>
        `;
    }

    return html;
}

// Display error
function displayError(type, message) {
    const resultsEl = document.getElementById(type === 'reproduce' ? 'results' : 'analyze-results');
    const contentEl = document.getElementById(type === 'reproduce' ? 'result-content' : 'analyze-content');

    const html = `
        <div class="result-card">
            <div class="result-header">
                <div class="result-icon error">
                    <i class="fas fa-times-circle"></i>
                </div>
                <div>
                    <div class="result-title">Error</div>
                    <div class="result-content">${message}</div>
                </div>
            </div>
            <div class="alert alert-warning" style="margin-top: 1rem;">
                <strong>Troubleshooting:</strong><br>
                • Make sure the backend API is running<br>
                • Check the API URL in app.js configuration<br>
                • Verify the paper source is valid<br>
                • See <a href="https://github.com/Patrickoo7/Reasearch-Litrature_Review" target="_blank">documentation</a> for help
            </div>
        </div>
    `;

    contentEl.innerHTML = html;
    resultsEl.classList.add('active');
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    console.log('Research Reproducer initialized');
    console.log('Demo mode:', CONFIG.USE_DEMO_MODE);
    console.log('API URL:', CONFIG.API_URL);
});
