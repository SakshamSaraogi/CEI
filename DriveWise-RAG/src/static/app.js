// DriveWise RAG Frontend Application

// State variables
let activeTab = 'chat';
let brands = [];
let allModels = [];
let activeBrand = null;
let activeModel = null;
let chatHistory = [];
let debugData = null;

// DOM Elements
const navButtons = document.querySelectorAll('.nav-btn');
const tabs = document.querySelectorAll('.tab-content');
const screenTitle = document.getElementById('screen-title');
const screenSubtitle = document.getElementById('screen-subtitle');
const modelGrid = document.getElementById('model-grid');
const brochureCount = document.getElementById('brochure-count');
const activeModelBanner = document.getElementById('active-model-banner');
const activeModelName = document.getElementById('active-model-name');
const clearModelFilter = document.getElementById('clear-model-filter');
const chatMessages = document.getElementById('chat-messages');
const chatForm = document.getElementById('chat-form');
const queryInput = document.getElementById('query-input');
const debugSidebar = document.getElementById('debug-sidebar');
const debugPanelContent = document.getElementById('debug-panel-content');
const debugToggleView = document.getElementById('debug-toggle-view');
const settingsMockToggle = document.getElementById('settings-mock-toggle');
const settingsModelName = document.getElementById('settings-model-name');
const ragStatusText = document.getElementById('rag-status-text');
const clearDbBtn = document.getElementById('clear-db-btn');
const citationModal = document.getElementById('citation-modal');
const citationModalBody = document.getElementById('citation-modal-body');
const closeCitationModal = document.getElementById('close-citation-modal');

// Compare tab elements
const compareModel1Select = document.getElementById('compare-model-1');
const compareModel2Select = document.getElementById('compare-model-2');
const runComparisonBtn = document.getElementById('run-comparison-btn');
const compareResultsGrid = document.getElementById('compare-results-grid');

// Chart instances
let chartVolume = null;
let chartConfidence = null;
let chartTopModels = null;

// Initialize App
function initAll() {
    try { initTabs(); } catch (e) { console.error("Error initializing tabs:", e); }
    try { initDebugPanel(); } catch (e) { console.error("Error initializing debug panel:", e); }
    try { initSettings(); } catch (e) { console.error("Error initializing settings:", e); }
    try { loadCatalog(); } catch (e) { console.error("Error loading catalog:", e); }
    try { loadAnalytics(); } catch (e) { console.error("Error loading analytics:", e); }
    try { setupChat(); } catch (e) { console.error("Error setting up chat:", e); }
    try { setupCompare(); } catch (e) { console.error("Error setting up compare:", e); }
    try { initLandingPage(); } catch (e) { console.error("Error initializing landing page:", e); }
    try { init3DModel(); } catch (e) { console.error("Error initializing 3D Model:", e); }
    try { setupMobileConsoleSidebar(); } catch (e) { console.error("Error setting up mobile sidebar:", e); }
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initAll);
} else {
    initAll();
}

// 1. Navigation & Tabs
function initTabs() {
    navButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabName = btn.dataset.tab;
            
            // Toggle nav classes
            navButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            // Toggle tab display
            tabs.forEach(t => t.classList.remove('active'));
            document.getElementById(`tab-${tabName}`).classList.add('active');
            
            activeTab = tabName;
            updateHeader(tabName);
            
            // Reload analytics when entering analytics tab
            if (tabName === 'analytics') {
                loadAnalytics();
            }
        });
    });
}

function updateHeader(tabName) {
    const subtitles = {
        chat: "Explore technical specifications with grounded facts",
        compare: "Inspect engine, dimensions, and capacities side-by-side",
        analytics: "SQLite logging metrics, pipeline latency, and RAG accuracy",
        settings: "Change execution parameters and reset data caches"
    };
    
    const titles = {
        chat: "Chat Assistant",
        compare: "Compare Models",
        analytics: "Analytics Dashboard",
        settings: "System Settings"
    };
    
    screenTitle.textContent = titles[tabName];
    screenSubtitle.textContent = subtitles[tabName];
}

// 2. Load Brochure Catalog (Brands & Models)
async function loadCatalog() {
    try {
        const brandsRes = await fetch('/api/brands');
        brands = await brandsRes.parse ? await brandsRes.parse() : await brandsRes.json();
        
        allModels = [];
        for (const brand of brands) {
            const modelsRes = await fetch(`/api/models?brand=${brand}`);
            const models = await modelsRes.json();
            allModels.push(...models.map(m => ({ ...m, brand })));
        }
        
        brochureCount.textContent = allModels.length;
        renderModelGrid();
        populateCompareSelectors();
    } catch (err) {
        console.error("Failed to load brochure catalog:", err);
        modelGrid.innerHTML = `<div class="grid-loading"><i class="fa-solid fa-circle-exclamation text-danger"></i> Failed to load catalog.</div>`;
    }
}

const brandLogos = {
    'hyundai': '/logos/hyundai.png',
    'maruti-suzuki': '/logos/maruti-suzuki.png',
    'tata': '/logos/tata.png'
};

function renderModelGrid() {
    if (allModels.length === 0) {
        modelGrid.innerHTML = `<div class="grid-loading">No models indexed.</div>`;
        return;
    }
    
    modelGrid.innerHTML = '';
    
    // Group models by brand
    const grouped = {};
    allModels.forEach(model => {
        if (!grouped[model.brand]) {
            grouped[model.brand] = [];
        }
        grouped[model.brand].push(model);
    });
    
    // Render grouped brands
    Object.keys(grouped).forEach(brand => {
        const brandSection = document.createElement('div');
        brandSection.className = 'brand-section';
        
        // Brand Header (Accordion toggler)
        const brandHeader = document.createElement('div');
        brandHeader.className = 'brand-accordion-header';
        
        const logoUrl = brandLogos[brand.toLowerCase()] || '';
        const brandLogoHtml = logoUrl ? `<img src="${logoUrl}" class="brand-logo-img" alt="${brand}">` : '';
        
        brandHeader.innerHTML = `
            <span class="brand-title">
                <i class="fa-solid fa-chevron-right accordion-arrow"></i>
                ${brand.toUpperCase()}
            </span>
            <div class="brand-header-right">
                ${brandLogoHtml}
                <span class="brand-count">${grouped[brand].length}</span>
            </div>
        `;
        
        const brandBody = document.createElement('div');
        brandBody.className = 'brand-accordion-body'; // Collapsed by default
        
        grouped[brand].forEach(model => {
            const row = document.createElement('div');
            row.className = 'model-row';
            row.dataset.brand = model.brand;
            row.dataset.model = model.name;
            
            if (activeBrand === model.brand && activeModel === model.name) {
                row.classList.add('active');
                brandBody.classList.add('active');
                const arrow = brandHeader.querySelector('.accordion-arrow');
                if (arrow) arrow.style.transform = 'rotate(90deg)';
            }
            
            row.innerHTML = `
                <span class="model-row-name"><i class="fa-regular fa-file-pdf"></i> ${model.name.replace(/-/g, ' ').toUpperCase()}</span>
                <span class="model-row-year">${model.year}</span>
            `;
            
            row.addEventListener('click', () => {
                if (activeBrand === model.brand && activeModel === model.name) {
                    setActiveModelFilter(null, null);
                } else {
                    setActiveModelFilter(model.brand, model.name);
                }
            });
            
            brandBody.appendChild(row);
        });
        
        // Toggle Accordion functionality
        brandHeader.addEventListener('click', () => {
            const isActive = brandBody.classList.toggle('active');
            const arrow = brandHeader.querySelector('.accordion-arrow');
            if (arrow) {
                arrow.style.transform = isActive ? 'rotate(90deg)' : 'rotate(0deg)';
            }
        });
        
        brandSection.appendChild(brandHeader);
        brandSection.appendChild(brandBody);
        modelGrid.appendChild(brandSection);
    });
}

function setActiveModelFilter(brand, model) {
    activeBrand = brand;
    activeModel = model;
    
    // Update active visual state in sidebar brand accordions
    const rows = modelGrid.querySelectorAll('.model-row');
    rows.forEach(row => {
        const rowBrand = row.dataset.brand;
        const rowModel = row.dataset.model;
        if (activeBrand === rowBrand && activeModel === rowModel) {
            row.classList.add('active');
        } else {
            row.classList.remove('active');
        }
    });
    
    // Update active model banner above chat
    if (activeModel) {
        activeModelName.textContent = `${activeBrand.toUpperCase()} / ${activeModel.replace(/-/g, ' ').toUpperCase()}`;
        activeModelBanner.classList.remove('hidden');
    } else {
        activeModelBanner.classList.add('hidden');
    }
}

// 3. Settings Interface
async function initSettings() {
    try {
        const res = await fetch('/api/settings');
        const settings = await res.json();
        
        settingsMockToggle.checked = settings.mock_llm;
        settingsModelName.textContent = settings.model_name;
        updateStatusIndicator(settings.mock_llm);
        
        settingsMockToggle.addEventListener('change', async (e) => {
            const val = e.target.checked;
            const putRes = await fetch('/api/settings', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mock_llm: val })
            });
            const data = await putRes.json();
            updateStatusIndicator(data.mock_llm);
        });
        
        clearDbBtn.addEventListener('click', async () => {
            if (confirm("Are you sure you want to delete all query logs? This will reset the analytics charts.")) {
                await fetch('/api/logs/clear', { method: 'POST' });
                alert("Database logs cleared.");
                loadAnalytics();
            }
        });
    } catch (err) {
        console.error("Failed to load settings:", err);
    }
}

function updateStatusIndicator(isMock) {
    if (isMock) {
        ragStatusText.textContent = "Mock Mode Active";
        ragStatusText.parentElement.querySelector('.status-dot').style.backgroundColor = "var(--warning)";
        ragStatusText.parentElement.querySelector('.status-dot').style.boxShadow = "0 0 8px var(--warning)";
    } else {
        ragStatusText.textContent = "Real RAG Active";
        ragStatusText.parentElement.querySelector('.status-dot').style.backgroundColor = "var(--success)";
        ragStatusText.parentElement.querySelector('.status-dot').style.boxShadow = "0 0 8px var(--success)";
    }
}

// 4. Chat Engine Implementation
function setupChat() {
    // Clear filter banner listener
    clearModelFilter.addEventListener('click', () => {
        setActiveModelFilter(null, null);
    });
    
    // Suggested chips prompts
    document.querySelectorAll('.prompt-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            const query = chip.dataset.query;
            queryInput.value = query;
            
            // Auto-select corresponding brochure in sidebar to match demo query context
            if (query.toLowerCase().includes("creta")) {
                setActiveModelFilter("hyundai", "creta");
            } else if (query.toLowerCase().includes("nexon")) {
                if (query.toLowerCase().includes("xuv3xo")) {
                    setActiveModelFilter(null, null);
                } else {
                    setActiveModelFilter("tata", "nexon");
                }
            } else {
                setActiveModelFilter(null, null);
            }
            
            submitUserQuery();
        });
    });
    
    // Input form submit
    chatForm.addEventListener('submit', (e) => {
        e.preventDefault();
        submitUserQuery();
    });
}

async function submitUserQuery() {
    const text = queryInput.value.trim();
    if (!text) return;
    
    // 1. Add User Message
    appendMessage('user', text);
    queryInput.value = '';
    
    // 2. Append Loading Message
    const loadingId = appendMessage('assistant', '<i class="fa-solid fa-circle-notch fa-spin"></i> Driving query request...');
    
    try {
        const payload = {
            query: text,
            history: chatHistory.slice(-6), // Send last 3 turns
            brand: activeBrand,
            model: activeModel
        };
        
        const res = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        
        if (!res.ok) {
            throw new Error(`API returned status ${res.status}`);
        }
        
        const data = await res.json();
        
        // Save to state history
        chatHistory.push({ role: 'user', content: text });
        chatHistory.push({ role: 'assistant', content: data.answer });
        
        // 3. Replace loading message with parsed result
        replaceMessage(loadingId, data);
        
        // Update debug panel with retrieval details
        updateDebugPanel(data);
        
    } catch (err) {
        console.error("Query failed:", err);
        updateMessageText(loadingId, `<span class="text-danger"><i class="fa-solid fa-triangle-exclamation"></i> Error communicating with backend API. ${err.message}</span>`);
    }
}

function appendMessage(role, text) {
    const msgId = 'msg-' + Date.now() + '-' + Math.random().toString(36).substr(2, 5);
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}`;
    msgDiv.id = msgId;
    
    const avatarIcon = role === 'user' ? '<i class="fa-regular fa-user"></i>' : '<i class="fa-solid fa-robot"></i>';
    
    msgDiv.innerHTML = `
        <div class="avatar">${avatarIcon}</div>
        <div class="message-bubble">${text}</div>
    `;
    
    chatMessages.appendChild(msgDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return msgId;
}

function updateMessageText(msgId, text) {
    const bubble = document.querySelector(`#${msgId} .message-bubble`);
    if (bubble) {
        bubble.innerHTML = text;
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
}

function replaceMessage(msgId, data) {
    const bubble = document.querySelector(`#${msgId} .message-bubble`);
    if (!bubble) return;
    
    // Clear content
    bubble.innerHTML = '';
    
    // Create Header (Badge + Latency)
    const header = document.createElement('div');
    header.className = 'message-bubble-header';
    header.innerHTML = `
        <span class="confidence-badge ${data.confidence}">
            <i class="fa-solid fa-circle-check"></i> ${data.confidence.replace(/_/g, ' ')} confidence
        </span>
        <span class="latency-label text-muted" style="font-size: 0.75rem;">
            <i class="fa-regular fa-clock"></i> ${data.latency_ms.toFixed(0)}ms
        </span>
    `;
    bubble.appendChild(header);
    
    // Create Answer Text
    const textDiv = document.createElement('div');
    textDiv.className = 'message-text';
    textDiv.innerHTML = data.answer.replace(/\n/g, '<br>');
    bubble.appendChild(textDiv);
    
    // Add Variant Disambiguation box if present
    if (data.variant_disambiguation && data.variant_disambiguation.length > 0) {
        const vContainer = document.createElement('div');
        vContainer.className = 'variant-disambig-container';
        
        let rowsHtml = '';
        data.variant_disambiguation.forEach(v => {
            rowsHtml += `
                <div class="variant-row">
                    <span class="variant-badge">${v.variant}</span>
                    <span class="variant-val">${v.value}</span>
                </div>
            `;
        });
        
        vContainer.innerHTML = `
            <div class="variant-disambig-header"><i class="fa-solid fa-layer-group"></i> Variant Disambiguation Specs</div>
            ${rowsHtml}
        `;
        bubble.appendChild(vContainer);
    }
    
    // Create Citations Chips
    if (data.citations && data.citations.length > 0) {
        const wrapper = document.createElement('div');
        wrapper.className = 'citations-wrapper';
        wrapper.innerHTML = `<span class="citation-title">Citations:</span>`;
        
        data.citations.forEach((c, idx) => {
            const chip = document.createElement('span');
            chip.className = 'citation-chip';
            chip.textContent = `[${c.source_file.replace(/_2026\.pdf|_brochure\.pdf/g, '')} p.${c.page_number}]`;
            
            // Modal click viewer
            chip.addEventListener('click', () => {
                showCitationDetails(c);
            });
            wrapper.appendChild(chip);
        });
        
        bubble.appendChild(wrapper);
    }
    
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function showCitationDetails(citation) {
    citationModalBody.innerHTML = `
        <div class="modal-field">
            <span class="modal-label">Chunk ID</span>
            <span class="modal-val debug-text-box" style="font-family: monospace;">${citation.chunk_id}</span>
        </div>
        <div class="modal-field">
            <span class="modal-label">Brochure Source File</span>
            <span class="modal-val"><i class="fa-regular fa-file-pdf text-danger"></i> ${citation.source_file}</span>
        </div>
        <div class="modal-field">
            <span class="modal-label">Page Number</span>
            <span class="modal-val">${citation.page_number}</span>
        </div>
        <div class="modal-field">
            <span class="modal-label">Section Type</span>
            <span class="modal-val"><span class="variant-badge">${citation.section_type}</span></span>
        </div>
        <div class="modal-field">
            <span class="modal-label">Brochure Year</span>
            <span class="modal-val">${citation.document_version}</span>
        </div>
    `;
    citationModal.classList.remove('hidden');
}

closeCitationModal.addEventListener('click', () => {
    citationModal.classList.add('hidden');
});

// 5. Debug Panel Drawer
function initDebugPanel() {
    debugToggleView.addEventListener('click', () => {
        debugSidebar.classList.toggle('collapsed');
        // Toggle icon direction
        const icon = debugToggleView.querySelector('i');
        if (debugSidebar.classList.contains('collapsed')) {
            icon.className = 'fa-solid fa-chevron-left';
        } else {
            icon.className = 'fa-solid fa-chevron-right';
        }
    });
}

function updateDebugPanel(data) {
    debugPanelContent.innerHTML = '';
    
    // Section 1: Resolved Standalone Query
    const s1 = document.createElement('div');
    s1.className = 'debug-section';
    s1.innerHTML = `
        <h4>Standalone Target Query</h4>
        <div class="debug-text-box">${data.rewritten_query}</div>
    `;
    debugPanelContent.appendChild(s1);
    
    // Section 2: Metadata Extraction Target
    const s2 = document.createElement('div');
    s2.className = 'debug-section';
    s2.innerHTML = `
        <h4>Metadata Targets</h4>
        <div class="debug-text-box" style="display: flex; gap: 12px;">
            <span>Brand: <strong>${activeBrand || 'All (Global)'}</strong></span>
            <span>Model: <strong>${activeModel || 'All (Global)'}</strong></span>
        </div>
    `;
    debugPanelContent.appendChild(s2);
    
    // Section 3: Extracted Keywords
    const s3 = document.createElement('div');
    s3.className = 'debug-section';
    s3.innerHTML = `
        <h4>Attributes/Search Keywords</h4>
        <div class="debug-keywords">
            <span class="keyword-badge">engine</span>
            <span class="keyword-badge">specifications</span>
            <span class="keyword-badge">dimensions</span>
        </div>
    `;
    debugPanelContent.appendChild(s3);
    
    // Section 4: Retrieved Context Chunks
    const s4 = document.createElement('div');
    s4.className = 'debug-section';
    s4.innerHTML = `<h4>Retrieved Brochure Chunks (${data.citations.length})</h4>`;
    
    if (data.citations.length === 0) {
        s4.innerHTML += `<div class="debug-empty-state" style="padding: 16px 0;">No chunks cited.</div>`;
    } else {
        data.citations.forEach(c => {
            const chunkCard = document.createElement('div');
            chunkCard.className = 'debug-chunk-card';
            chunkCard.innerHTML = `
                <div class="debug-chunk-header">
                    <span>${c.chunk_id}</span>
                    <span class="chunk-score">Page ${c.page_number}</span>
                </div>
                <div class="chunk-text-val">Source: ${c.source_file}\nSection: ${c.section_type}\nVersion: ${c.document_version}</div>
            `;
            s4.appendChild(chunkCard);
        });
    }
    debugPanelContent.appendChild(s4);
}

// 6. Compare Mode Implementation
function populateCompareSelectors() {
    compareModel1Select.innerHTML = '<option value="">Select Car Model...</option>';
    compareModel2Select.innerHTML = '<option value="">Select Car Model...</option>';
    
    allModels.forEach(m => {
        const option1 = document.createElement('option');
        option1.value = `${m.brand}/${m.name}`;
        option1.textContent = `${m.brand.toUpperCase()} - ${m.name.replace(/-/g, ' ').toUpperCase()}`;
        compareModel1Select.appendChild(option1);
        
        const option2 = option1.cloneNode(true);
        compareModel2Select.appendChild(option2);
    });
}

function setupCompare() {
    runComparisonBtn.addEventListener('click', async () => {
        const target1 = compareModel1Select.value;
        const target2 = compareModel2Select.value;
        
        if (!target1 || !target2) {
            alert("Please select both Model 1 and Model 2 to run comparison.");
            return;
        }
        
        compareResultsGrid.innerHTML = `<div class="grid-loading"><i class="fa-solid fa-circle-notch fa-spin"></i> Retrieving comparison specs...</div>`;
        
        try {
            // Split values
            const [b1, m1] = target1.split('/');
            const [b2, m2] = target2.split('/');
            
            // Call API chat with a comparison format query asking for the three specific fields
            const payload = {
                query: `Compare the engine displacement in cc, turbocharger support details, and boot space in litres of the ${m1} and ${m2}.`,
                brand: b1,
                model: m1
            };
            
            const res = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            
            renderComparisonResults(m1, m2, data);
        } catch (err) {
            compareResultsGrid.innerHTML = `<div class="grid-loading text-danger">Failed to run comparison. ${err.message}</div>`;
        }
    });
}

function renderComparisonResults(model1, model2, data) {
    compareResultsGrid.innerHTML = '';
    
    // Create Table element
    const table = document.createElement('table');
    table.className = 'specs-comparison-table';
    
    // Parse specs for each model from the comparison JSON answers
    let spec1 = { engine: 'No data found', turbo: 'No data found', boot: 'No data found' };
    let spec2 = { engine: 'No data found', turbo: 'No data found', boot: 'No data found' };
    
    if (data.comparison && data.comparison.per_model_answers) {
        const ans1 = data.comparison.per_model_answers[model1];
        const ans2 = data.comparison.per_model_answers[model2];
        
        if (ans1) {
            try {
                spec1 = JSON.parse(ans1.trim());
            } catch (e) {
                // Graceful fallback to raw string
                spec1.engine = ans1;
            }
        }
        if (ans2) {
            try {
                spec2 = JSON.parse(ans2.trim());
            } catch (e) {
                spec2.engine = ans2;
            }
        }
    }
    
    const confidenceClass = data.confidence ? data.confidence.toLowerCase() : 'low';
    const confidenceText = data.confidence ? data.confidence.toUpperCase() + ' Confidence' : 'Not Found';
    
    // Headers
    table.innerHTML = `
        <thead>
            <tr>
                <th>Specification Feature</th>
                <th>${model1.replace(/-/g, ' ').toUpperCase()}</th>
                <th>${model2.replace(/-/g, ' ').toUpperCase()}</th>
            </tr>
        </thead>
        <tbody>
            <tr class="spec-category-row">
                <td colspan="3">Engine & Powertrain Details</td>
            </tr>
            <tr>
                <td>Engine Displacement</td>
                <td><div class="spec-val-box">${spec1.engine || 'No data found'}</div></td>
                <td><div class="spec-val-box">${spec2.engine || 'No data found'}</div></td>
            </tr>
            <tr>
                <td>Turbocharger Support</td>
                <td><div class="spec-val-box">${spec1.turbo || 'No data found'}</div></td>
                <td><div class="spec-val-box">${spec2.turbo || 'No data found'}</div></td>
            </tr>
            <tr class="spec-category-row">
                <td colspan="3">Capacities & Grounding Facts</td>
            </tr>
            <tr>
                <td>Boot Luggage Space</td>
                <td><div class="spec-val-box">${spec1.boot || 'No data found'}</div></td>
                <td><div class="spec-val-box">${spec2.boot || 'No data found'}</div></td>
            </tr>
            <tr>
                <td>RAG Verification Grounding</td>
                <td><span class="confidence-badge ${confidenceClass}" style="display:inline-flex;">${confidenceText}</span></td>
                <td><span class="confidence-badge ${confidenceClass}" style="display:inline-flex;">${confidenceText}</span></td>
            </tr>
        </tbody>
    `;
    
    compareResultsGrid.appendChild(table);
}

// 7. Analytics Dashboard Charts (ChartJS)
async function loadAnalytics() {
    try {
        const res = await fetch('/api/analytics');
        const stats = await res.json();
        
        // Update stats top cards
        document.getElementById('analytics-total-queries').textContent = stats.total_queries;
        document.getElementById('analytics-avg-latency').textContent = `${stats.avg_latency.toFixed(0)} ms`;
        
        // Calculate success/high confidence rate
        const total = stats.total_queries || 1;
        const high = stats.confidence_breakdown.high || 0;
        const med = stats.confidence_breakdown.medium || 0;
        const rate = ((high + med) / total) * 100.0;
        document.getElementById('analytics-success-rate').textContent = `${rate.toFixed(0)}%`;
        
        // Render Chart 1: Volume over time
        renderVolumeChart(stats.daily_volume);
        
        // Render Chart 2: Confidence breakdown Donut
        renderConfidenceChart(stats.confidence_breakdown);
        
        // Render Chart 3: Top brochure models
        renderTopModelsChart(stats.top_models);
        
    } catch (err) {
        console.error("Failed to load analytics dashboard stats:", err);
    }
}

function renderVolumeChart(data) {
    if (chartVolume) chartVolume.destroy();
    
    const ctx = document.getElementById('chart-volume').getContext('2d');
    
    // Empty case
    if (!data || data.length === 0) {
        data = [{ date: 'Today', count: 0, avg_lat: 0 }];
    }
    
    const labels = data.map(d => d.date);
    const counts = data.map(d => d.count);
    
    chartVolume = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Query Volume',
                data: counts,
                borderColor: '#6366f1',
                backgroundColor: 'rgba(99, 102, 241, 0.1)',
                fill: true,
                tension: 0.3,
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#9ca3af' } },
                y: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#9ca3af' } }
            }
        }
    });
}

function renderConfidenceChart(breakdown) {
    if (chartConfidence) chartConfidence.destroy();
    
    const ctx = document.getElementById('chart-confidence').getContext('2d');
    
    const labels = ['High', 'Medium', 'Low', 'Not Found'];
    const dataVals = [
        breakdown.high || 0,
        breakdown.medium || 0,
        breakdown.low || 0,
        breakdown.not_found || 0
    ];
    
    chartConfidence = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: dataVals,
                backgroundColor: ['#10b981', '#f59e0b', '#ef4444', '#4b5563'],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { color: '#9ca3af', font: { family: 'Inter' } }
                }
            }
        }
    });
}

function renderTopModelsChart(models) {
    if (chartTopModels) chartTopModels.destroy();
    
    const ctx = document.getElementById('chart-top-models').getContext('2d');
    
    if (!models || models.length === 0) {
        models = [{ model: 'None', count: 0 }];
    }
    
    const labels = models.map(m => m.model.replace(/-/g, ' ').toUpperCase());
    const counts = models.map(m => m.count);
    
    chartTopModels = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                data: counts,
                backgroundColor: 'rgba(99, 102, 241, 0.65)',
                borderRadius: 6
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#9ca3af' } },
                y: { grid: { display: false }, ticks: { color: '#9ca3af' } }
            }
        }
    });
}

// --- EventFlow Premium Dark-Mode Landing Page Controller ---
function initLandingPage() {
    const landingPage = document.getElementById('landing-page');
    const appContainer = document.getElementById('app-container');
    const hamburgerBtn = document.getElementById('hamburger-btn');
    const mobileOverlay = document.getElementById('mobile-overlay');
    const landingSearchForm = document.getElementById('landing-search-form');
    const landingSearchInput = document.getElementById('landing-search-input');
    
    // 1. Hamburger menu toggle
    if (hamburgerBtn && mobileOverlay) {
        const toggleMobileMenu = () => {
            hamburgerBtn.classList.toggle('active');
            mobileOverlay.classList.toggle('active');
        };
        hamburgerBtn.addEventListener('click', toggleMobileMenu);
    }
    
    // 2. Transition to RAG Explorer Area
    const enterRAG = (tabName = 'chat', autoQuery = null) => {
        // Close mobile overlay if open
        if (hamburgerBtn) hamburgerBtn.classList.remove('active');
        if (mobileOverlay) mobileOverlay.classList.remove('active');
        
        // Fade out landing
        if (landingPage) landingPage.classList.add('fade-out');
        
        setTimeout(() => {
            if (landingPage) landingPage.style.display = 'none';
            if (appContainer) appContainer.style.display = 'flex';
            
            // Switch to requested tab
            if (tabName) {
                const tabBtn = document.querySelector(`.nav-btn[data-tab="${tabName}"]`);
                if (tabBtn) tabBtn.click();
            }
            
            // Auto query injection
            if (autoQuery) {
                if (autoQuery.includes("Compare") || autoQuery.includes("displacement")) {
                    // Switch to compare tab
                    const compareBtn = document.querySelector('.nav-btn[data-tab="compare"]');
                    if (compareBtn) compareBtn.click();
                    
                    // Set Nexon and Baleno selections
                    compareModel1Select.value = 'tata/nexon';
                    compareModel2Select.value = 'maruti-suzuki/baleno';
                    
                    // Trigger comparison
                    runComparisonBtn.click();
                } else {
                    // Chat query
                    if (autoQuery.toLowerCase().includes("creta")) {
                        setActiveModelFilter("hyundai", "creta");
                    } else if (autoQuery.toLowerCase().includes("nexon")) {
                        setActiveModelFilter("tata", "nexon");
                    }
                    queryInput.value = autoQuery;
                    submitUserQuery();
                }
            }
        }, 500);
    };
    
    // Wire up Enter buttons
    const enterCta = document.getElementById('landing-enter-cta');
    if (enterCta) {
        enterCta.addEventListener('click', () => enterRAG('chat'));
    }
    
    const mobileEnterCta = document.getElementById('mobile-enter-cta');
    if (mobileEnterCta) {
        mobileEnterCta.addEventListener('click', () => enterRAG('chat'));
    }
    
    const heroEnterBtn = document.getElementById('hero-enter-btn');
    if (heroEnterBtn) {
        heroEnterBtn.addEventListener('click', () => enterRAG('chat'));
    }
    
    const landingLogo = document.getElementById('landing-logo');
    if (landingLogo) {
        landingLogo.addEventListener('click', () => {
            // Optional click handler
        });
    }
    
    // Navigation link buttons
    document.querySelectorAll('.nav-link-btn, .mobile-link:not(.mobile-cta)').forEach(link => {
        link.addEventListener('click', () => {
            const tab = link.dataset.tab;
            enterRAG(tab);
        });
    });
    
    // Suggested chips prompts
    document.querySelectorAll('.suggest-chip').forEach(btn => {
        btn.addEventListener('click', () => {
            const tab = btn.dataset.tab;
            const query = btn.dataset.query;
            if (tab) {
                enterRAG(tab);
            } else if (query) {
                enterRAG('chat', query);
            }
        });
    });
    
    // Landing Search submit
    if (landingSearchForm) {
        landingSearchForm.addEventListener('submit', (e) => {
            e.preventDefault();
            const text = landingSearchInput ? landingSearchInput.value.trim() : "";
            if (text) {
                if (landingSearchInput) landingSearchInput.value = '';
                enterRAG('chat', text);
            }
        });
    }
    
    // Copy email button defensive check
    const copyEmailBtn = document.getElementById('copy-email-btn');
    if (copyEmailBtn) {
        copyEmailBtn.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            const email = "support@drivewise.ai";
            navigator.clipboard.writeText(email).then(() => {
                const btnSpan = copyEmailBtn.querySelector('span');
                const originalText = btnSpan.textContent;
                btnSpan.textContent = "Copied!";
                setTimeout(() => { btnSpan.textContent = originalText; }, 2000);
            });
        });
    }
    
    // Back button: Click logo on main dashboard sidebar to return to landing
    const brandHeader = document.querySelector('.brand-header');
    if (brandHeader) {
        brandHeader.style.cursor = 'pointer';
        brandHeader.addEventListener('click', () => {
            if (appContainer) appContainer.style.display = 'none';
            if (landingPage) {
                landingPage.style.display = 'flex';
                landingPage.classList.remove('fade-out');
            }
        });
    }
}

// --- Interactive 3D Car Model Viewer (Three.js) ---
function init3DModel() {
    const container = document.getElementById('car-3d-container');
    if (!container) return;
    
    // Fallback dimension chain (prevents startup 0px / 300x150 default canvas bugs)
    const parent = container.parentElement;
    let width = container.clientWidth || (parent ? parent.clientWidth : 0) || Math.floor(window.innerWidth * 0.5) || 800;
    let height = container.clientHeight || (parent ? parent.clientHeight : 0) || 600;
    
    // 1. Scene Setup
    const scene = new THREE.Scene();
    
    // 2. Camera Setup (Three-quarter front view, looking directly at origin)
    const camera = new THREE.PerspectiveCamera(40, width / height, 0.1, 100);
    camera.position.set(4.0, 1.1, -5.2); // Frame the car massive and centered inside canvas
    camera.lookAt(0, -0.15, 0);
    
    // 3. Renderer Setup
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.5; // Brighter natural look
    
    // Force canvas styles to stretch to fill the outer wrapper space
    renderer.domElement.style.width = '100%';
    renderer.domElement.style.height = '100%';
    renderer.domElement.style.display = 'block';
    
    container.innerHTML = ''; // Clear fallback artifacts
    container.appendChild(renderer.domElement);
    
    // 4. Photorealistic Environment Mapping (PMREM Generator)
    try {
        const pmremGenerator = new THREE.PMREMGenerator(renderer);
        pmremGenerator.compileEquirectangularShader();
        
        const envScene = new THREE.Scene();
        
        // White overhead softbox light reflection
        const topGeom = new THREE.BoxGeometry(20, 1, 20);
        const topMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
        const topMesh = new THREE.Mesh(topGeom, topMat);
        topMesh.position.set(0, 8, 0);
        envScene.add(topMesh);
        
        // Left blue-violet reflection panel
        const leftGeom = new THREE.BoxGeometry(1, 20, 20);
        const leftMat = new THREE.MeshBasicMaterial({ color: 0x818cf8 });
        const leftMesh = new THREE.Mesh(leftGeom, leftMat);
        leftMesh.position.set(-8, 0, 0);
        envScene.add(leftMesh);
        
        // Right warm reflection panel
        const rightGeom = new THREE.BoxGeometry(1, 20, 20);
        const rightMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
        const rightMesh = new THREE.Mesh(rightGeom, rightMat);
        rightMesh.position.set(8, 0, 0);
        envScene.add(rightMesh);
        
        const renderTarget = pmremGenerator.fromScene(envScene);
        scene.environment = renderTarget.texture;
        pmremGenerator.dispose();
    } catch (err) {
        console.warn("Failed to generate studio environment reflection map:", err);
    }
    
    // 5. Balanced Lighting (Key, Fill, Rim & Ambient)
    const hemisphereLight = new THREE.HemisphereLight(0xffffff, 0x333333, 1.5);
    scene.add(hemisphereLight);
    
    // Key Light (Front Right)
    const keyLight = new THREE.DirectionalLight(0xffffff, 2.0);
    keyLight.position.set(5, 7, 5);
    keyLight.castShadow = true;
    scene.add(keyLight);
    
    // Fill Light (Front Left)
    const fillLight = new THREE.DirectionalLight(0xffffff, 1.2);
    fillLight.position.set(-5, 5, 5);
    scene.add(fillLight);
    
    // Rim Light (Back Top)
    const rimLight = new THREE.DirectionalLight(0xffffff, 1.6);
    rimLight.position.set(0, 7, -5);
    scene.add(rimLight);
    
    // Ambient Light
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);
    
    // 6. Load GLTF/GLB Model
    const loader = new THREE.GLTFLoader();
    let carModel = null;
    
    loader.load(
        '/model/car.glb',
        (gltf) => {
            carModel = gltf.scene;
            
            // Adjust materials & textures for reflection support
            carModel.traverse((node) => {
                if (node.isMesh) {
                    node.castShadow = true;
                    node.receiveShadow = true;
                    if (node.material) {
                        node.material.envMapIntensity = 2.0; // High intensity for reflections
                        node.material.needsUpdate = true;
                    }
                }
            });
            
            // Add to scene first so matrices are updated correctly
            scene.add(carModel);
            carModel.updateMatrixWorld(true);
            
            // Calculate bounding box using standard setFromObject on updated matrices
            const box = new THREE.Box3().setFromObject(carModel);
            const size = box.getSize(new THREE.Vector3());
            const maxDim = Math.max(size.x, size.y, size.z);
            
            if (maxDim > 0) {
                const scaleFactor = 6.0 / maxDim; // Normalize car to be MASSIVE and cover the right side area properly
                carModel.scale.set(scaleFactor, scaleFactor, scaleFactor);
            }
            
            // Center the model relative to scaled bounds
            carModel.updateMatrixWorld(true);
            const scaledBox = new THREE.Box3().setFromObject(carModel);
            const center = scaledBox.getCenter(new THREE.Vector3());
            carModel.position.sub(center);
            carModel.position.y = -0.3; // Sit centered vertically in canvas
        },
        undefined,
        (error) => {
            console.warn("Failed to load /model/car.glb, loading torus fallback:", error);
            
            // Glassy Torus Fallback (large & bold)
            const geometry = new THREE.TorusGeometry(1.4, 0.5, 32, 100);
            const material = new THREE.MeshPhysicalMaterial({
                color: 0x6366f1,
                roughness: 0.05,
                metalness: 0.9,
                transmission: 0.4,
                ior: 1.6,
                thickness: 0.8,
                clearcoat: 1.0
            });
            carModel = new THREE.Mesh(geometry, material);
            carModel.rotation.x = Math.PI / 2 - 0.2;
            scene.add(carModel);
        }
    );
    
    // 7. Animation loop
    const animate = () => {
        requestAnimationFrame(animate);
        
        // Auto-rotate model slowly
        if (carModel) {
            carModel.rotation.y += 0.0035;
        }
        
        renderer.render(scene, camera);
    };
    animate();
    
    // 8. Handle resize events dynamically
    window.addEventListener('resize', () => {
        const w = container.clientWidth || (parent ? parent.clientWidth : 0) || Math.floor(window.innerWidth * 0.5) || 800;
        const h = container.clientHeight || (parent ? parent.clientHeight : 0) || 600;
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        renderer.setSize(w, h);
    });
}

// --- Mobile Console Sidebar Drawer Toggle ---
function setupMobileConsoleSidebar() {
    const toggleBtn = document.getElementById('mobile-sidebar-toggle');
    const sidebar = document.querySelector('.sidebar');
    const backdrop = document.getElementById('sidebar-backdrop');
    
    if (!toggleBtn || !sidebar || !backdrop) return;
    
    const openSidebar = () => {
        sidebar.classList.add('active');
        backdrop.classList.add('active');
    };
    
    const closeSidebar = () => {
        sidebar.classList.remove('active');
        backdrop.classList.remove('active');
    };
    
    toggleBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        if (sidebar.classList.contains('active')) {
            closeSidebar();
        } else {
            openSidebar();
        }
    });
    
    backdrop.addEventListener('click', closeSidebar);
    
    // Close sidebar drawer on clicking any car model selection or navigation tab
    document.addEventListener('click', (e) => {
        if (e.target.closest('.model-row') || e.target.closest('.nav-btn')) {
            closeSidebar();
        }
    });
}
