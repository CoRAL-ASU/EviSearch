// Global state
let extractionResults = {};
let currentPdfInfo = null;
let availableColumns = [];

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    initializeUpload();
    initializeQueryMethods();
    initializeForms();
    loadAvailableColumns();
});

// ==================== PDF Upload ====================

function initializeUpload() {
    const uploadArea = document.getElementById('upload-area');
    const pdfInput = document.getElementById('pdf-input');
    const browseBtn = document.getElementById('browse-btn');
    
    // Click to browse
    browseBtn.addEventListener('click', () => pdfInput.click());
    uploadArea.addEventListener('click', () => pdfInput.click());
    
    // File selection
    pdfInput.addEventListener('change', handleFileSelect);
    
    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

async function handleFile(file) {
    if (!file.name.toLowerCase().endsWith('.pdf')) {
        showError('Please select a PDF file');
        return;
    }
    
    showLoading();
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            currentPdfInfo = result;
            showPdfInfo(result.filename);
            showQuerySection();
        } else {
            showError(result.error || 'Upload failed');
        }
    } catch (error) {
        showError('Upload failed: ' + error.message);
    } finally {
        hideLoading();
    }
}

function showPdfInfo(filename) {
    document.getElementById('pdf-name').textContent = filename;
    document.getElementById('pdf-info').style.display = 'block';
}

function showQuerySection() {
    document.getElementById('query-section').style.display = 'block';
    document.getElementById('query-section').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ==================== Query Methods ====================

function initializeQueryMethods() {
    document.getElementById('method-single').addEventListener('click', () => showForm('single'));
    document.getElementById('method-all').addEventListener('click', () => showForm('all'));
    document.getElementById('method-csv').addEventListener('click', () => showForm('csv'));
}

function showForm(type) {
    // Hide method selection
    document.getElementById('query-section').style.display = 'none';
    
    // Hide all forms
    document.querySelectorAll('.query-form').forEach(form => {
        form.style.display = 'none';
    });
    
    // Show selected form
    const formId = type + '-form';
    const form = document.getElementById(formId);
    form.style.display = 'block';
    form.scrollIntoView({ behavior: 'smooth', block: 'start' });
    
    // If showing the all-form, load available documents
    if (type === 'all') {
        loadAvailableDocuments();
    }
}

function backToMethods() {
    document.querySelectorAll('.query-form').forEach(form => {
        form.style.display = 'none';
    });
    document.getElementById('query-section').style.display = 'block';
    document.getElementById('query-section').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ==================== Load Available Columns ====================

async function loadAvailableColumns() {
    try {
        const response = await fetch('/api/columns');
        const result = await response.json();
        
        if (result.success) {
            availableColumns = result.columns;
            populateColumnSelect();
        }
    } catch (error) {
        console.error('Failed to load columns:', error);
    }
}

function populateColumnSelect() {
    const select = document.getElementById('column-select');
    
    availableColumns.forEach(col => {
        const option = document.createElement('option');
        option.value = col.column_name;
        option.textContent = `${col.column_name} (${col.label})`;
        option.dataset.definition = col.definition;
        select.appendChild(option);
    });
    
    select.addEventListener('change', (e) => {
        const selectedOption = e.target.selectedOptions[0];
        if (selectedOption && selectedOption.dataset.definition) {
            document.getElementById('custom-definition').value = selectedOption.dataset.definition;
        }
    });
}

// ==================== Single Column Extraction ====================

function initializeForms() {
    // Single column extraction
    document.getElementById('extract-single-btn').addEventListener('click', extractSingleColumn);
    
    // CSV upload
    const csvInput = document.getElementById('csv-input');
    const browseCsvBtn = document.getElementById('browse-csv-btn');
    const extractCsvBtn = document.getElementById('extract-csv-btn');
    
    browseCsvBtn.addEventListener('click', () => csvInput.click());
    
    csvInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            document.getElementById('csv-name').textContent = e.target.files[0].name;
            extractCsvBtn.disabled = false;
        }
    });
    
    extractCsvBtn.addEventListener('click', extractFromCsv);
    
    // Document selection for viewing extractions
    const documentSelect = document.getElementById('document-select');
    const loadExtractionBtn = document.getElementById('load-extraction-btn');
    
    documentSelect.addEventListener('change', (e) => {
        const docInfo = document.getElementById('document-info');
        if (e.target.value) {
            loadExtractionBtn.disabled = false;
            const selectedOption = e.target.selectedOptions[0];
            const model = selectedOption.dataset.model;
            docInfo.textContent = `Model: ${model}`;
            docInfo.style.display = 'block';
        } else {
            loadExtractionBtn.disabled = true;
            docInfo.style.display = 'none';
        }
    });
    
    loadExtractionBtn.addEventListener('click', loadExtractionData);
    
    // Export buttons
    document.getElementById('export-json-btn').addEventListener('click', () => exportResults('json'));
    document.getElementById('export-csv-btn').addEventListener('click', () => exportResults('csv'));
    
    // Toggle view button
    document.getElementById('toggle-view-btn').addEventListener('click', toggleResultsView);
}

// ==================== Toggle Results View ====================

let currentView = 'cards'; // 'cards' or 'table'

function toggleResultsView() {
    const cardsView = document.getElementById('results-container');
    const tableView = document.getElementById('results-table-view');
    const toggleBtn = document.getElementById('toggle-view-btn');
    
    if (currentView === 'cards') {
        cardsView.style.display = 'none';
        tableView.style.display = 'block';
        toggleBtn.textContent = '📋 Card View';
        currentView = 'table';
    } else {
        cardsView.style.display = 'block';
        tableView.style.display = 'none';
        toggleBtn.textContent = '📊 Table View';
        currentView = 'cards';
    }
}

async function extractSingleColumn() {
    const columnSelect = document.getElementById('column-select');
    const customColumn = document.getElementById('custom-column').value.trim();
    const customDefinition = document.getElementById('custom-definition').value.trim();
    
    let columnName = customColumn || columnSelect.value;
    let definition = customDefinition;
    
    if (!columnName) {
        showError('Please select or enter a column name');
        return;
    }
    
    // If using predefined column, get its definition
    if (!customColumn && columnSelect.value) {
        const selectedOption = columnSelect.selectedOptions[0];
        definition = selectedOption.dataset.definition;
    }
    
    const btn = document.getElementById('extract-single-btn');
    toggleButtonLoading(btn, true);
    
    try {
        const response = await fetch('/api/extract/single', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                column_name: columnName,
                definition: definition || undefined
            })
        });
        
        const result = await response.json();
        
        if (result.success) {
            extractionResults = {};
            extractionResults[result.column] = {
                value: result.value,
                page_number: result.page_number,
                modality: result.modality,
                evidence: result.evidence,
                definition: result.definition
            };
            displayResults();
        } else {
            showError(result.error || 'Extraction failed');
        }
    } catch (error) {
        showError('Extraction failed: ' + error.message);
    } finally {
        toggleButtonLoading(btn, false);
    }
}

// ==================== CSV Extraction ====================

async function extractFromCsv() {
    const csvInput = document.getElementById('csv-input');
    
    if (csvInput.files.length === 0) {
        showError('Please select a CSV file');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', csvInput.files[0]);
    
    const btn = document.getElementById('extract-csv-btn');
    toggleButtonLoading(btn, true);
    
    try {
        const response = await fetch('/api/extract/csv', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            extractionResults = result.results;
            displayResults();
            
            if (result.errors && result.errors.length > 0) {
                console.warn('Extraction warnings:', result.errors);
            }
        } else {
            showError(result.error || 'CSV extraction failed');
        }
    } catch (error) {
        showError('CSV extraction failed: ' + error.message);
    } finally {
        toggleButtonLoading(btn, false);
    }
}

// ==================== Load Available Documents ====================

async function loadAvailableDocuments() {
    const documentSelect = document.getElementById('document-select');
    
    try {
        const response = await fetch('/api/documents/available');
        const result = await response.json();
        
        if (result.success) {
            // Clear existing options
            documentSelect.innerHTML = '<option value="">-- Select a document --</option>';
            
            // Add document options
            result.documents.forEach(doc => {
                const option = document.createElement('option');
                option.value = doc.id;
                option.textContent = doc.name;
                option.dataset.model = doc.model;
                documentSelect.appendChild(option);
            });
            
            if (result.documents.length === 0) {
                documentSelect.innerHTML = '<option value="">-- No documents available --</option>';
            }
        } else {
            documentSelect.innerHTML = '<option value="">-- Error loading documents --</option>';
            console.error('Failed to load documents:', result.error);
        }
    } catch (error) {
        documentSelect.innerHTML = '<option value="">-- Error loading documents --</option>';
        console.error('Error loading documents:', error);
    }
}

// ==================== Load Extraction Data ====================

async function loadExtractionData() {
    const documentSelect = document.getElementById('document-select');
    const docId = documentSelect.value;
    
    if (!docId) {
        showError('Please select a document');
        return;
    }
    
    const btn = document.getElementById('load-extraction-btn');
    toggleButtonLoading(btn, true);
    
    try {
        const response = await fetch(`/api/documents/${encodeURIComponent(docId)}/extraction`);
        const result = await response.json();
        
        if (result.success) {
            extractionResults = result.results;
            displayResults(result.document_id);
            
            // Show summary if available
            if (result.summary && result.summary.overall) {
                console.log('Extraction Quality:', result.summary.overall);
            }
        } else {
            showError(result.error || 'Failed to load extraction data');
        }
    } catch (error) {
        showError('Failed to load extraction: ' + error.message);
    } finally {
        toggleButtonLoading(btn, false);
    }
}

// ==================== Display Results ====================

function displayResults(documentName = null) {
    const resultsSection = document.getElementById('results-section');
    const resultsContainer = document.getElementById('results-container');
    const resultsTableBody = document.getElementById('results-table-body');
    const resultsStats = document.getElementById('results-stats');
    const toggleBtn = document.getElementById('toggle-view-btn');
    
    // Clear previous results
    resultsContainer.innerHTML = '';
    resultsTableBody.innerHTML = '';
    resultsStats.innerHTML = '';
    
    const totalColumns = Object.keys(extractionResults).length;
    const foundValues = Object.values(extractionResults).filter(r => 
        r.value && r.value !== 'not found' && r.value !== 'Not reported'
    ).length;
    
    // Add document name if provided
    let documentInfo = '';
    if (documentName) {
        documentInfo = `<div class="stat-item" style="grid-column: 1 / -1;">
            <div class="stat-label">Document</div>
            <div class="stat-value" style="font-size: 1.2rem; color: var(--text-primary);">${escapeHtml(documentName)}</div>
        </div>`;
    }
    
    // Display stats
    resultsStats.innerHTML = `
        ${documentInfo}
        <div class="stat-item">
            <div class="stat-value">${totalColumns}</div>
            <div class="stat-label">Total Columns</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">${foundValues}</div>
            <div class="stat-label">Values Found</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">${((foundValues / totalColumns) * 100).toFixed(1)}%</div>
            <div class="stat-label">Success Rate</div>
        </div>
    `;
    
    // Show/hide toggle button based on result count
    if (totalColumns > 10) {
        toggleBtn.style.display = 'inline-flex';
        // Default to table view for large result sets
        currentView = 'cards';
        toggleResultsView();
    } else {
        toggleBtn.style.display = 'none';
        currentView = 'cards';
        document.getElementById('results-container').style.display = 'block';
        document.getElementById('results-table-view').style.display = 'none';
    }
    
    // Display each result in both card and table formats
    for (const [columnName, data] of Object.entries(extractionResults)) {
        const pageInfo = data.page_number ? `Page ${data.page_number}` : 'Page unknown';
        const modalityInfo = data.modality || 'Unknown';
        
        // Card view
        const resultItem = document.createElement('div');
        resultItem.className = 'result-item';
        
        resultItem.innerHTML = `
            <div class="result-header">
                <div class="result-column">${columnName}</div>
                <div class="result-meta">
                    <span class="meta-badge">📄 ${pageInfo}</span>
                    <span class="meta-badge">📊 ${modalityInfo}</span>
                </div>
            </div>
            <div class="result-value">${escapeHtml(data.value)}</div>
            ${data.evidence ? `<div class="result-evidence"><strong>Evidence:</strong> ${escapeHtml(data.evidence)}</div>` : ''}
            ${data.definition ? `<div class="result-definition"><strong>Definition:</strong> ${escapeHtml(data.definition)}</div>` : ''}
        `;
        
        resultsContainer.appendChild(resultItem);
        
        // Table view
        const tableRow = document.createElement('tr');
        tableRow.innerHTML = `
            <td>${escapeHtml(columnName)}</td>
            <td>${escapeHtml(data.value)}</td>
            <td>${data.page_number || '-'}</td>
            <td>${escapeHtml(modalityInfo)}</td>
            <td>${escapeHtml(data.evidence || '-')}</td>
        `;
        
        resultsTableBody.appendChild(tableRow);
    }
    
    // Show results section
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ==================== Export Results ====================

async function exportResults(format) {
    try {
        const response = await fetch(`/api/export/${format}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ results: extractionResults })
        });
        
        if (format === 'json') {
            const result = await response.json();
            downloadJson(result, 'extraction_results.json');
        } else if (format === 'csv') {
            const csvData = await response.text();
            downloadCsv(csvData, 'extraction_results.csv');
        }
    } catch (error) {
        showError('Export failed: ' + error.message);
    }
}

function downloadJson(data, filename) {
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    downloadBlob(blob, filename);
}

function downloadCsv(data, filename) {
    const blob = new Blob([data], { type: 'text/csv' });
    downloadBlob(blob, filename);
}

function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// ==================== UI Helpers ====================

function toggleButtonLoading(btn, isLoading) {
    const textSpan = btn.querySelector('.btn-text');
    const loaderSpan = btn.querySelector('.btn-loader');
    
    if (isLoading) {
        textSpan.style.display = 'none';
        loaderSpan.style.display = 'inline';
        btn.disabled = true;
    } else {
        textSpan.style.display = 'inline';
        loaderSpan.style.display = 'none';
        btn.disabled = false;
    }
}

function showLoading() {
    document.getElementById('loading-overlay').style.display = 'flex';
}

function hideLoading() {
    document.getElementById('loading-overlay').style.display = 'none';
}

function showError(message) {
    alert('Error: ' + message);
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
