document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const uploadForm = document.getElementById("uploadForm");
    const uploadArea = document.getElementById("uploadArea");
    const fileInput = document.getElementById("pdfFile");
    const uploadBtn = document.getElementById("uploadBtn");
    const uploadStatus = document.getElementById("uploadStatus");
    const statusText = document.getElementById("statusText");
    const progressBar = document.getElementById("progressBar");
    const progressBarFill = document.getElementById("progressBarFill");
    const currentFile = document.getElementById("currentFile");
    const fileName = document.getElementById("fileName");
    const techniqueSelect = document.getElementById("techniqueSelect");
    const techniqueInfo = document.getElementById("techniqueInfo");
    const questionInput = document.getElementById("questionInput");
    const askBtn = document.getElementById("askBtn");
    const messagesArea = document.getElementById("messagesArea");
    const messagesContainer = document.getElementById("messagesContainer");
    const welcomeMessage = document.getElementById("welcomeMessage");
    const connectionStatus = document.getElementById("connectionStatus");
    const quickQuestions = document.querySelectorAll(".quick-question");

    // Technique descriptions
    const techniqueDescriptions = {
        "auto": "Automatically selects the best technique based on question complexity",
        "decomposition": "Breaks complex questions into simpler sub-questions and synthesizes answers",
        "cot": "Uses step-by-step reasoning to answer questions",
        "graph": "Uses knowledge graph and entity relationships for reasoning",
        "basic": "Standard retrieval and generation approach"
    };

    // Initialize
    let isUploading = false;
    let currentSession = null;

    // Check for existing session
    const storedPDF = sessionStorage.getItem("currentPDF");
    if (storedPDF) {
        enableChat();
        showCurrentFile(storedPDF);
    }

    // Upload Area Drag & Drop
    uploadArea.addEventListener("click", () => fileInput.click());
    
    uploadArea.addEventListener("dragover", (e) => {
        e.preventDefault();
        uploadArea.classList.add("dragover");
    });
    
    uploadArea.addEventListener("dragleave", () => {
        uploadArea.classList.remove("dragover");
    });
    
    uploadArea.addEventListener("drop", (e) => {
        e.preventDefault();
        uploadArea.classList.remove("dragover");
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            handleFileSelect();
        }
    });

    // File Input Change
    fileInput.addEventListener("change", handleFileSelect);

    function handleFileSelect() {
        const file = fileInput.files[0];
        if (file) {
            if (file.type !== "application/pdf") {
                showAlert("Please select a PDF file.", "warning");
                return;
            }
            if (file.size > 25 * 1024 * 1024) {
                showAlert("File size exceeds 25MB limit.", "warning");
                return;
            }
            uploadBtn.disabled = false;
            uploadArea.innerHTML = `
                <i class="bi bi-file-earmark-pdf text-primary fs-1 mb-3"></i>
                <p class="text-primary mb-2 fw-semibold">${file.name}</p>
                <small class="text-muted">Ready to upload</small>
            `;
        }
    }

    // Upload Form Submit
    uploadForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        if (isUploading) return;

        const file = fileInput.files[0];
        if (!file) {
            showAlert("Please select a PDF file.", "warning");
            return;
        }

        isUploading = true;
        uploadBtn.disabled = true;
        uploadBtn.innerHTML = '<i class="bi bi-hourglass-split me-2"></i>Uploading...';
        
        showAlert("Uploading and indexing document...", "info");
        progressBar.classList.remove("d-none");
        progressBarFill.style.width = "0%";

        const formData = new FormData();
        formData.append("file", file);

        try {
            const response = await fetch("/upload", {
                method: "POST",
                body: formData
            });

            const result = await response.text();
            
            if (response.ok) {
                showAlert("Document uploaded and indexed successfully!", "success");
                sessionStorage.setItem("currentPDF", file.name);
                enableChat();
                showCurrentFile(file.name);
                clearUploadArea();
            } else {
                showAlert(result, "danger");
            }
        } catch (error) {
            showAlert("Upload failed. Please try again.", "danger");
            console.error("Upload error:", error);
        } finally {
            isUploading = false;
            uploadBtn.disabled = false;
            uploadBtn.innerHTML = '<i class="bi bi-cloud-upload me-2"></i>Upload & Index';
            progressBar.classList.add("d-none");
        }
    });

    // Technique Selection
    techniqueSelect.addEventListener("change", () => {
        const selectedTechnique = techniqueSelect.value;
        techniqueInfo.textContent = techniqueDescriptions[selectedTechnique];
    });

    // Quick Questions
    quickQuestions.forEach(btn => {
        btn.addEventListener("click", () => {
            const question = btn.dataset.question;
            questionInput.value = question;
            askQuestion();
        });
    });

    // Question Input - Enhanced auto-resize and scrolling
    questionInput.addEventListener("input", () => {
        askBtn.disabled = !questionInput.value.trim() || !sessionStorage.getItem("currentPDF");
        
        // Enhanced auto-resize textarea
        autoResizeTextarea();
    });

    questionInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            if (!askBtn.disabled) {
                askQuestion();
            }
        }
    });

    // Auto-resize textarea function
    function autoResizeTextarea() {
        questionInput.style.height = 'auto';
        const newHeight = Math.min(questionInput.scrollHeight, 120);
        questionInput.style.height = newHeight + 'px';
        
        // Adjust input container padding if needed
        const inputContainer = questionInput.closest('.chat-input-container');
        if (inputContainer) {
            const currentPadding = parseInt(window.getComputedStyle(inputContainer).paddingBottom);
            if (newHeight > 40) {
                inputContainer.style.paddingBottom = '1rem';
            } else {
                inputContainer.style.paddingBottom = '1.5rem';
            }
        }
    }

    // Smooth scroll to bottom function
    function scrollToBottom(smooth = true) {
        if (messagesArea) {
            const scrollOptions = {
                top: messagesArea.scrollHeight,
                behavior: smooth ? 'smooth' : 'auto'
            };
            messagesArea.scrollTo(scrollOptions);
        }
    }

    // Ask Question Function
    window.askQuestion = async function() {
        const question = questionInput.value.trim();
        if (!question || !sessionStorage.getItem("currentPDF")) return;

        // Add user message
        addMessage(question, "user");
        questionInput.value = "";
        autoResizeTextarea();
        askBtn.disabled = true;

        // Add thinking indicator
        const thinkingId = addThinkingIndicator();

        try {
            const response = await fetch("/ask-question/", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    question: question,
                    technique: techniqueSelect.value
                })
            });

            const data = await response.json();
            removeThinkingIndicator(thinkingId);

            if (data.error) {
                addMessage(`Error: ${data.error}`, "assistant", "error");
            } else {
                addMessage(data.answer, "assistant", data.technique, data);
            }
        } catch (error) {
            removeThinkingIndicator(thinkingId);
            addMessage("Failed to get response. Please try again.", "assistant", "error");
            console.error("Question error:", error);
        }
    };

    // Helper Functions
    function enableChat() {
        welcomeMessage.style.display = "none";
        askBtn.disabled = false;
        connectionStatus.innerHTML = '<i class="bi bi-circle-fill me-1"></i>Connected';
        connectionStatus.className = 'badge bg-success me-2';
    }

    function showCurrentFile(name) {
        currentFile.classList.remove("d-none");
        fileName.textContent = name;
    }

    function clearUploadArea() {
        uploadArea.innerHTML = `
            <i class="bi bi-file-earmark-pdf text-muted fs-1 mb-3"></i>
            <p class="text-muted mb-2">Drop your PDF here or click to browse</p>
            <small class="text-muted">Max size: 25MB</small>
        `;
        fileInput.value = "";
        uploadBtn.disabled = true;
    }

    function showAlert(message, type) {
        uploadStatus.className = `alert alert-${type} d-block`;
        statusText.textContent = message;
        
        setTimeout(() => {
            uploadStatus.classList.add("d-none");
        }, 5000);
    }

    function addMessage(content, sender, technique = null, data = null) {
        const messageDiv = document.createElement("div");
        messageDiv.className = `message ${sender}`;
        
        const bubbleClass = technique ? `message-bubble ${getTechniqueClass(technique)}` : "message-bubble";
        
        let messageContent = `<div class="${bubbleClass}">${content}</div>`;
        
        // Add technique badge for assistant messages
        if (sender === "assistant" && technique && technique !== "error") {
            messageContent = `<div class="technique-badge ${getTechniqueClass(technique)}">${technique}</div>` + messageContent;
        }
        
        // Add sub-questions for decomposition
        if (data && data.sub_questions && data.sub_questions.length > 0) {
            messageContent += `
                <div class="sub-questions">
                    <h6>Sub-questions analyzed:</h6>
                    <ul>
                        ${data.sub_questions.map((q, i) => `<li>${i + 1}. ${q}</li>`).join('')}
                    </ul>
                </div>
            `;
        }
        
        // Add graph entities for graph reasoning
        if (data && data.graph_entities && data.graph_entities.length > 0) {
            messageContent += `
                <div class="graph-entities">
                    <strong>Knowledge Graph Entities:</strong> ${data.graph_entities.join(', ')}
                </div>
            `;
        }
        
        // Add sources
        if (data && data.sources && data.sources.length > 0) {
            messageContent += `
                <div class="sources-list">
                    <h6>Sources:</h6>
                    <ul>
                        ${data.sources.map(src => `<li>${src}</li>`).join('')}
                    </ul>
                </div>
            `;
        }
        
        // Add metadata
        const now = new Date();
        const timeString = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        messageContent += `<div class="message-meta">${timeString}</div>`;
        
        messageDiv.innerHTML = messageContent;
        messagesArea.appendChild(messageDiv);
        
        // Enhanced smooth scroll to bottom with delay for animation
        setTimeout(() => {
            scrollToBottom(true);
        }, 150);
    }

    function addThinkingIndicator() {
        const thinkingDiv = document.createElement("div");
        thinkingDiv.className = "message assistant";
        thinkingDiv.id = "thinking-" + Date.now();
        thinkingDiv.innerHTML = `
            <div class="message-bubble">
                <div class="thinking-indicator">
                    <div class="spinner-border text-primary" role="status"></div>
                    <span>Thinking...</span>
                </div>
            </div>
        `;
        messagesArea.appendChild(thinkingDiv);
        
        // Scroll to show thinking indicator
        setTimeout(() => {
            scrollToBottom(true);
        }, 100);
        
        return thinkingDiv.id;
    }

    function removeThinkingIndicator(id) {
        const thinkingDiv = document.getElementById(id);
        if (thinkingDiv) {
            thinkingDiv.remove();
        }
    }

    function getTechniqueClass(technique) {
        const techniqueMap = {
            "LLM Decomposition": "decomposition",
            "Chain-of-Thought": "cot",
            "Graph Reasoning": "graph",
            "Basic RAG": "basic",
            "Basic RAG (Fallback)": "basic"
        };
        return techniqueMap[technique] || "basic";
    }

    // Initialize technique info
    techniqueInfo.textContent = techniqueDescriptions[techniqueSelect.value];

    // Initialize textarea height
    autoResizeTextarea();

    // Add scroll event listener for better UX
    messagesArea.addEventListener('scroll', () => {
        // Optional: Add scroll-based effects here
        const isAtBottom = messagesArea.scrollTop + messagesArea.clientHeight >= messagesArea.scrollHeight - 10;
        
        // You can add visual indicators or effects based on scroll position
        if (isAtBottom) {
            // User is at bottom - could show "new message" indicator if they scroll up
        }
    });

    // Handle window resize for responsive behavior
    window.addEventListener('resize', () => {
        autoResizeTextarea();
        // Ensure proper scrolling after resize
        setTimeout(() => {
            scrollToBottom(false);
        }, 100);
    });
});
