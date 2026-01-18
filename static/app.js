/**
 * 智慧交通诱导智能体 - 前端交互逻辑
 */

// ===== DOM Elements =====
const chatMessages = document.getElementById('chatMessages');
const chatInput = document.getElementById('chatInput');
const sendBtn = document.getElementById('sendBtn');
const debugToggle = document.getElementById('debugToggle');
const debugPanel = document.getElementById('debugPanel');
const welcomeMessage = document.getElementById('welcomeMessage');
const logsTab = document.getElementById('logsTab');
const stateTab = document.getElementById('stateTab');
const stateJson = document.getElementById('stateJson');
const debugTabs = document.querySelectorAll('.debug-tab');
const examplePrompts = document.querySelectorAll('.example-prompt');

// ===== State =====
let isProcessing = false;

// ===== Event Listeners =====

// 发送按钮点击
sendBtn.addEventListener('click', sendMessage);

// 回车发送 (Shift+Enter 换行)
chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// 自动调整输入框高度
chatInput.addEventListener('input', () => {
    chatInput.style.height = 'auto';
    chatInput.style.height = Math.min(chatInput.scrollHeight, 200) + 'px';
});

// 调试面板切换
debugToggle.addEventListener('click', () => {
    debugPanel.classList.toggle('hidden');
    debugToggle.classList.toggle('active');
});

// 调试面板 Tab 切换
debugTabs.forEach(tab => {
    tab.addEventListener('click', () => {
        const tabName = tab.dataset.tab;
        
        // 更新 Tab 状态
        debugTabs.forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        
        // 切换内容
        if (tabName === 'logs') {
            logsTab.classList.remove('hidden');
            stateTab.classList.add('hidden');
        } else {
            logsTab.classList.add('hidden');
            stateTab.classList.remove('hidden');
        }
    });
});

// 示例提示词点击
examplePrompts.forEach(btn => {
    btn.addEventListener('click', () => {
        const prompt = btn.dataset.prompt;
        chatInput.value = prompt;
        chatInput.focus();
        sendMessage();
    });
});

// ===== Functions =====

/**
 * 发送消息
 */
async function sendMessage() {
    const message = chatInput.value.trim();
    if (!message || isProcessing) return;
    
    // 清空输入框
    chatInput.value = '';
    chatInput.style.height = 'auto';
    
    // 隐藏欢迎消息
    if (welcomeMessage) {
        welcomeMessage.style.display = 'none';
    }
    
    // 添加用户消息
    addMessage(message, 'user');
    
    // 添加 AI 思考中占位
    const thinkingId = addThinkingMessage();
    
    // 清空调试日志
    clearDebugLogs();
    
    // 设置处理状态
    isProcessing = true;
    sendBtn.disabled = true;
    
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message })
        });
        
        const data = await response.json();
        
        // 移除思考中占位
        removeThinkingMessage(thinkingId);
        
        if (data.success) {
            // 添加 AI 回复 (打字机效果)
            await addMessageWithTyping(data.recommendation, 'assistant');
            
            // 更新调试信息
            updateDebugLogs(data.debug_logs || []);
            updateStateJson(data.state || {});
        } else {
            addMessage('抱歉，处理请求时发生错误: ' + (data.error || '未知错误'), 'assistant');
        }
    } catch (error) {
        removeThinkingMessage(thinkingId);
        addMessage('抱歉，网络请求失败。请检查服务是否正常运行。', 'assistant');
        console.error('Chat error:', error);
    } finally {
        isProcessing = false;
        sendBtn.disabled = false;
        chatInput.focus();
    }
}

/**
 * 添加消息到聊天区域
 */
function addMessage(content, role) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    
    const avatar = role === 'user' ? '👤' : '🤖';
    
    messageDiv.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-content">
            <p>${escapeHtml(content)}</p>
        </div>
    `;
    
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

/**
 * 添加思考中占位消息
 */
function addThinkingMessage() {
    const id = 'thinking-' + Date.now();
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    messageDiv.id = id;
    
    messageDiv.innerHTML = `
        <div class="message-avatar">🤖</div>
        <div class="message-content">
            <div class="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
    `;
    
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
    
    return id;
}

/**
 * 移除思考中占位消息
 */
function removeThinkingMessage(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

/**
 * 添加消息 (打字机效果 + Markdown 渲染)
 */
async function addMessageWithTyping(content, role) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    
    const avatar = role === 'user' ? '👤' : '🤖';
    
    messageDiv.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-content markdown-body">
        </div>
    `;
    
    chatMessages.appendChild(messageDiv);
    const contentEl = messageDiv.querySelector('.message-content');
    
    // 打字机效果：逐步显示内容
    const chars = content.split('');
    const batchSize = 5; // 每次渲染多少字符
    let currentText = '';
    
    for (let i = 0; i < chars.length; i += batchSize) {
        const batch = chars.slice(i, i + batchSize).join('');
        currentText += batch;
        // 使用 marked 渲染 Markdown
        contentEl.innerHTML = marked.parse(currentText);
        scrollToBottom();
        await sleep(8);
    }
    
    // 最终完整渲染，确保格式正确
    contentEl.innerHTML = marked.parse(content);
    scrollToBottom();
}

/**
 * 更新调试日志
 */
function updateDebugLogs(logs) {
    logsTab.innerHTML = '';
    
    if (!logs || logs.length === 0) {
        logsTab.innerHTML = '<p style="color: var(--text-muted); text-align: center; padding: 1rem;">无日志</p>';
        return;
    }
    
    logs.forEach(log => {
        const entry = document.createElement('div');
        const logType = log.type || 'unknown';
        let cssClass = '';
        let icon = '📋';
        let title = logType;
        
        switch (logType) {
            case 'llm_response':
                cssClass = 'llm';
                icon = '🤖';
                title = 'LLM 响应';
                break;
            case 'tool_execution':
                cssClass = 'tool';
                icon = '🛠️';
                title = '工具执行';
                break;
            case 'reflection':
                cssClass = 'reflection';
                icon = '🤔';
                title = '反思评估';
                break;
            case 'tool_error':
                cssClass = 'error';
                icon = '❌';
                title = '工具错误';
                break;
            case 'perception':
                cssClass = 'tool';
                icon = '🔍';
                title = '感知';
                break;
            case 'final_output':
                cssClass = 'llm';
                icon = '📄';
                title = '最终输出';
                break;
            default:
                cssClass = '';
                icon = 'ℹ️';
        }
        
        entry.className = `log-entry ${cssClass}`;
        entry.innerHTML = `
            <div class="log-header">
                <span class="log-icon">${icon}</span>
                <span>${title}</span>
                <span class="log-time">${log.timestamp || ''}</span>
            </div>
            <div class="log-body">${formatLogContent(log.content)}</div>
        `;
        
        logsTab.appendChild(entry);
    });
}

/**
 * 格式化日志内容
 */
function formatLogContent(content) {
    if (!content) return '';
    if (typeof content === 'string') return escapeHtml(content);
    
    try {
        return escapeHtml(JSON.stringify(content, null, 2));
    } catch {
        return escapeHtml(String(content));
    }
}

/**
 * 更新状态 JSON
 */
function updateStateJson(state) {
    stateJson.textContent = JSON.stringify(state, null, 2);
}

/**
 * 清空调试日志
 */
function clearDebugLogs() {
    logsTab.innerHTML = '<p style="color: var(--text-muted); text-align: center; padding: 1rem;">正在处理...</p>';
    stateJson.textContent = '{ "status": "处理中..." }';
}

/**
 * 滚动到底部
 */
function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

/**
 * HTML 转义
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * 延时函数
 */
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// ===== Init =====
chatInput.focus();
