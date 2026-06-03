/**
 * P4H MedBot — Oberhausen  |  app.js
 * Full chat client: REST + SSE streaming, markdown, session management
 */

'use strict';

// ── Config ──────────────────────────────────────────────────────────────────
const API_BASE   = '';          // same-origin (FastAPI serves frontend)
const MAX_CHARS  = 2000;

// ── State ────────────────────────────────────────────────────────────────────
let sessionId   = localStorage.getItem('medbot_session') || generateUUID();
let activeLang  = localStorage.getItem('medbot_lang')    || 'en';
let isStreaming = false;
let currentSSE  = null;

// ── DOM refs ─────────────────────────────────────────────────────────────────
const $messages  = () => document.getElementById('messages');
const $input     = () => document.getElementById('userInput');
const $sendBtn   = () => document.getElementById('sendBtn');
const $sendIcon  = () => document.getElementById('sendIcon');
const $agentInd  = () => document.getElementById('agentIndicator');
const $agentTxt  = () => document.getElementById('agentText');
const $status    = () => document.getElementById('statusDot');
const $charCount = () => document.getElementById('charCount');
const $sessionEl = () => document.getElementById('session-display');
const $sidebar   = () => document.getElementById('sidebar');

// ── Translations ─────────────────────────────────────────────────────────────
const T = {
  en: {
    welcome_title: 'Welcome to P4H MedBot',
    welcome_body:  'I can help you find doctors, understand your health rights, locate hospitals & pharmacies in Oberhausen, and answer medical questions — in German, English, Turkish or Ukrainian.',
    placeholder:   'Ask me anything about health, doctors, or your rights... (DE | EN | TR | UK)',
    thinking:      'Thinking…',
    you:           'You',
    bot:           'MedBot',
    clear_confirm: 'Clear this conversation?',
    error_net:     'Connection error. Is the server running?',
    error_rate:    'Too many messages. Please wait a moment.',
    error_generic: 'Something went wrong. Please try again.',
    agent_labels: {
      emergency:        '🚨 Emergency Agent',
      doctor_search:    '👨‍⚕️ Doctor Search',
      medical_knowledge:'🔬 Medical Knowledge',
      policy_rights:    '📋 Policy & Rights',
      location_maps:    '📍 Location & Maps',
      migrant_health:   '🌍 Migrant Health',
      supervisor:       '🧠 Supervisor',
    }
  },
  de: {
    welcome_title: 'Willkommen beim P4H MedBot',
    welcome_body:  'Ich helfe Ihnen, Ärzte zu finden, Ihre Gesundheitsrechte zu verstehen, Krankenhäuser & Apotheken in Oberhausen zu finden — auf Deutsch, Englisch, Türkisch oder Ukrainisch.',
    placeholder:   'Fragen Sie mich alles über Gesundheit, Ärzte oder Ihre Rechte... (DE | EN | TR | UK)',
    thinking:      'Bitte warten…',
    you:           'Sie',
    bot:           'MedBot',
    clear_confirm: 'Dieses Gespräch löschen?',
    error_net:     'Verbindungsfehler. Läuft der Server?',
    error_rate:    'Zu viele Nachrichten. Bitte warten.',
    error_generic: 'Etwas ist schiefgelaufen. Bitte erneut versuchen.',
    agent_labels: {
      emergency:        '🚨 Notfall-Agent',
      doctor_search:    '👨‍⚕️ Arztsuche',
      medical_knowledge:'🔬 Medizinisches Wissen',
      policy_rights:    '📋 Recht & Versicherung',
      location_maps:    '📍 Standort & Karten',
      migrant_health:   '🌍 Migrantengesundheit',
      supervisor:       '🧠 Supervisor',
    }
  },
  tr: {
    welcome_title: "P4H MedBot'a Hoş Geldiniz",
    welcome_body:  "Doktor bulmak, sağlık haklarınızı öğrenmek, Oberhausen'daki hastane ve eczaneleri bulmak için yardımcı olabilirim — Almanca, İngilizce, Türkçe veya Ukraynaca.",
    placeholder:   'Sağlık, doktorlar veya haklarınız hakkında bir şey sorun... (DE | EN | TR | UK)',
    thinking:      'Düşünüyor…',
    you:           'Siz',
    bot:           'MedBot',
    clear_confirm: 'Bu konuşmayı temizle?',
    error_net:     'Bağlantı hatası. Sunucu çalışıyor mu?',
    error_rate:    'Çok fazla mesaj. Lütfen bekleyin.',
    error_generic: 'Bir şeyler yanlış gitti. Tekrar deneyin.',
    agent_labels: {
      emergency:        '🚨 Acil Durum',
      doctor_search:    '👨‍⚕️ Doktor Arama',
      medical_knowledge:'🔬 Tıbbi Bilgi',
      policy_rights:    '📋 Haklar & Sigorta',
      location_maps:    '📍 Konum & Haritalar',
      migrant_health:   '🌍 Göçmen Sağlığı',
      supervisor:       '🧠 Süpervizör',
    }
  },
  uk: {
    welcome_title: 'Ласкаво просимо до P4H MedBot',
    welcome_body:  'Я допоможу знайти лікарів, зрозуміти ваші права на охорону здоров\'я, знайти лікарні та аптеки в Оберхаузені — німецькою, англійською, турецькою або українською.',
    placeholder:   'Запитайте про здоров\'я, лікарів або ваші права... (DE | EN | TR | UK)',
    thinking:      'Обробка…',
    you:           'Ви',
    bot:           'MedBot',
    clear_confirm: 'Очистити цю розмову?',
    error_net:     'Помилка з\'єднання. Сервер запущено?',
    error_rate:    'Занадто багато повідомлень. Зачекайте.',
    error_generic: 'Щось пішло не так. Спробуйте ще раз.',
    agent_labels: {
      emergency:        '🚨 Екстрена Допомога',
      doctor_search:    '👨‍⚕️ Пошук Лікаря',
      medical_knowledge:'🔬 Медичні Знання',
      policy_rights:    '📋 Права & Страхування',
      location_maps:    '📍 Місцезнаходження',
      migrant_health:   '🌍 Охорона Мігрантів',
      supervisor:       '🧠 Супервізор',
    }
  }
};

function t(key) {
  return (T[activeLang] && T[activeLang][key]) || (T.en[key] || key);
}

function agentLabel(agentKey) {
  return activeLang === 'de' ? '🧠 Supervisor-Agent' :
         activeLang === 'tr' ? '🧠 Süpervizör Ajanı' :
         activeLang === 'uk' ? '🧠 Супевайзер Агент' : '🧠 Supervisor Agent';
}

// ── Init ─────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  // Restore session
  localStorage.setItem('medbot_session', sessionId);
  updateSessionDisplay();

  // Apply saved language
  applyLang(activeLang);

  // Textarea auto-resize on load
  const inp = $input();
  if (inp) {
    inp.addEventListener('input', () => autoResize(inp));
    inp.addEventListener('input', updateCharCount);
  }

  // Check server health
  checkHealth();

  // Load sessions list in sidebar
  loadSessions();

  // Chat continuation: check if current session has history on server
  (async () => {
    try {
      const resp = await fetch(`${API_BASE}/api/chat/history/${sessionId}`);
      if (resp.ok) {
        const data = await resp.json();
        const history = data.history || [];
        if (history.length > 0) {
          removeWelcome();
          const msgs = $messages();
          if (msgs) msgs.innerHTML = '';
          history.forEach(h => {
            if (h.role === 'user') {
              appendUserMsg(h.content);
            } else {
              appendBotMsg({
                text: h.content,
                agent: 'supervisor',
                isEmergency: h.is_emergency
              });
            }
          });
          return;
        }
      }
    } catch (_) {}
    renderWelcome();
  })();

  // Mobile overlay
  document.body.insertAdjacentHTML('beforeend', '<div class="sidebar-overlay" id="sidebarOverlay" onclick="closeSidebar()"></div>');
});

// ── Language ──────────────────────────────────────────────────────────────────
function setLang(lang) {
  activeLang = lang;
  localStorage.setItem('medbot_lang', lang);
  applyLang(lang);
}

function applyLang(lang) {
  // Update active button
  document.querySelectorAll('.lang-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.lang === lang);
  });
  // Update placeholder
  const inp = $input();
  if (inp) inp.placeholder = t('placeholder');
}

// ── UUID ─────────────────────────────────────────────────────────────────────
function generateUUID() {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
    const r = Math.random() * 16 | 0;
    return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
  });
}

function updateSessionDisplay() {
  const el = $sessionEl();
  if (el) el.textContent = 'Session: ' + sessionId.slice(0, 8) + '…';
}

// ── Markdown (minimal, no deps) ───────────────────────────────────────────────
function renderMarkdown(text) {
  if (!text) return '';
  let html = escapeHtml(text);

  // Headers
  html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
  html = html.replace(/^## (.+)$/gm,  '<h2>$1</h2>');
  html = html.replace(/^# (.+)$/gm,   '<h1>$1</h1>');

  // Bold / italic
  html = html.replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>');
  html = html.replace(/\*\*(.+?)\*\*/g,     '<strong>$1</strong>');
  html = html.replace(/\*(.+?)\*/g,         '<em>$1</em>');
  html = html.replace(/__(.+?)__/g,         '<strong>$1</strong>');
  html = html.replace(/_(.+?)_/g,           '<em>$1</em>');

  // Inline code
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

  // Blockquote
  html = html.replace(/^&gt; (.+)$/gm, '<blockquote>$1</blockquote>');

  // HR
  html = html.replace(/^(---|\*\*\*|___)$/gm, '<hr>');

  // Unordered lists
  html = html.replace(/^[-*+] (.+)$/gm, '<li>$1</li>');
  html = html.replace(/(<li>.*<\/li>\n?)+/g, m => '<ul>' + m + '</ul>');

  // Ordered lists
  html = html.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');

  // Links
  html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g,
    '<a href="$2" target="_blank" rel="noopener">$1</a>');

  // Line breaks → paragraphs
  html = html.replace(/\n{2,}/g, '</p><p>');
  html = html.replace(/\n/g, '<br>');
  html = '<p>' + html + '</p>';

  // Fix empty paragraphs
  html = html.replace(/<p><\/p>/g, '');
  html = html.replace(/<p>(<h[1-3]>)/g, '$1');
  html = html.replace(/(<\/h[1-3]>)<\/p>/g, '$1');
  html = html.replace(/<p>(<ul>)/g, '$1');
  html = html.replace(/(<\/ul>)<\/p>/g, '$1');
  html = html.replace(/<p>(<blockquote>)/g, '$1');
  html = html.replace(/(<\/blockquote>)<\/p>/g, '$1');
  html = html.replace(/<p>(<hr>)<\/p>/g, '$1');

  return html;
}

function escapeHtml(str) {
  return str
    .replace(/&/g,  '&amp;')
    .replace(/</g,  '&lt;')
    .replace(/>/g,  '&gt;')
    .replace(/"/g,  '&quot;')
    .replace(/'/g,  '&#39;');
}

// ── Render welcome ────────────────────────────────────────────────────────────
function renderWelcome() {
  const msgs = $messages();
  if (!msgs) return;
  msgs.innerHTML = `
    <div class="welcome-msg">
      <div class="welcome-flags">🇩🇪 🇬🇧 🇹🇷 🇺🇦</div>
      <h2>${t('welcome_title')}</h2>
      <p>${t('welcome_body')}</p>
    </div>`;
}

// ── Message rendering ─────────────────────────────────────────────────────────
function appendUserMsg(text) {
  const msgs = $messages();
  const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  const div  = document.createElement('div');
  div.className = 'msg user';
  div.innerHTML = `
    <div class="msg-avatar">👤</div>
    <div class="msg-body">
      <div class="msg-meta"><span>${escapeHtml(t('you'))}</span><span>${time}</span></div>
      <div class="msg-bubble">${escapeHtml(text)}</div>
    </div>`;
  msgs.appendChild(div);
  scrollBottom();
  return div;
}

function appendBotMsg({ text = '', agent = '', sources = [], isEmergency = false, streaming = false }) {
  const msgs = $messages();
  const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  const div  = document.createElement('div');
  div.className = 'msg bot' + (isEmergency ? ' emergency' : '');

  const agentBadge = agent
    ? `<span class="msg-agent-badge">${agentLabel(agent)}</span>`
    : '';

  const sourcesHtml = sources && sources.length
    ? `<div class="msg-sources">${sources.map(s =>
        `<span class="source-chip">📎 ${escapeHtml(s)}</span>`).join('')}</div>`
    : '';

  const cursor = streaming ? '<span class="stream-cursor"></span>' : '';

  div.innerHTML = `
    <div class="msg-avatar">🏥</div>
    <div class="msg-body">
      <div class="msg-meta">${agentBadge}<span>${escapeHtml(t('bot'))}</span><span>${time}</span></div>
      <div class="msg-bubble" id="bot-bubble-live">${renderMarkdown(text)}${cursor}</div>
      <div class="msg-sources-wrap">${sourcesHtml}</div>
    </div>`;

  msgs.appendChild(div);
  scrollBottom();
  return div;
}

function appendTypingIndicator() {
  const msgs = $messages();
  const div  = document.createElement('div');
  div.className = 'msg bot typing-indicator';
  div.id = 'typing-indicator';
  div.innerHTML = `
    <div class="msg-avatar">🏥</div>
    <div class="msg-body">
      <div class="msg-bubble">
        <div class="typing-dots"><span></span><span></span><span></span></div>
      </div>
    </div>`;
  msgs.appendChild(div);
  scrollBottom();
  return div;
}

function removeTypingIndicator() {
  const el = document.getElementById('typing-indicator');
  if (el) el.remove();
}

function scrollBottom() {
  const msgs = $messages();
  if (msgs) msgs.scrollTop = msgs.scrollHeight;
}

// ── Input handlers ────────────────────────────────────────────────────────────
function handleKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
}

function autoResize(ta) {
  ta.style.height = 'auto';
  ta.style.height = Math.min(ta.scrollHeight, 160) + 'px';
}

function updateCharCount() {
  const inp = $input();
  const cc  = $charCount();
  if (!inp || !cc) return;
  const len = inp.value.length;
  cc.textContent = `${len} / ${MAX_CHARS}`;
  cc.className   = 'char-count' + (len > 1800 ? ' warn' : '') + (len >= MAX_CHARS ? ' over' : '');
}

// ── Send ──────────────────────────────────────────────────────────────────────
async function sendMessage() {
  const inp  = $input();
  if (!inp) return;
  const text = inp.value.trim();
  if (!text || isStreaming) return;

  // Clear input
  inp.value = '';
  autoResize(inp);
  updateCharCount();

  // Append user bubble
  removeWelcome();
  appendUserMsg(text);

  // Check mode
  const useStream = document.getElementById('streamMode')?.checked;

  if (useStream) {
    await sendStreaming(text);
  } else {
    await sendRest(text);
  }
}

function removeWelcome() {
  const w = document.querySelector('.welcome-msg');
  if (w) w.remove();
}

// ── REST send ─────────────────────────────────────────────────────────────────
async function sendRest(text) {
  setLoading(true);
  appendTypingIndicator();
  try {
    const resp = await fetch(`${API_BASE}/api/chat/message`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ message: text, session_id: sessionId })
    });

    removeTypingIndicator();

    if (resp.status === 429) { showToast(t('error_rate'), 'error'); return; }
    if (!resp.ok)            { showToast(t('error_generic'), 'error'); return; }

    const data = await resp.json();

    // Update session
    if (data.session_id) {
      sessionId = data.session_id;
      localStorage.setItem('medbot_session', sessionId);
      updateSessionDisplay();
    }

    appendBotMsg({
      text:        data.response,
      agent:       data.agent,
      sources:     data.sources,
      isEmergency: data.is_emergency,
    });

    showAgentIndicator(data.agent, false);
    loadSessions();

  } catch (err) {
    removeTypingIndicator();
    showToast(t('error_net'), 'error');
    console.error('[MedBot REST]', err);
  } finally {
    setLoading(false);
  }
}

// ── SSE streaming send ────────────────────────────────────────────────────────
async function sendStreaming(text) {
  setLoading(true);
  showAgentIndicator('supervisor', true);

  // Build streaming bot bubble
  const botDiv = appendBotMsg({ text: '', agent: 'supervisor', streaming: true });
  const bubble = botDiv.querySelector('#bot-bubble-live');

  let accumulated = '';
  let metaData    = {};

  try {
    // Close any existing SSE
    if (currentSSE) { currentSSE.close(); currentSSE = null; }

    const url = `${API_BASE}/api/stream/chat?message=${encodeURIComponent(text)}&session_id=${encodeURIComponent(sessionId)}`;
    const es  = new EventSource(url);
    currentSSE = es;

    es.onmessage = () => {};   // default ignored

    es.addEventListener('token', e => {
      try {
        const d = JSON.parse(e.data);
        accumulated += (d.token || '');
        if (bubble) {
          bubble.innerHTML = renderMarkdown(accumulated) + '<span class="stream-cursor"></span>';
          scrollBottom();
        }
      } catch (_) {}
    });

    es.addEventListener('agent', e => {
      try {
        const d = JSON.parse(e.data);
        if (d.agent) showAgentIndicator(d.agent, true);
        // Update badge
        const badge = botDiv.querySelector('.msg-agent-badge');
        if (badge && d.agent) badge.textContent = agentLabel(d.agent);
      } catch (_) {}
    });

    es.addEventListener('done', e => {
      try { metaData = JSON.parse(e.data); } catch (_) {}
      es.close();
      currentSSE = null;
      finishStream(bubble, accumulated, metaData, botDiv);
      setLoading(false);
      showAgentIndicator(metaData.agent, false);
    });

    es.addEventListener('error_event', e => {
      try {
        const d = JSON.parse(e.data);
        showToast(d.message || t('error_generic'), 'error');
      } catch (_) { showToast(t('error_generic'), 'error'); }
      es.close();
      currentSSE = null;
      setLoading(false);
    });

    es.onerror = () => {
      if (es.readyState === EventSource.CLOSED) return;
      showToast(t('error_net'), 'error');
      es.close();
      currentSSE = null;
      setLoading(false);
    };

  } catch (err) {
    showToast(t('error_net'), 'error');
    setLoading(false);
    console.error('[MedBot SSE]', err);
  }
}

function finishStream(bubble, text, meta, msgDiv) {
  // Remove cursor
  if (bubble) {
    bubble.id = '';   // remove live id
    bubble.innerHTML = renderMarkdown(text);
  }

  // Update session
  if (meta.session_id) {
    sessionId = meta.session_id;
    localStorage.setItem('medbot_session', sessionId);
    updateSessionDisplay();
  }

  // Add sources
  if (meta.sources && meta.sources.length) {
    const wrap = msgDiv.querySelector('.msg-sources-wrap');
    if (wrap) {
      wrap.innerHTML = `<div class="msg-sources">${
        meta.sources.map(s => `<span class="source-chip">📎 ${escapeHtml(s)}</span>`).join('')
      }</div>`;
    }
  }

  // Emergency class
  if (meta.is_emergency) {
    msgDiv.classList.add('emergency');
  }

  scrollBottom();
  loadSessions();
}

// ── Quick message ─────────────────────────────────────────────────────────────
function quickMessage(text) {
  const inp = $input();
  if (!inp) return;
  inp.value = text;
  autoResize(inp);
  updateCharCount();
  closeSidebar();
  sendMessage();
}

// ── Session Management (New, List, Select, Delete, Clear) ───────────────────

async function loadSessions() {
  const listEl = document.getElementById('recentChatsList');
  if (!listEl) return;
  try {
    const resp = await fetch(`${API_BASE}/api/chat/sessions`);
    if (!resp.ok) return;
    const data = await resp.json();
    const sessions = data.sessions || [];
    
    listEl.innerHTML = '';
    if (sessions.length === 0) {
      listEl.innerHTML = '<div style="font-size:0.75rem;color:rgba(255,255,255,0.3);text-align:center;padding:10px 0;">No recent chats</div>';
      return;
    }
    
    sessions.forEach(s => {
      const item = document.createElement('div');
      item.className = 'recent-chat-item' + (s.session_id === sessionId ? ' active' : '');
      item.dataset.sessionId = s.session_id;
      item.onclick = () => selectSession(s.session_id);
      
      const title = document.createElement('span');
      title.className = 'recent-chat-title';
      title.textContent = s.title || 'Conversation';
      title.title = s.last_message || '';
      
      const delBtn = document.createElement('button');
      delBtn.className = 'recent-chat-delete';
      delBtn.innerHTML = '🗑️';
      delBtn.onclick = (e) => {
        e.stopPropagation();
        deleteSession(s.session_id);
      };
      
      item.appendChild(title);
      item.appendChild(delBtn);
      listEl.appendChild(item);
    });
  } catch (err) {
    console.error('[MedBot loadSessions]', err);
  }
}

async function startNewChat() {
  try {
    const resp = await fetch(`${API_BASE}/api/chat/sessions`, { method: 'POST' });
    if (resp.ok) {
      const data = await resp.json();
      if (data.session_id) {
        sessionId = data.session_id;
        localStorage.setItem('medbot_session', sessionId);
        updateSessionDisplay();
        renderWelcome();
        hideAgentIndicator();
        loadSessions();
        const msgs = $messages();
        if (msgs) msgs.innerHTML = '';
        renderWelcome();
        return;
      }
    }
  } catch (_) {}
  
  // Local fallback
  sessionId = generateUUID();
  localStorage.setItem('medbot_session', sessionId);
  updateSessionDisplay();
  const msgs = $messages();
  if (msgs) msgs.innerHTML = '';
  renderWelcome();
  hideAgentIndicator();
  loadSessions();
}

async function selectSession(sid) {
  if (isStreaming) return;
  sessionId = sid;
  localStorage.setItem('medbot_session', sessionId);
  updateSessionDisplay();
  
  // Highlight active in list
  document.querySelectorAll('.recent-chat-item').forEach(item => {
    item.classList.toggle('active', item.dataset.sessionId === sid);
  });
  
  const msgs = $messages();
  if (!msgs) return;
  msgs.innerHTML = '';
  appendTypingIndicator();
  
  try {
    const resp = await fetch(`${API_BASE}/api/chat/history/${sessionId}`);
    removeTypingIndicator();
    if (!resp.ok) {
      renderWelcome();
      return;
    }
    const data = await resp.json();
    const history = data.history || [];
    
    if (history.length === 0) {
      renderWelcome();
    } else {
      removeWelcome();
      history.forEach(h => {
        if (h.role === 'user') {
          appendUserMsg(h.content);
        } else {
          appendBotMsg({
            text: h.content,
            agent: 'supervisor',
            isEmergency: h.is_emergency
          });
        }
      });
    }
  } catch (err) {
    removeTypingIndicator();
    renderWelcome();
    console.error('[MedBot selectSession]', err);
  }
  loadSessions();
}

async function deleteSession(sid) {
  if (!confirm(t('clear_confirm'))) return;
  try {
    const resp = await fetch(`${API_BASE}/api/chat/session/${sid}`, { method: 'DELETE' });
    if (resp.ok) {
      showToast('Chat deleted', 'success');
      if (sid === sessionId) {
        await startNewChat();
      } else {
        loadSessions();
      }
    }
  } catch (err) {
    console.error('[MedBot deleteSession]', err);
  }
}

async function clearChat() {
  if (!confirm(t('clear_confirm'))) return;
  try {
    await fetch(`${API_BASE}/api/chat/session/${sessionId}/clear`, { method: 'POST' });
    showToast('Chat cleared', 'success');
    const msgs = $messages();
    if (msgs) msgs.innerHTML = '';
    renderWelcome();
    hideAgentIndicator();
    loadSessions();
  } catch (err) {
    console.error('[MedBot clearChat]', err);
  }
}

// ── UI helpers ────────────────────────────────────────────────────────────────
function setLoading(on) {
  isStreaming = on;
  const btn  = $sendBtn();
  const icon = $sendIcon();
  if (!btn || !icon) return;
  btn.disabled = on;
  if (on) {
    icon.outerHTML = '<div class="spinner" id="sendIcon"></div>';
  } else {
    const spinner = document.getElementById('sendIcon');
    if (spinner) spinner.outerHTML = '<span id="sendIcon">➤</span>';
  }
}

function showAgentIndicator(agent, active) {
  const ind = $agentInd();
  const txt = $agentTxt();
  if (!ind || !txt) return;
  if (active) {
    txt.textContent = agentLabel(agent) + ' …';
    ind.style.display = 'flex';
  } else {
    txt.textContent = agentLabel(agent);
  }
}

function hideAgentIndicator() {
  const ind = $agentInd();
  if (ind) ind.style.display = 'none';
}

function showToast(msg, type = '') {
  const t = document.createElement('div');
  t.className = 'toast' + (type ? ' ' + type : '');
  t.textContent = msg;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 4000);
}

// ── Sidebar (mobile) ──────────────────────────────────────────────────────────
function toggleSidebar() {
  const s  = $sidebar();
  const ov = document.getElementById('sidebarOverlay');
  if (!s) return;
  const open = s.classList.toggle('open');
  if (ov) ov.classList.toggle('show', open);
}

function closeSidebar() {
  const s  = $sidebar();
  const ov = document.getElementById('sidebarOverlay');
  if (s)  s.classList.remove('open');
  if (ov) ov.classList.remove('show');
}

// ── Health check ──────────────────────────────────────────────────────────────
async function checkHealth() {
  const dot = $status();
  if (!dot) return;
  try {
    const r = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(5000) });
    if (r.ok) {
      dot.className = 'status-dot online';
      dot.title = 'System online';
    } else {
      dot.className = 'status-dot busy';
      dot.title = 'Degraded';
    }
  } catch {
    dot.className = 'status-dot offline';
    dot.title = 'Server offline';
  }

  // Re-check every 30s
  setTimeout(checkHealth, 30_000);
}
